import concurrent.futures
from collections import deque
from threading import Event
from time import sleep
from typing import Any, Callable

import logfire

from ai_sub.config import Settings
from ai_sub.data_models import Job, JobState


class JobRunner:
    """
    A generic, concurrent job processor.

    This class provides a framework for processing `JobState` objects from a
    deque in a thread-safe manner using a `ThreadPoolExecutor`. It handles job
    acquisition, retries on failure, and graceful shutdown.

    Subclasses must implement the `process` method to define the actual work.
    The `on_complete` callback can be used to chain dependent jobs.
    """

    def __init__(
        self,
        queue: deque[JobState],
        settings: Settings,
        max_workers: int,
        on_complete: Callable[[JobState, Any], None] | None = None,
        stop_events: list[Event] | None = None,
        name: str = "JobRunner",
    ):
        """Initializes the JobRunner.

        Args:
            queue (deque[JobState]): The queue from which to pull job states for processing.
            settings (Settings): The application's configuration settings.
            max_workers (int): The maximum number of threads to use for concurrent processing.
            on_complete (Callable[[JobState, Any], None] | None): An optional callback
                function that is executed upon successful completion of a job. It receives
                the job state and the result of the `process` method.
            stop_events (list[Event] | None): A list of threading.Event objects. The runner's
                worker threads will gracefully exit when the queue is empty and all of these
                events are set, indicating that no more jobs will be added.
            name (str): The name of the runner. This name is crucial as it's used to
                dynamically access the corresponding job attribute from the `JobState` object
                (e.g., a runner with name 'reencode' will process `job_state.reencode`).
        """
        self.queue = queue
        self.settings = settings
        self.max_workers = max_workers
        self.on_complete = on_complete
        self.stop_events = stop_events or []
        self.name = name
        self.executor: concurrent.futures.ThreadPoolExecutor | None = None
        self.futures: list[concurrent.futures.Future] = []

    def start(self) -> None:
        """Starts the worker threads."""
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        self.futures = [self.executor.submit(self.run) for _ in range(self.max_workers)]

    def wait(self) -> None:
        """Waits for all worker threads to complete."""
        if self.futures:
            concurrent.futures.wait(self.futures)

    def shutdown(self) -> None:
        """Shuts down the executor."""
        if self.executor:
            self.executor.shutdown()

    def run(self) -> None:
        """
        The main worker loop that processes jobs from the queue.

        This method runs in a continuous loop on each worker thread. It attempts to
        pop a `JobState` from the queue. If the queue is empty, it checks the
        `stop_events` to determine whether to exit or wait for more jobs.

        For each `JobState`, it:
        1. Uses `get_job()` to retrieve the specific `Job` object for this runner.
        2. Increments the job's retry counters.
        3. Calls the `process()` method to perform the work.
        4. On success, calls the `on_complete` callback if it exists.
        5. On failure, calls `_handle_retry()` to potentially re-queue the job.
        6. Always calls `post_process()` for any cleanup tasks.
        """
        while True:
            job_state: JobState | None = None
            current_job: Job | None = None

            try:
                # Attempt to get a JobState container from the left of the queue.
                job_state = self.queue.popleft()

                # Get the specific job for this runner from the JobState container.
                current_job = self.get_job(job_state)

                # Increment retry counters for the specific job.
                current_job.run_num_retries += 1
                current_job.total_num_retries += 1

            except IndexError:
                # The queue is empty. Check if we should wait for more jobs or exit.
                if self.stop_events:
                    if all(e.is_set() for e in self.stop_events):
                        # All upstream runners are done, so no more jobs will be added.
                        break
                    else:
                        # At least one upstream runner is still active, so wait for more jobs.
                        sleep(1)
                        continue
                else:
                    # No stop events are configured, so an empty queue means we are done.
                    break

            with logfire.span(f"Executing {self.name} job for {current_job.name}"):
                try:
                    result = self.process(job_state)
                    if self.on_complete:
                        self.on_complete(job_state, result)

                except Exception:
                    job_name = f"'{current_job.name}'" if current_job else "unknown"
                    logfire.exception(
                        f"Exception while running {self.name} job {job_name}"
                    )
                    if job_state is not None and current_job is not None:
                        self._handle_retry(job_state, current_job)

                finally:
                    if job_state is not None:
                        try:
                            self.post_process(job_state)
                        except Exception:
                            logfire.exception(
                                f"Exception in post_process for {self.name} job"
                            )

    def get_job(self, job_state: JobState) -> Job:
        """
        Selects the correct Job from the JobState based on the runner's name.

        This method uses the `name` attribute of the runner to dynamically
        access the corresponding job field within the `JobState` container.
        For example, if the runner's name is "reencode", this method will
        return `job_state.reencode`.

        Args:
            job_state (JobState): The container holding all jobs for a segment.

        Raises:
            ValueError: If the job corresponding to the runner's name is not
                        found in the `JobState`.

        Returns:
            Job: The specific job instance for this runner to process.
        """
        job = getattr(job_state, self.name)
        if job is None:
            raise ValueError(f"Job {self.name} is missing in JobState")
        return job

    def process(self, job: JobState) -> Any:
        """
        Performs the actual processing for a job. Must be implemented by subclasses.

        Args:
            job (JobState): The `JobState` container. Subclasses can access their
                            specific job object (e.g., `job.reencode`) and any
                            other prerequisite data from this container.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Returns:
            Any: The result of the processing, which will be passed to the
                 `on_complete` callback.
        """
        raise NotImplementedError

    def post_process(self, job: JobState) -> None:
        """
        A hook for executing code after a job is processed, regardless of success.

        This method is called in a `finally` block after `process()` and any
        exception handling. Subclasses can override it to perform cleanup,
        save state, or other final actions.

        Args:
            job (JobState): The `JobState` container that was just processed.
        """
        pass

    def _handle_retry(self, job_state: JobState, job: Job) -> None:
        """Handles the logic for re-queuing a failed job.

        It checks if the job's retry counts (`run_num_retries` for the current
        application execution and `total_num_retries` across all executions)
        are within the configured limits. If they are, it waits for a delay
        and inserts the `JobState` back at the front of the queue for immediate
        reprocessing.

        Args:
            job_state (JobState): The `JobState` container of the failed job.
            job (Job): The specific `Job` instance that failed.
        """
        can_retry_run = job.run_num_retries < self.settings.retry.run
        can_retry_total = job.total_num_retries < self.settings.retry.max

        if can_retry_run and can_retry_total:
            sleep(self.settings.retry.delay)
            # Insert at the front of the queue for immediate reprocessing.
            self.queue.insert(0, job_state)
