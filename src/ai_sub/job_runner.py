"""Generic job processing framework for the subtitle generation pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import logfire

from ai_sub.config import Settings
from ai_sub.data_models import Job, SegmentJobs


class JobRunner:
    """A generic, concurrent job processor.

    This class provides a framework for processing `SegmentJobs` objects from an
    asyncio.Queue in a concurrent manner using `asyncio.Task`. It handles job
    acquisition, retries on failure, and graceful shutdown.

    Subclasses must implement the `process` method to define the actual work.
    The `on_complete` callback can be used to chain dependent jobs by creating the next job in the pipeline.
    """

    def __init__(
        self,
        settings: Settings,
        max_workers: int,
        on_complete: Callable[[SegmentJobs, Any], Awaitable[None]] | None = None,
        name: str = "JobRunner",
    ):
        """Initializes the JobRunner.

        Args:
            settings (Settings): The application's configuration settings.
            max_workers (int): The maximum number of concurrent tasks.
            on_complete (Callable[[SegmentJobs, Any], Awaitable[None]] | None): An optional callback
                function that is executed upon successful completion of a job. It receives
                the job container and the result of the `process` method.
            name (str): The name of the runner. This name is crucial as it's used to
                dynamically access the corresponding job attribute from the `SegmentJobs` object
                (e.g., a runner with name 'reencode' will process `job_state.reencode`).

        """
        self.queue: asyncio.Queue[SegmentJobs] = asyncio.Queue()
        self.settings = settings
        self.max_workers = max_workers
        self.on_complete = on_complete
        self.name = name
        self.tasks: list[asyncio.Task] = []

    async def add_job(self, job: SegmentJobs) -> None:
        """Adds a job to the runner's queue.

        Args:
            job: The job container to add to the queue.
        """
        await self.queue.put(job)

    async def join(self) -> None:
        """Waits until all items in the queue have been processed."""
        await self.queue.join()

    async def start(self) -> None:
        """Starts the worker tasks.

        Raises:
            ValueError: If max_workers is less than or equal to 0.
        """
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {self.max_workers}")
        self.tasks = [asyncio.create_task(self.run()) for _ in range(self.max_workers)]

    async def shutdown(self) -> None:
        """Cancels all worker tasks and waits for them to exit."""
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    async def run(self) -> None:
        """The main worker loop that processes jobs from the async queue.

        This method runs in a continuous loop on each worker task. It attempts to
        get a `SegmentJobs` container from the queue. It exits when the task is cancelled.

        For each `SegmentJobs`, it:
        1. Uses `get_job()` to retrieve the specific `Job` object (e.g. `reencode`) for this runner.
        2. Increments the job's retry counters.
        3. Calls the `process()` method to perform the work (async).
        4. On success, calls the `on_complete` callback if it exists.
        5. On failure, calls `_handle_retry()` to potentially re-queue the job.
        6. Always calls `post_process()` for any cleanup tasks (async).
        """
        while True:
            job_state: SegmentJobs | None = None

            try:
                # Attempt to get a SegmentJobs container from the queue (async blocking).
                job_state = await self.queue.get()
            except asyncio.CancelledError:
                # Task cancellation requested.
                break

            try:
                current_job: Job | None = None
                try:
                    # Get the specific job for this runner from the SegmentJobs container.
                    current_job = self.get_job(job_state)

                    # Increment retry counters for the specific job.
                    current_job.run_num_retries += 1
                    current_job.total_num_retries += 1

                except Exception:
                    logfire.exception(f"Unexpected error in {self.name} runner loop")
                    continue

                with logfire.span(f"Executing {self.name} job for {current_job.name}"):
                    try:
                        result = await self.process(job_state)
                        if self.on_complete:
                            await self.on_complete(job_state, result)

                    except Exception:
                        job_name = f"'{current_job.name}'" if current_job else "unknown"
                        logfire.exception(f"Exception while running {self.name} job {job_name}")
                        if job_state is not None and current_job is not None:
                            await self._handle_retry(job_state, current_job)

                    finally:
                        if job_state is not None:
                            try:
                                await self.post_process(job_state)
                            except Exception:
                                logfire.exception(f"Exception in post_process for {self.name} job")
            finally:
                self.queue.task_done()

    def get_job(self, job_state: SegmentJobs) -> Job:
        """Selects the correct Job from the SegmentJobs container based on the runner's name.

        This method uses the `name` attribute of the runner to dynamically
        access the corresponding job field within the `SegmentJobs` container.
        For example, if the runner's name is "reencode", this method will
        return `job_state.reencode`.

        Args:
            job_state (SegmentJobs): The container holding all jobs for a segment.

        Returns:
            Job: The specific job instance for this runner to process.

        Raises:
            ValueError: If the job corresponding to the runner's name is not
                        found in the `SegmentJobs` container.

        """
        job = getattr(job_state, self.name)
        if job is None:
            raise ValueError(f"Job {self.name} is missing in JobState")
        return job

    async def process(self, job: SegmentJobs) -> Any:
        """Performs the actual processing for a job. Must be implemented by subclasses.

        Args:
            job (SegmentJobs): The `SegmentJobs` container. Subclasses can access their
                            specific job object (e.g., `job.reencode`) and any
                            other prerequisite data from this container.

        Returns:
            Any: The result of the processing, which will be passed to the
                 `on_complete` callback.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        """
        raise NotImplementedError

    async def post_process(self, job: SegmentJobs) -> None:
        """A hook for executing code after a job is processed, regardless of success.

        This method is called in a `finally` block after `process()` and any
        exception handling. Subclasses can override it to perform cleanup,
        save state, or other final actions.

        Args:
            job (SegmentJobs): The `SegmentJobs` container that was just processed.

        """
        pass

    async def _handle_retry(self, job_state: SegmentJobs, job: Job) -> None:
        """Handles the logic for re-queuing a failed job.

        It checks if the job's retry counts (`run_num_retries` for the current
        application execution and `total_num_retries` across all executions)
        are within the configured limits. If they are, it waits for a delay
        and puts the `SegmentJobs` back into the queue.

        Args:
            job_state (SegmentJobs): The `SegmentJobs` container of the failed job.
            job (Job): The specific `Job` instance that failed.

        """
        can_retry_run = job.run_num_retries < self.settings.retry.run
        can_retry_total = job.total_num_retries < self.settings.retry.max

        if can_retry_run and can_retry_total:
            await asyncio.sleep(self.settings.retry.delay)
            # Put back into the queue. Note: asyncio.Queue is FIFO, so it goes to the back.
            await self.add_job(job_state)
