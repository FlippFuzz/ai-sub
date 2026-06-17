"""Generic job processing framework for the subtitle generation pipeline."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import logfire

from ai_sub.config import Settings
from ai_sub.data_models import Job, QuotaExceededError, SegmentJobs


class JobRunner:
    """A generic, concurrent job processor.

    This class provides a framework for processing `SegmentJobs` objects from an
    asyncio.Queue in a concurrent manner using `asyncio.Task`. It handles job
    acquisition, retries on failure, and graceful shutdown.

    **Retry Logic (Persistence Layer):**
    This runner manages retries that survive application restarts:
    1. **Persistence:** The `total_attempts` field on the `Job` object is
       updated after a failed or successful attempt and saved to disk in `post_process`.
    2. **Gatekeeping:** `add_job` checks if a job has already exceeded
       `settings.retry.max_runs`. If so, it is never queued.
    3. **Resilience:** If the process is interrupted (e.g. cancellation) or
       hits a provider quota limit (`QuotaExceededError`), the counter is NOT incremented.
       This is because the failure was due to provider limits, not the content
       of the video segment. This ensures we don't 'waste' a segment's retry
       attempts when we just need to wait for a daily reset.
    4. **Failure Handling:** Generic exceptions are caught and logged, leaving
       the job in an incomplete state. Because the counter was incremented,
       re-running the application will attempt the job again until `max_runs`
       is hit.

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
        current_job = self.get_job(job)
        if current_job.total_attempts < self.settings.retry.max_runs:
            await self.queue.put(job)
        else:
            logfire.warning(
                f"Skipping {self.name} job for {current_job.name}: Max attempts reached ({current_job.total_attempts})"
            )

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
        5. On failure, logs the exception; the job remains incomplete for this execution.
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

                except Exception:
                    logfire.exception(f"Unexpected error in {self.name} runner loop")
                    continue

                with logfire.span(f"Executing {self.name} job for {current_job.name}"):
                    try:
                        result = await self.process(job_state)
                        if self.on_complete:
                            await self.on_complete(job_state, result)

                        # Success: increment the attempt counter
                        current_job.total_attempts += 1

                    except QuotaExceededError:
                        logfire.warning(f"Free tier daily quota exceeded for {self.name} job '{current_job.name}'.")

                    except Exception:
                        # Failure: increment the attempt counter
                        current_job.total_attempts += 1
                        logfire.exception(f"Exception while running {self.name} job '{current_job.name}'")

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
