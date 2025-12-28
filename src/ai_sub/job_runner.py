import concurrent.futures
from collections import deque
from threading import Event
from time import sleep
from typing import Generic, TypeVar

import logfire

from ai_sub.config import Settings
from ai_sub.data_models import Job

TJob = TypeVar("TJob", bound=Job)


class JobRunner(Generic[TJob]):
    """
    Abstracts the logic for running jobs from a queue with retries and optional
    stop events.
    """

    def __init__(
        self,
        queue: deque[TJob],
        settings: Settings,
        max_workers: int,
        stop_events: list[Event] = [],
        name: str = "JobRunner",
    ):
        self.queue = queue
        self.settings = settings
        self.max_workers = max_workers
        self.stop_events = stop_events
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
        Worker function that processes jobs from the queue.
        """
        while True:
            job: TJob | None = None
            try:
                # Attempt to get a job from the left of the queue.
                job = self.queue.popleft()

                # Increment retry counts
                job.run_num_retries += 1
                job.total_num_retries += 1

                self.process(job)

            except IndexError:
                # The queue is empty.
                if self.stop_events:
                    if all(e.is_set() for e in self.stop_events):
                        # All stop events are set, so no more jobs will come.
                        break
                    else:
                        # Stop events not set, wait for more jobs.
                        sleep(1)
                        continue
                else:
                    # No stop event configured, so we are done.
                    break

            except Exception:
                logfire.exception(f"Exception while running {self.name} job")
                if job is not None:
                    self._handle_retry(job)

            finally:
                if job is not None:
                    try:
                        self.post_process(job)
                    except Exception:
                        logfire.exception(
                            f"Exception in post_process for {self.name} job"
                        )

    def process(self, job: TJob) -> None:
        raise NotImplementedError

    def post_process(self, job: TJob) -> None:
        pass

    def _handle_retry(self, job: TJob) -> None:
        """Handles re-queuing the job if retry limits are not exceeded."""
        can_retry_run = job.run_num_retries < self.settings.retry.run
        can_retry_total = job.total_num_retries < self.settings.retry.max

        if can_retry_run and can_retry_total:
            sleep(self.settings.retry.delay)
            # Insert at the front of the queue for immediate reprocessing.
            self.queue.insert(0, job)
