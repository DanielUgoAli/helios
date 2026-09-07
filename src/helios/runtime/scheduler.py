import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Thread
from typing import Generic, TypeVar

Payload = TypeVar("Payload")
Result = TypeVar("Result")


class QueueFullError(RuntimeError):
    pass


class SchedulerClosedError(RuntimeError):
    pass


@dataclass
class Job(Generic[Payload, Result]):
    payload: Payload
    request_ids: tuple[str, ...]
    enqueued_at: float = field(default_factory=time.perf_counter)
    future: Future[Result] = field(default_factory=Future)


class Scheduler(Generic[Payload, Result]):
    """One worker that lets the engine schedule one generation iteration at a time."""

    def __init__(
        self,
        tick: Callable[["Scheduler[Payload, Result]"], bool],
        *,
        max_batch_size: int,
        max_queue_size: int,
        batch_wait_seconds: float,
    ) -> None:
        self._tick = tick
        self._max_batch_size = max_batch_size
        self._max_queue_size = max_queue_size
        self._batch_wait_seconds = batch_wait_seconds
        self._condition = Condition()
        self._waiting: deque[Job[Payload, Result]] = deque()
        self._active: tuple[Job[Payload, Result], ...] = ()
        self._closed = False
        self._worker = Thread(target=self._run, name="helios-scheduler", daemon=True)
        self._worker.start()

    def enqueue(self, job: Job[Payload, Result]) -> Future[Result]:
        with self._condition:
            if self._closed:
                raise SchedulerClosedError("The generation scheduler is closed.")
            self._discard_cancelled_waiting()
            if len(self._waiting) >= self._max_queue_size:
                raise QueueFullError("The generation waiting queue is full.")
            self._waiting.append(job)
            self._condition.notify()
        return job.future

    def peek(self) -> Job[Payload, Result] | None:
        with self._condition:
            self._discard_cancelled_waiting()
            return self._waiting[0] if self._waiting else None

    def take(
        self, expected: Job[Payload, Result] | None = None
    ) -> Job[Payload, Result] | None:
        with self._condition:
            self._discard_cancelled_waiting()
            if expected is not None and (
                not self._waiting or self._waiting[0] is not expected
            ):
                return None
            while self._waiting:
                job = self._waiting.popleft()
                if job.future.set_running_or_notify_cancel():
                    self._active = (*self._active, job)
                    return job
            return None

    def set_active(self, jobs: tuple[Job[Payload, Result], ...]) -> None:
        with self._condition:
            self._active = jobs

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "waiting": [
                    request_id
                    for job in self._waiting
                    for request_id in job.request_ids
                ],
                "active": [
                    request_id for job in self._active for request_id in job.request_ids
                ],
                "max_batch_size": self._max_batch_size,
                "max_queue_size": self._max_queue_size,
                "batch_wait_ms": self._batch_wait_seconds * 1_000,
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            jobs = (*self._waiting, *self._active)
            self._waiting.clear()
            self._condition.notify_all()
        error = SchedulerClosedError("The generation scheduler is closed.")
        for job in jobs:
            if not job.future.done():
                job.future.set_exception(error)
        self._worker.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._discard_cancelled_waiting()
                while not self._waiting and not self._active and not self._closed:
                    self._condition.wait()
                    self._discard_cancelled_waiting()
                if self._closed:
                    return
                if not self._active:
                    self._wait_for_initial_requests()
            try:
                has_active = self._tick(self)
            except Exception as error:  # noqa: BLE001
                with self._condition:
                    active = self._active
                    self._active = ()
                for job in active:
                    if not job.future.done():
                        job.future.set_exception(error)
                has_active = False
            if not has_active:
                with self._condition:
                    self._active = ()

    def _wait_for_initial_requests(self) -> None:
        if self._max_batch_size == 1:
            return
        first = self._waiting[0]
        deadline = first.enqueued_at + self._batch_wait_seconds
        while len(self._waiting) < self._max_batch_size and not self._closed:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            self._condition.wait(remaining)
            self._discard_cancelled_waiting()

    def _discard_cancelled_waiting(self) -> None:
        self._waiting = deque(
            job for job in self._waiting if not job.future.cancelled()
        )
