from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import torch


def host_batch(values, pin=False):
    return {
        key: torch.from_numpy(value).pin_memory() if pin else torch.from_numpy(value)
        for key, value in values.items()
    }


@contextmanager
def device_batches(source, selected_device):
    """One bounded lookahead batch; no duplicate dataset, no unbounded pinned memory."""
    iterator = iter(source)
    if selected_device.type != "cuda":
        yield (
            {key: torch.from_numpy(value).to(selected_device) for key, value in values.items()}
            for values in iterator
        )
        return
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bob-prefetch")

    def prepare_next():
        values = next(iterator, None)
        return host_batch(values, pin=True) if values is not None else None

    pending = pool.submit(prepare_next)

    def ready_batches():
        nonlocal pending
        while True:
            host = pending.result()
            if host is None:
                return
            pending = pool.submit(prepare_next)
            yield {key: value.to(selected_device, non_blocking=True) for key, value in host.items()}

    ready = ready_batches()
    try:
        yield ready
    finally:
        ready.close()
        pending.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
