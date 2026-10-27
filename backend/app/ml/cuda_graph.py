import torch
from functools import lru_cache
from .phrase_losses import parts, objective


@lru_cache(maxsize=4)
def warmup_stream(device):
    # cuBLAS workspaces are per-stream. Reuse one stream across training runs.
    return torch.cuda.Stream(device=device)


class CudaTrainingGraph:
    """Capture forward/loss/backward only. LR, clipping, checks and optimizer stay dynamic.

    Requires one microbatch per update. Variable final batches use eager execution.
    Warmup does not update weights or consume the training RNG sequence.
    """

    def __init__(self, model, runtime, batch, onset_weight):
        self.model = model
        self.static = {key: value.clone() for key, value in batch.items()}
        self.graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(runtime.device)
        warmup = warmup_stream(runtime.device)
        warmup.wait_stream(current)
        index = runtime.device.index if runtime.device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index]):
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    model.zero_grad(set_to_none=True)
                    loss = objective(
                        parts(model, runtime.forward(model, self.static), self.static, onset_weight)
                    )
                    loss.backward()
            current.wait_stream(warmup)
            model.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph):
                self.terms = parts(model, runtime.forward(model, self.static), self.static, onset_weight)
                self.loss = objective(self.terms)
                self.loss.backward()
            self.gradients = [parameter.grad for parameter in model.parameters()]
        model.zero_grad(set_to_none=True)

    def matches(self, batch):
        return batch.keys() == self.static.keys() and all(
            value.shape == self.static[key].shape
            and value.dtype == self.static[key].dtype
            and value.device == self.static[key].device
            for key, value in batch.items()
        )

    def replay(self, batch):
        for key, value in batch.items():
            self.static[key].copy_(value, non_blocking=True)
        for parameter, gradient in zip(self.model.parameters(), self.gradients):
            parameter.grad = gradient
        self.graph.replay()
        return self.loss, self.terms
