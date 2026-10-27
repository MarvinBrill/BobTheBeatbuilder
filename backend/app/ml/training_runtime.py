from contextlib import nullcontext
import time
import torch


class TrainingRuntime:
    """Execution policy only; weights, targets and mathematical objectives stay unchanged."""

    def __init__(self, selected_device, precision="auto"):
        self.device = selected_device
        self.cuda = selected_device.type == "cuda"
        if precision not in ("auto", "float32", "bfloat16"):
            raise ValueError("Unknown training precision")
        supported = self.cuda and torch.cuda.is_bf16_supported()
        if precision == "bfloat16" and not supported:
            raise ValueError("BF16 training requires a CUDA GPU with BF16 support")
        self.bf16 = supported and precision != "float32"
        self.precision = "bfloat16" if self.bf16 else "float32"
        if self.cuda:
            # FP32 remains strict FP32, matching validation and the regression benchmark.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def forward(self, model, batch):
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else nullcontext()
        with precision:
            output = model(batch)
        # Mixture likelihoods, tiny scales, motion derivatives and critic losses
        # stay in FP32, outside autocast. FP32 master weights need no loss scaler for BF16.
        return {key: value.float() for key, value in output.items()}

    def optimizer(self, model, learning_rate, weight_decay):
        options = {"fused": True} if self.cuda else {"foreach": False}
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **options)

    def clip_and_check(self, model, finite_loss):
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, foreach=self.cuda)
        # One synchronization per optimizer update, not per term and microbatch.
        if not (torch.isfinite(norm) & finite_loss).item():
            raise ValueError("Non-finite loss or gradients; model update refused")

    def begin_measurement(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def measurement(self, start, examples, updates):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        return {
            "training_seconds": seconds,
            "examples_per_second": examples / max(seconds, 1e-9),
            "optimizer_updates": updates,
            "precision": self.precision,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20 if self.cuda else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20 if self.cuda else None,
        }
