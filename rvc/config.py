import os
import sys
import torch

sys.path.append(os.getcwd())

from rvc import opencl

class Config:
    """Runtime configuration for the RVC inference pipeline.

    Selects the best available compute device (CUDA / MPS / OpenCL / CPU),
    configures half-precision behaviour, and exposes the padding parameters
    consumed by :class:`rvc.infer.pipeline.Pipeline`.
    """

    def __init__(self, is_half: bool = False, cpu_mode: bool = False):
        self.is_half = is_half
        self.cpu_mode = cpu_mode
        self.device = self._pick_device()
        # Disable half precision on CPU; fp16 path requires a CUDA / MPS GPU.
        if self.device == "cpu" or self.cpu_mode:
            self.is_half = False
        # OpenCL backend currently cannot run fp16 reliably through the path
        # used by the synthesizers, so fall back to fp32 there as well.
        if self.device.startswith("ocl"):
            self.is_half = False

        self.n_cpu = os.cpu_count() or 1
        # Use only a fraction of the available cores so that batched pitch
        # extraction does not starve the audio thread on small machines.
        self.n_cpu = max(1, min(self.n_cpu, 4))

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    def _pick_device(self) -> str:
        if self.cpu_mode:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        try:
            if opencl.is_available():
                return "ocl:0"
        except Exception:
            pass

        return "cpu"

    # ------------------------------------------------------------------
    # Pipeline padding parameters.  These mirror the original RVC defaults:
    # larger pads for fp32 to keep accuracy, smaller pads for fp16 to keep
    # memory usage in check on consumer GPUs.
    # ------------------------------------------------------------------
    def device_config(self):
        if self.is_half:
            pad = 1
            query = 6
            center = 30
            max_pad = 32
        else:
            pad = 3
            query = 10
            center = 60
            max_pad = 65
        return pad, query, center, max_pad

    def __repr__(self) -> str:
        return (
            f"Config(device={self.device!r}, is_half={self.is_half}, "
            f"cpu_mode={self.cpu_mode})"
        )
