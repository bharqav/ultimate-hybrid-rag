import asyncio

from config.logging import setup_logging
from config.settings import get_settings
from core.deps import torch

log = setup_logging()


class GPUScheduler:
    def __init__(self, max_mem=None):
        settings = get_settings()
        self.max_mem = max_mem if max_mem is not None else settings.max_gpu_memory
        self._lock = asyncio.Lock()

    async def reserve(self, required_mem_gb: float):
        async with self._lock:
            if torch and torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                if required_mem_gb > free * self.max_mem:
                    log.warning("GPU memory low; consider evicting models")
