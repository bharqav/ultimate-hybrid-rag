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
        self._cond = asyncio.Condition(self._lock)
        self.allocated_mem_gb = 0.0

    async def reserve(self, required_mem_gb: float):
        async with self._cond:
            while True:
                if torch and torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    free_gb = free / (1024**3)
                    total_gb = total / (1024**3)
                    limit = total_gb * self.max_mem
                    
                    if (self.allocated_mem_gb + required_mem_gb) <= limit and required_mem_gb <= free_gb:
                        self.allocated_mem_gb += required_mem_gb
                        log.debug(f"Reserved {required_mem_gb:.2f}GB GPU Memory. Total allocated: {self.allocated_mem_gb:.2f}GB")
                        return
                    else:
                        log.warning(f"Waiting for GPU memory. Required: {required_mem_gb:.2f}GB, Free: {free_gb:.2f}GB")
                else:
                    # CPU fallback, just pretend we allocated it
                    self.allocated_mem_gb += required_mem_gb
                    return
                
                # Wait until memory is released
                await self._cond.wait()

    async def release(self, released_mem_gb: float):
        async with self._cond:
            self.allocated_mem_gb = max(0.0, self.allocated_mem_gb - released_mem_gb)
            log.debug(f"Released {released_mem_gb:.2f}GB GPU Memory. Total allocated: {self.allocated_mem_gb:.2f}GB")
            self._cond.notify_all()
