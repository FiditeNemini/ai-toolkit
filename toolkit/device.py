"""
Cross-platform device utilities for AI Toolkit.
Supports CUDA, MPS (Apple Silicon), and CPU backends.
"""
import torch
from typing import Optional, Tuple, Union


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def get_device(preferred: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the best available device with optional preference.
    
    Priority: CUDA > MPS > CPU (unless preferred is specified)
    
    Args:
        preferred: Optional device preference. If specified, returns that device.
                   Can be a string like "cuda", "mps", "cpu" or a torch.device.
    
    Returns:
        torch.device: The selected device
    """
    if preferred is not None:
        if isinstance(preferred, torch.device):
            return preferred
        return torch.device(preferred)
    
    if is_cuda_available():
        return torch.device("cuda")
    elif is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_type() -> str:
    """
    Get device type string for torch.autocast().
    
    Returns:
        str: "cuda", "mps", or "cpu"
    """
    if is_cuda_available():
        return "cuda"
    elif is_mps_available():
        return "mps"
    return "cpu"


def empty_cache():
    """Clear GPU memory cache on the current device."""
    if is_cuda_available():
        torch.cuda.empty_cache()
    elif is_mps_available():
        torch.mps.empty_cache()
    # CPU has no cache to clear


def synchronize():
    """Synchronize the current device."""
    if is_cuda_available():
        torch.cuda.synchronize()
    elif is_mps_available():
        torch.mps.synchronize()
    # CPU is always synchronous


def manual_seed(seed: int):
    """
    Set random seed on all available devices.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # MPS uses torch.manual_seed() which we already called


def get_rng_state() -> Optional[torch.Tensor]:
    """
    Get the RNG state for the current GPU device.
    
    Returns:
        torch.Tensor or None: The RNG state tensor, or None if no GPU available
    """
    if is_cuda_available():
        return torch.cuda.get_rng_state()
    elif is_mps_available():
        # MPS doesn't have a separate RNG state API, use global state
        return torch.get_rng_state()
    return None


def set_rng_state(state: torch.Tensor):
    """
    Set the RNG state for the current GPU device.
    
    Args:
        state: The RNG state tensor to restore
    """
    if state is None:
        return
    if is_cuda_available():
        torch.cuda.set_rng_state(state)
    elif is_mps_available():
        torch.set_rng_state(state)


def get_gpu_memory_info() -> Optional[Tuple[Optional[int], int, Optional[int]]]:
    """
    Get GPU memory info (total, used, free) in MB.
    
    Returns:
        Tuple of (total_mb, used_mb, free_mb) or None if no GPU available.
        For MPS (unified memory), total and free may be None as they're not
        directly queryable.
    """
    if is_cuda_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory // (1024 * 1024)
        allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
        free = total - allocated
        return (total, allocated, free)
    elif is_mps_available():
        # MPS uses unified memory - only allocated is queryable
        allocated = torch.mps.current_allocated_memory() // (1024 * 1024)
        return (None, allocated, None)
    return None


def get_device_name() -> str:
    """
    Get a human-readable name for the current GPU device.
    
    Returns:
        str: Device name or "CPU" if no GPU available
    """
    if is_cuda_available():
        return torch.cuda.get_device_name(0)
    elif is_mps_available():
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def device_count() -> int:
    """
    Get the number of available GPU devices.
    
    Returns:
        int: Number of GPUs (0 for CPU-only, 1 for MPS)
    """
    if is_cuda_available():
        return torch.cuda.device_count()
    elif is_mps_available():
        return 1  # MPS always has exactly 1 device
    return 0
