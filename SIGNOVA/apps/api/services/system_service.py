"""
System status service for SIGNOVA API.
"""

import platform
import sys
from typing import Any, Dict


def get_system_health() -> Dict[str, Any]:
    """Gather non-blocking system diagnostics."""
    diagnostics = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
    }
    try:
        import torch
        diagnostics["pytorch"] = torch.__version__
        diagnostics["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diagnostics["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        diagnostics["pytorch"] = "not_installed"
        diagnostics["cuda_available"] = False

    return diagnostics
