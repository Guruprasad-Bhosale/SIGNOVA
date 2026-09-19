"""
Integration tests for environment checker tools.
"""

from scripts.check_environment import get_disk_space, get_gpu_info, get_ram_info


def test_environment_check_diagnostics():
    disk = get_disk_space(".")
    assert "GB" in disk or "Unknown" in disk

    ram = get_ram_info()
    assert "GB" in ram or "Unable" in ram

    gpu = get_gpu_info()
    assert isinstance(gpu, dict)
    assert "detected" in gpu
