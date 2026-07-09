"""可选的 CUDA 硬件集成测试。

通过 ``YANGUA_RUN_CUDA_TESTS=1 pytest -m cuda`` 显式启用。
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(
    os.environ.get("YANGUA_RUN_CUDA_TESTS") != "1",
    reason="set YANGUA_RUN_CUDA_TESTS=1 to enable CUDA tests",
)
def test_cuda_kernel_in_fresh_process():
    code = """
import numpy as np
import taichi as ti
ti.init(arch=ti.cuda, random_seed=42)
from yan_gua.physics import CloudParticles
cloud = CloudParticles(count=100, win_w=320, win_h=180)
cloud.update(0.016, [])
assert np.isfinite(cloud.px).all()
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
