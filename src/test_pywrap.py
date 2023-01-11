import numpy as np
from ttrs import TTVc64

def test_get_kernels():
    tt = TTVc64([2, 2, 3, 2, 1, 3, 5], 15)
    kernels = tt.get_kernels()
    agr = np.array([1.])
    for kernel in kernels:
        agr = np.tensordot(agr, kernel.sum(1), axes=1)
    assert(np.abs(agr - np.exp(tt.log_sum())) < 1e-8)
