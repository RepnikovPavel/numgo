#!/usr/bin/env python3
import os
import numpy as np

TESTDATA_DIR = "testdata"

def generate():
    os.makedirs(TESTDATA_DIR, exist_ok=True)

    # scalar float64
    np.save(os.path.join(TESTDATA_DIR, "scalar_float64.npy"), np.float64(3.1415))

    # 1D int32
    np.save(os.path.join(TESTDATA_DIR, "1d_int32.npy"), np.array([10, 20, 30], dtype=np.int32))

    # 2D float32 Fortran order
    arr_f = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')
    np.save(os.path.join(TESTDATA_DIR, "2d_float32_F.npy"), arr_f)

    # NPZ without compression
    np.savez(os.path.join(TESTDATA_DIR, "multi_arrays.npz"),
             a=np.arange(5, dtype=np.uint16),
             b=np.linspace(0, 1, 4, dtype=np.float64),
             c=np.array([True, False, True]))

    # NPZ with compression
    np.savez_compressed(os.path.join(TESTDATA_DIR, "multi_compressed.npz"),
                        d=np.eye(3, dtype=np.complex64),
                        e=np.array(["hello", "world"], dtype=np.str_))

    print(f"Test data generated in '{TESTDATA_DIR}'")

if __name__ == "__main__":
    generate()