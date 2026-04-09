import numpy as np
import os

os.makedirs("testdata", exist_ok=True)

# Скаляр
np.save("testdata/scalar_float64.npy", np.array(3.1415))

# 1D int32
np.save("testdata/1d_int32.npy", np.array([10, 20, 30], dtype=np.int32))

# 2D float32 Fortran order
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order='F')
np.save("testdata/2d_float32_F.npy", a)

# Байтовые строки (S)
arr_s = np.array(['hello', 'world'], dtype='S10')
np.save("testdata/string_array_S.npy", arr_s)

arr_s = np.array(['two', 'three'], dtype='S10')
np.save("testdata/string_array_S_var.npy", arr_s)

# Unicode строки (U)
arr_u = np.array(['hello', 'Привет'], dtype='U10')
np.save("testdata/string_array_U.npy", arr_u)

arr_u = np.array(['世界世界', '世界'], dtype='U10')
np.save("testdata/string_array_U_var.npy", arr_u)

# NPZ архив
np.savez("testdata/multi_arrays.npz",
         a=np.arange(5, dtype=np.uint16),
         b=np.linspace(0, 1, 4),
         c=np.array([True, False, True]))