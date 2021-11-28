import numpy as np
import matplotlib.pyplot as plt
import tpc_utils as myModule
from time import time

def carrega_e_roda():
    arrays = np.load("arrays.npz", allow_pickle = True)
    # print(dir(arrays))
    # print(arrays['arr_0'])
    t0    = time()
    arrays_a = arrays['arr_0'].astype(float)
    del arrays
    picos = myModule.get_peaks(arrays_a)
    # picos = np.array(picos, dtype = object)
    print(f"Tempo consumido = {time() - t0} segundos.")
    np.save("picos.npy", picos)

def load2():
    arrays = np.load("arrays.npz", allow_pickle = True)['arr_0']
    picos  = np.load("picos.npy", allow_pickle = True)
    xt = np.arange(0.5, 512, 1)
    for i in range(len(picos)):
        fig = plt.figure(dpi = 200)
        plt.plot(xt, arrays[i][512:])
        peaks = np.array(picos[i]).astype(float).round().astype(int)
        plt.plot(xt[peaks], arrays[i][peaks + 512], "x")
        plt.show()

if __name__ == "__main__":
    # carrega_e_roda()
    load2()