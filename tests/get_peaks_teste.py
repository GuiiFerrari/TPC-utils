import numpy as np
import matplotlib.pyplot as plt
from tpc_utils import get_peaks
from time import time
import os


def carrega_e_roda(PATH: str):
    arrays = np.load(os.path.join(PATH, "arrays.npz"), allow_pickle=True)
    t0 = time()
    arrays_a = arrays["arr_0"].astype(float)
    del arrays
    picos = get_peaks(arrays_a)
    print(f"Elapsed time = {time() - t0:.4f} segundos.")
    np.save(os.path.join(PATH, "picos.npy"), picos)


def load2():
    arrays = np.load("./data/arrays.npz", allow_pickle=True)["arr_0"]
    picos = np.load("./data/picos.npy", allow_pickle=True)
    xt = np.arange(0.5, 512, 1)
    for i in range(len(picos)):
        plt.figure(dpi=200)
        plt.plot(xt, arrays[i][512:])
        peaks = np.array(picos[i]).astype(float).round().astype(int)
        plt.plot(xt[peaks], arrays[i][peaks + 512], "x")
        plt.show()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(BASE_DIR, "..", "data")
    carrega_e_roda(PATH)
    # load2()
