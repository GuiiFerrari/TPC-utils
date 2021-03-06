import numpy as np
import matplotlib.pyplot as plt
from tpc_utils import search_high_res
import os


def load_data(path):
    input = np.load(path + "/raw_signals_wo_baseline.npy")
    return input


if __name__ == "__main__":
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "..", "data")
    raw_signals, _ = load_data(path)
    num = np.random.randint(0, len(raw_signals), size=1)[0]
    raw_signal = raw_signals[num]
    deconv_tpc, peaks = search_high_res(
        raw_signal, 5.0, 20.0, False, 700, False, 3
    )
    peaks = peaks.round().astype(int)
    xt = np.arange(0.5, 512, 1)
    fig = plt.figure(dpi=150)
    plt.plot(xt, raw_signal, label="raw signal", lw=2)
    plt.plot(xt, deconv_tpc, label="deconv_tpc_module", alpha=0.5, lw=2)
    plt.scatter(
        xt[peaks],
        deconv_tpc[peaks],
        marker="x",
        label="peaks tpc_module",
        alpha=1,
        lw=2,
        zorder=3,
    )
    plt.legend()
    plt.show()
