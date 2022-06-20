import numpy as np
import matplotlib.pyplot as plt
from tpc_utils import background
import os


def load_data(path):
    input = np.load(path + "/raw_signals.npy")
    array = np.load(path + "/baselines.npy")
    return input, array


if __name__ == "__main__":
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "..", "data")
    raw_signals, baselines_root = load_data(path)
    num = np.random.randint(0, len(raw_signals), size=1)[0]
    raw_signal = raw_signals[num]
    baseline = baselines_root[num]
    background_tpc = background(raw_signal, 24, 1, 0, True, 3, True)
    print(np.array_equal(baseline, background_tpc))
    xt = np.arange(0.5, 512, 1)
    fig = plt.figure(dpi=150)
    plt.plot(xt, raw_signal, label="raw signal", lw=2)
    plt.plot(xt, baseline, label="baseline_tspectrum", ls="--", alpha=0.5, lw=2)
    plt.plot(xt, background_tpc, label="background_tpc_module", alpha=0.5, lw=2)
    plt.legend()
    plt.show()
