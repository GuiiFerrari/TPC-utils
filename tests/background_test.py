import numpy as np
import matplotlib.pyplot as plt
from tpc_utils import background
import os
import pytest


def load_data(path):
    input = np.load(path + "/raw_signals.npy")
    array = np.load(path + "/baselines.npy")
    return input, array


@pytest.mark.skip("deprecated")
def deprecated_test_background():
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "..", "data")
    raw_signals, baselines_root = load_data(path)
    num = np.random.randint(0, len(raw_signals), size=1)[0]
    raw_signal = raw_signals[num]
    baseline = baselines_root[num]
    background_tpc = background(raw_signal, 24, 1, 0, True, 3, True)
    print(np.array_equal(baseline, background_tpc))
    xt = np.arange(0.5, 512, 1)
    _ = plt.figure(dpi=150)
    plt.plot(xt, raw_signal, label="raw signal", lw=2)
    plt.plot(xt, baseline, label="baseline_tspectrum", ls="--", alpha=0.5, lw=2)
    plt.plot(xt, background_tpc, label="background_tpc_module", alpha=0.5, lw=2)
    plt.legend()
    plt.show()
    nums = np.random.randint(0, len(raw_signals), size=5)
    raw_signal = raw_signals[nums]
    baseline = baselines_root[nums]
    background_tpc = background(raw_signal, 24, 1, 0, True, 3, True)
    assert np.array_equal(baseline, background_tpc)
    for num in range(5):
        _ = plt.figure(dpi=150)
        plt.plot(xt, raw_signal[num], label="raw signal", lw=2)
        plt.plot(
            xt,
            baseline[num],
            label="baseline_tspectrum",
            ls="--",
            alpha=0.5,
            lw=2,
        )
        plt.plot(
            xt,
            background_tpc[num],
            label="background_tpc_module",
            alpha=0.5,
            lw=2,
        )
        plt.legend()
        plt.show()


@pytest.mark.core
class TestBackground:
    def test_background(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "..", "data")
        raw_signals, baselines_root = load_data(path)
        num = np.random.randint(0, len(raw_signals), size=1)[0]
        raw_signal = raw_signals[num]
        baseline = baselines_root[num]
        background_tpc = background(raw_signal, 24, 1, 0, True, 3, True)
        assert np.array_equal(baseline, background_tpc)

    def test_background_batch(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "..", "data")
        raw_signals, baselines_root = load_data(path)
        nums = np.random.randint(0, len(raw_signals), size=5)
        raw_signal = raw_signals[nums]
        baseline = baselines_root[nums]
        background_tpc = background(raw_signal, 24, 1, 0, True, 3, True)
        assert np.array_equal(baseline, background_tpc)


if __name__ == "__main__":
    deprecated_test_background()
