import numpy as np
import matplotlib.pyplot as plt
from tpc_utils import search_high_res
import os
import pytest


def load_data(path):
    input = np.load(path + "/raw_signals_wo_baseline.npy")
    return input


@pytest.mark.skip("deprecated")
def test_search_high_res():
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, "..", "data")
    raw_signals = load_data(path)
    num = np.random.randint(0, len(raw_signals), size=1)[0]
    raw_signal = raw_signals[num]
    deconv_tpc, peaks = search_high_res(raw_signal, 5.0, 20.0, False, 700, False, 3)
    peaks = peaks.round().astype(int)
    xt = np.arange(0.5, 512, 1)
    _ = plt.figure(dpi=150)
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
    nums = np.random.randint(0, len(raw_signals), size=5)
    raw_signal = raw_signals[nums]
    deconv_tpc, peaks = search_high_res(raw_signal, 5.0, 20.0, False, 700, False, 3)
    for num in range(5):
        peaks_p = peaks[num].round().astype(int)
        _ = plt.figure(dpi=150)
        plt.plot(xt, raw_signal[num], label="raw signal", lw=2)
        plt.plot(
            xt,
            deconv_tpc[num],
            label="deconv_tpc_module",
            alpha=0.5,
            lw=2,
        )
        plt.scatter(
            xt[peaks_p],
            deconv_tpc[num][peaks_p],
            marker="x",
            label="peaks tpc_module",
            alpha=1,
            lw=2,
            zorder=3,
        )
        plt.legend()
        plt.show()


class TestDeconvolution:
    @pytest.mark.core
    def test_search_high_res_shape_type(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "..", "data")
        raw_signals = load_data(path)
        num = np.random.randint(0, len(raw_signals), size=1)[0]
        raw_signal = raw_signals[num]
        deconv_tpc, _ = search_high_res(raw_signal, 5.0, 20.0, False, 700, False, 3)
        assert deconv_tpc.shape == (512,)
        assert deconv_tpc.dtype == np.float64

    @pytest.mark.core
    def test_batch_search_high_res_shape_type(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, "..", "data")
        raw_signals = load_data(path)
        num = np.random.randint(0, len(raw_signals), size=5)
        raw_signal = raw_signals[num]
        deconv_tpc, _ = search_high_res(raw_signal, 5.0, 20.0, False, 700, False, 3)
        assert deconv_tpc.shape == (5, 512)
        assert deconv_tpc.dtype == np.float64


if __name__ == "__main__":
    test_search_high_res()
