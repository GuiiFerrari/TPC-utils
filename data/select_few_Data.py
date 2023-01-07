import numpy as np
import uproot4 as up
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

with up.open(os.path.join(BASE_DIR, "tspec_out_224_700.root")) as f:
    branches = f[f.keys()[0]].arrays(library="np")

NUM_DADOS = 100
input = []
input_with_no_baseline = []
deconv = []
baselines = []
peaks = []
for i in range(NUM_DADOS):
    input.append(branches["input"][i])
    input_with_no_baseline.append(branches["inputBK"][i])
    deconv.append(branches["target"][i])
    baselines.append(branches["fundo"][i])
    peaks.append(branches["peaks"][i].astype(float).round().astype(int).reshape(-1))
input = np.array(input, dtype=float)
input_with_no_baseline = np.array(input_with_no_baseline, dtype=float)
deconv = np.array(deconv, dtype=float)
baselines = np.array(baselines, dtype=float)
peaks = np.array(peaks, dtype=object)

np.save("raw_signals.npy", input, allow_pickle=True)
np.save("raw_signals_wo_baseline.npy", input_with_no_baseline, allow_pickle=True)
np.save("deconv_signals.npy", deconv, allow_pickle=True)
np.save("baselines.npy", baselines, allow_pickle=True)
np.save("peaks.npy", peaks, allow_pickle=True)
