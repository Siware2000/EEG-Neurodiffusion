import numpy as np
from scipy.signal import welch

def psd_similarity(real_sig, synth_sig, fs=128):
    f_r, p_r = welch(real_sig, fs=fs)
    f_s, p_s = welch(synth_sig, fs=fs)
    return np.corrcoef(p_r, p_s)[0,1]
