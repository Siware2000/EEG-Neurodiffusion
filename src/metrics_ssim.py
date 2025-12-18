from skimage.metrics import structural_similarity as ssim
import numpy as np

def avg_ssim(real_imgs, synth_imgs):
    scores = []
    for r, s in zip(real_imgs, synth_imgs):
        scores.append(ssim(r, s, channel_axis=2, data_range=1.0))
    return np.mean(scores)
