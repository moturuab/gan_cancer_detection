import pydicom
import cv2
import os
import numpy as np
import pandas as pd
from skimage import exposure
from skimage import restoration
from matplotlib import pyplot as plt


def remove_clips(volume):
    volume[volume == 0] = np.nan
    volume[volume == 1] = np.nan
    for i in range(8):
        v = pd.DataFrame(data=volume[i])
        volume[i] = v.interpolate(method='linear')


def correct_background(volume):
    # select background columns
    # volume[0, 50:150, 0]
    row_background = np.mean(volume[0, :, 50:100], axis=1)
    for i in range(8):
        volume[i] -= row_background[..., np.newaxis]
    volume = (volume-min(volume.flatten()))/(max(volume.flatten())-min(volume.flatten()))
    return volume


def correct_contrast(volume):
    for i in range(8):
        #volume[i] = exposure.equalize_hist(volume[i])        # < BAD
        volume[i] = exposure.equalize_adapthist(volume[i])   # < GOOD
        #volume[i] = exposure.rescale_intensity(volume[i])    # < GOOD
        #volume[i] = exposure.adjust_gamma(volume[i])         # < NEEDS TWEAKING
        #volume[i] = exposure.adjust_sigmoid(volume[i])       # < NEEDS TWEAKING
        #volume[i] = exposure.adjust_log(volume[i])           # < GOOD
    return volume


def preprocess(volume):
    remove_clips(volume[:,:,:50]) # left clips
    remove_clips(volume[:,:,-50:]) # right clips
    volume = correct_background(volume) # normalize band brightness
    return volume


def remove_noise(volume):
    for i in range(8):
        #volume[i] = restoration.denoise_tv_bregman(volume[i], weight=1e+10)       # < NEEDS TWEAKING
        #volume[i] = restoration.denoise_tv_chambolle(volume[i])                   # < BAD
        #volume[i] = restoration.denoise_bilateral(volume[i], multichannel=False)  # < GOOD
        volume[i] = restoration.denoise_wavelet(volume[i])                        # < GOOD
        #volume[i] = restoration.denoise_nl_means(volume[i])                       # < BAD
    return volume


def stitch_volume(dir):
    nz = len(os.listdir(dir))
    temp = pydicom.dcmread(os.path.join(dir, os.listdir(dir)[0]))
    nx = temp.Rows
    ny = temp.Columns
    volume = np.zeros((nz, nx, ny))
    for absdir,_,fns in os.walk(dir):
        i = 0
        for fn in sorted(fns):
            print(absdir, fn)
            abspath = os.path.join(absdir, fn)
            ds = pydicom.dcmread(abspath)
            slice = ds.pixel_array
            volume[i] = slice
            i += 1
    return volume/4000


def main():
    dir = "Abhi-Alex-Full-Body-MR"
    volume = preprocess(stitch_volume(dir))
    volume = correct_contrast(volume) # contrast correction
    volume = remove_noise(volume)
    volume2 = preprocess(stitch_volume(dir))
    for i in range(8):
        plt.imshow(volume2[i], cmap='gray', vmin=0, vmax=1)
        plt.show()
        plt.imshow(volume[i], cmap='gray', vmin=0, vmax=1)
        plt.show()


if __name__ == "__main__":
    main()
