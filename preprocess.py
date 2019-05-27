import pydicom
import cv2
import os
import numpy as np
import pandas as pd
from skimage import exposure
from skimage import restoration
from skimage import transform
from matplotlib import pyplot as plt
import SimpleITK as sitk

from eda import DatasetAnalyzer

def remove_clips(volume):
    volume[volume == 0] = np.nan
    volume[volume == 1] = np.nan
    for i in range(volume.shape[0]):
        v = pd.DataFrame(data=volume[i])
        volume[i] = v.interpolate(method='linear')


def correct_background(volume):
    # select background columns
	sides = np.concatenate((volume[0, :, 0:100], volume[0, :, -100:]), axis=1)
	blur = cv2.GaussianBlur(sides, (7, 7), 0)
	#plt.imshow(blur, cmap="gray")
	#plt.show()
	row_background = np.mean(blur, axis=1)
	for i in range(volume.shape[0]):
		volume[i] -= row_background[..., np.newaxis]
	volume = (volume-min(volume.flatten()))/(max(volume.flatten())-min(volume.flatten()))
	return volume


def correct_contrast(volume):
	#implement 3D CLAHE (later)
	#volume = exposure.rescale_intensity(volume)
	#return volume

    for i in range(volume.shape[0]):
        #volume[i] = exposure.rescale_intensity(volume[i])    # < GOOD
        #volume[i] = exposure.equalize_hist(volume[i])        # < BAD
        volume[i] = exposure.equalize_adapthist(volume[i])   # < GOOD
        #volume[i] = exposure.adjust_gamma(volume[i])         # < NEEDS TWEAKING
        #volume[i] = exposure.adjust_sigmoid(volume[i])       # < NEEDS TWEAKING
        #volume[i] = exposure.adjust_log(volume[i])           # < GOOD
    return volume


def preprocess(volume):
    remove_clips(volume[:,:,:50]) # left clips
    remove_clips(volume[:,:,-50:]) # right clips
    volume = correct_background(volume) # normalize band brightness
    volume = correct_contrast(volume) # contrast correction
    return volume


'''
def remove_noise(volume):
    for i in range(8):
        #volume[i] = restoration.denoise_tv_bregman(volume[i], weight=1e+10)       # < NEEDS TWEAKING
        #volume[i] = restoration.denoise_tv_chambolle(volume[i])                   # < BAD
        #volume[i] = restoration.denoise_bilateral(volume[i], multichannel=False)  # < GOOD
        volume[i] = restoration.denoise_wavelet(volume[i])                        # < GOOD
        #volume[i] = restoration.denoise_nl_means(volume[i])                       # < BAD
    return volume
'''

'''
def correct_bias_field(volume):
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    for i in range(8):
        volume_image = sitk.GetImageFromArray(volume[i])
        mask_image = sitk.OtsuThreshold( volume_image, 0, 1, 200 )
        corrected_image = corrector.Execute(volume_image,mask_image)
        volume[i] = sitk.GetArrayFromImage(corrected_image)
    return volume
'''

	

def stitch_volume(filename_list):
	nz = len(filename_list)
	if nz < 10:
		return "yikes"
	temp = pydicom.dcmread(filename_list[0])
	nx = temp.Rows
	ny = temp.Columns
	volume = np.zeros((nz, nx, ny))
	
	l = []
	for fn in filename_list:
		temp = pydicom.dcmread(fn)
		l.append((fn, temp.InstanceNumber))
		
	i = 0  
	for fn in sorted(l, key=lambda x: x[1]):
		ds = pydicom.dcmread(fn[0])
		slice = ds.pixel_array
		if slice.shape != volume[i].shape:
			return "yikes"
			
		volume[i] = slice
		i += 1
	return volume/4000


def main():
	mri_directory = '../mri_data/linking-anna-alex-abhi-2'
	dataset_analyzer = DatasetAnalyzer(['PatientName', 'AccessionNumber'], mri_directory)
	dataset_analyzer.read_dicoms()
	print(len(dataset_analyzer.organized_folders))
	# dir = "/home/abhishekmoturu/Desktop/ScalarVolume_22"
	i = 0
	for k, v in dataset_analyzer.organized_folders.items():
		print("Number of slices: {}".format(len(v)))
		volume = stitch_volume(v)
		if volume == "yikes":
			print("yikes")
			continue			
		volume = preprocess(volume)
		print(k)
		volume = resize_or_pad(volume)
		#for i in range(volume.shape[0]):
		#	plt.imshow(volume[i], cmap='gray', vmin=0, vmax=1)
		#	plt.show()
		np.save('./wbmri/volume_{}.npy'.format(i), volume)
		i += 1

def resize_or_pad(volume):
	nz = len(volume)
	if nz < 24:
		volume = np.pad(volume, ((24-nz, 0), (0, 0), (0, 0)), 'minimum')
	volume = transform.resize(volume, (24,1448,512))
	return volume

if __name__ == "__main__":
    main()
