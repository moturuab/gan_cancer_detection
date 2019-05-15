import pydicom

from matplotlib import pyplot as plt

import cv2
import os
import numpy as np
from skimage import exposure


def remove_background(volume):
    # select background columns
    # volume[0, 50:200, 0]


    row_background = np.mean(volume[0, :, 50:200], axis=1)

    for i in range(8):

        volume[i] -= row_background[..., np.newaxis]

        # plt.imshow(noiseless, cmap='gray')
        # plt.show()



# def histeq(volume):
# 	# flatten array before equalizing histogram


# 	return histeq_volume




def preproc(volume):

	remove_background(volume)

	print("applied background removal")
	plt.imshow(volume[3], cmap='gray')
	plt.show()



	# denoise
	flat = volume.flatten()


	plt.hist(flat , bins='auto')
	plt.show()

	p2, p98 = np.percentile(flat, (0.5, 99.5))

	# normalize image to full spectrum
	img_rescale = exposure.rescale_intensity(flat, in_range=(min(flat), max(flat)), out_range=(0, 1))

	plt.hist(img_rescale , bins='auto')
	plt.show()

	plt.imshow(img_rescale.reshape(volume.shape)[3], cmap="gray")
	plt.show()

	he = exposure.equalize_hist(img_rescale)




	plt.hist(he , bins='auto')
	plt.show()



	# plt.hist(volume[:,:,0:2].flatten() , bins='auto')
	# plt.show()


	histeq_volume = he.reshape(volume.shape)

	print("a")
	plt.imshow(histeq_volume[3], cmap='gray')
	plt.show()
	print("b")


	# a = cv2.fastNlMeansDenoising(img_rescale.reshape(volume.shape)[3],None,10,10,7,21)
	# plt.imshow(a, cmap='gray')
	# plt.show()

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

    print(np.where(volume == 4000))


    # for i in range(8):

    #     plt.imshow(volume[i], cmap='gray')
    #     # plt.yticks(np.arange(0, 1300, 20))
    #     plt.show()
    #     preproc(volume)

    # plt.hist(volume[:,:,0:2].flatten() , bins='auto')
    # plt.show()

    # plt.hist(volume[:,:,0:200].flatten() , bins='auto')
    # plt.show()

    # plt.hist(volume[:,:,0:200].flatten() , bins='auto')
    # plt.show()

    # plt.hist(volume[:,:,0:200].flatten() , bins='auto')
    # plt.show()

    return volume





def main():
    dir = "Abhi-Alex-Full-Body-MR"

    volume = stitch_volume(dir)
    preproc(volume)


    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # cl1 = clahe.apply(data)

    # plt.imshow(data, cmap='gray')
    # plt.show()



    # image = cv2.imread("lenacolor512.tiff", cv2.IMREAD_COLOR)  # uint8 image
    # norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



if __name__ == "__main__":
    main()

