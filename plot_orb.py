from skimage import data
from PIL import Image
import numpy as np
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

jpg = Image.open(r"C:\Users\User\Downloads\Telegram Desktop\day1.jpg")
jpg1 = Image.open(r"C:\Users\User\Downloads\Telegram Desktop\night1.jpg")

MatImg=np.array(jpg)
MatImg1=np.array(jpg1)
img1 = rgb2gray(MatImg)
img2 = rgb2gray(MatImg1)


descriptor_extractor = ORB(n_keypoints=50)

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors


matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)


fig, ax = plt.subplots()

plt.gray()

plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
ax.axis('off')
ax.set_title("Lit Image vs. Dark Image : ORB")


plt.show()
