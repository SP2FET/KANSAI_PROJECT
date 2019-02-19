from __future__ import division
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageChops
from matplotlib import pyplot as plt


img = cv2.imread("img02.png")
#gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

w,h = gray.shape

stepi = int(np.floor(h / 15.0))
stepj = int(np.floor(w / 15.0))



gray = Image.fromarray(gray)


plt.imshow(gray)
plt.show()

out = Image.new('L', (w,h), 0)


def eeeehh(i, j, hist):
    minx = max(0, j - stepj)
    maxx = min(w, j + stepj)
    miny = max(0, i - stepi)
    maxy = min(h, i + stepi)


    big_hist = gray.crop((minx, miny, maxx, maxy)).histogram()


    sub_hist = map(int.__sub__, big_hist, hist)

    avg = 0
    for i in range(255):
        avg = avg + hist[i] * i
    avg = int(avg / sum(hist))

    inv_avg = int(255 - avg)

    crop_img = gray.crop((j, i, j + stepj, i + stepi ))

    avg_img = Image.new('L', (stepj, stepi), color = (inv_avg))

    #new_img = crop_img.point(lambda x: x - inv_avg)
    new_img = ImageChops.subtract(crop_img, avg_img)

    #plt.imshow(new_img)
    #plt.show()

    return new_img



for i in range(0, w, stepi):
    for j in range(0, h, stepj):

        crop_img = gray.crop((j, i, j + stepj, i + stepi ))


        #np.histogram(np.array(crop_img.flatten(), bins = np.arange(256 + 1)))
        hist = crop_img.histogram()
        hist = hist[220:256]
        hist_sum = sum(hist) / stepi / stepj

        if(hist_sum > 0.65):
            crop_img = eeeehh(i,j, crop_img.histogram())
        out.paste(crop_img, (j, i))





plt.imshow(out)
plt.show()