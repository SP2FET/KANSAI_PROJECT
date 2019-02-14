"""Single image dehazing."""
from __future__ import division
import cv2
import numpy as np
from PIL import Image, ImageEnhance


class Channel_value:
    val = -1.0
    intensity = -1.0


def find_intensity_of_atmospheric_light(img, gray):
    top_num = int(img.shape[0] * img.shape[1] * 0.001)
    toplist = [Channel_value()] * top_num
    dark_channel = find_dark_channel(img)

    for y in range(img.shape[0]):
        print("\rfind intensity...{}%".format(int((y / img.shape[0]) * 100)), end="")
        for x in range(img.shape[1]):
            val = img.item(y, x, dark_channel)
            intensity = gray.item(y, x)
            for t in toplist:
                if t.val < val or (t.val == val and t.intensity < intensity):
                    t.val = val
                    t.intensity = intensity
                    break

    print("\r", end="")

    max_channel = Channel_value()
    for t in toplist:
        if t.intensity > max_channel.intensity:
            max_channel = t

    return max_channel.intensity


def find_dark_channel(img):
    return np.unravel_index(np.argmin(img), img.shape)[2]


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))


def dehaze(img, light_intensity, windowSize, t0, w):
    size = (img.shape[0], img.shape[1])

    outimg = np.zeros(img.shape, img.dtype)

    for y in range(size[0]):
        print("\rdehaze...{}%".format(int((y / size[0]) * 100)), end="")
        for x in range(size[1]):
            x_low = max(x-(windowSize//2), 0)
            y_low = max(y-(windowSize//2), 0)
            x_high = min(x+(windowSize//2), size[1])
            y_high = min(y+(windowSize//2), size[0])

            sliceimg = img[y_low:y_high, x_low:x_high]

            dark_channel = find_dark_channel(sliceimg)
            t = 1.0 - (w * img.item(y, x, dark_channel) / light_intensity)

            outimg.itemset((y,x,0), clamp(0, ((img.item(y,x,0) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,1), clamp(0, ((img.item(y,x,1) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,2), clamp(0, ((img.item(y,x,2) - light_intensity) / max(t, t0) + light_intensity), 255))
    print("\r", end="")

    return outimg



def dehazing(frame, w, t0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    light_intensity = find_intensity_of_atmospheric_light(frame, gray)
    print("find intensity...done")

    outimg = dehaze(frame, light_intensity, 20, t0, w)
    print("dehaze...done")

    return outimg



def cv_get_concat_h(im1, im2):

    im1, im2 = Image.fromarray(im1), Image.fromarray(im2)

    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))

    dst = np.asarray(dst)

    return dst



def cv_color_adjustment(img, saturation, contrast, brightness, sharpness):

    img = Image.fromarray(img)

    # 彩度 (saturation)
    if saturation != 1.0:
        con6 = ImageEnhance.Color(img)
        img = con6.enhance(saturation)

    #コントラスト (contrast)
    if contrast != 1.0:
        con7 = ImageEnhance.Contrast(img)
        img = con7.enhance(contrast)

    #明度 (brightness)
    if brightness != 1.0:
        con9 = ImageEnhance.Brightness(img)
        img = con9.enhance(brightness)

    #シャープネス (sharpness)
    if sharpness != 1.0:
        con11 = ImageEnhance.Sharpness(img)
        img = con11.enhance(sharpness)

    img = np.asarray(img)

    return img



def mark_orb2(img1, img2):

    # ORB (Oriented FAST and Rotated BRIEF)
    detector = cv2.ORB_create()

    # 特徴検出
    keypoints1 = detector.detect(img1)
    keypoints2 = detector.detect(img2)

    print("\rkp = {0}  kp = {1}".format(len(keypoints1), len(keypoints2)), end="")

    # 画像への特徴点の書き込み
    out1 = cv2.drawKeypoints(img1, keypoints1, None)
    out2 = cv2.drawKeypoints(img2, keypoints2, None)

    return out1, out2


def main():
    cap = cv2.VideoCapture('1.h264')

    while(True):
        ret, frame = cap.read()

        outimg = cv_color_adjustment(frame, saturation=1.0, contrast=1.5, brightness=1.0, sharpness=15.0)

        frame, outimg = mark_orb2(frame, outimg)

        compare_img = cv_get_concat_h(frame, outimg)

        cv2.imshow('name', compare_img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# def main():
#     #cap = cv2.VideoCapture('1.h264')
#
#
#     #     ret, frame = cap.read()
#     frame = cv2.imread("someFrame.png")
#     print("imread...done")
#
#     #outimg = dehazing(frame, w = 0.95, t0 = 0.55)
#
#     #defaults of each variables are 1.0
#     outimg = cv_color_adjustment(frame, saturation = 0.0, contrast = 1.5, brightness = 1.0, sharpness = 15.0)
#
#     print("-------------------------------")
#     frame = mark_orb(frame, 1)
#     outimg = mark_orb(outimg, 2)
#
#     compare_img = cv_get_concat_h(frame, outimg)
#
#     cv2.imshow('name', compare_img)
#     cv2.waitKey(0)


if __name__ == "__main__": main()