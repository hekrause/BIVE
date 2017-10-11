from builtins import print

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_Histo(img):
    
    histo = np.zeros(shape=(256))

    for x in range(0, img.width):
        for y in range(0, img.height):
            brightness = round(img.getpixel((x, y)), 0)
            histo[brightness] += 1

    return histo


def bin_Histo(img, bin=1):
    intervalls = np.ceil(256 / bin)
    histo = np.zeros(shape=intervalls, dtype=np.int)

    for x in range(0, img.width):
        for y in range(0, img.height):
            brightness = round(img.getpixel((x, y)), 0)
            index = (brightness * intervalls) / 256
            histo[index] += 1

    return histo


def brighten(img, offset):
    # add offset to img
    for x in range(0, img.width):
        for y in range(0, img.height):
            edit = img.getpixel((x, y)) + offset
            # check clamping
            if edit > 255 : edit = 255
            img.putpixel((x, y), edit)

    return img


def get_lut(k=256):
    # create lut-table
    # which only brightens the darker pixel values (e.g. < 200)
    # bright pixel values should not change that much
    lut = np.zeros(shape=256, dtype=np.int)

    for i in range(0, 256):
        lut[i] = i
        if i < 200:
            # check clamping
            if (i + k) > 255:
                lut[i] = 255
            else:
                lut[i] = i + k

    return lut


def brighten_with_lut(img, lut):
    for x in range(0, img.width):
        for y in range(0, img.height):
            edit = img.getpixel((x, y))
            img.putpixel((x, y), lut[edit].item())

    return img


def rgb2gray(img):
    # convert to grayscale image (only one channel)
    return img.convert('L')


if __name__ == "__main__":
    # read img
    img = Image.open("bild01.jpg")

    # convert to grayscale
    img = rgb2gray(img)

    # brighten image
    #img = brighten(img, 0)

    # brighten image with lut-table
    img = brighten_with_lut(img, get_lut(50))

    # compute histogram (with bin-size)
    #histo = bin_Histo(img, 1)

    # convert to numpy array
    histo = compute_Histo(img)

    # plot histogram
    N = histo.size
    x = range(N)
    width = 1

    plt.figure(1)
    plt.subplot(211)
    plt.bar(x, histo, width, color="blue")
    plt.xlim([0,N-1])

    # plot processed img
    plt.subplot(212)
    plt.imshow(img, cmap = cm.Greys_r)

    plt.show()