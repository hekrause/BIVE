from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_cumHisto(img, binSize=1):

    pass

    return histo



def match_Histo(img_histo, ref_histo):

    #img_histo . . . original histogram
    #ref_histo . . . reference histogram
    #returns the mapping function LUT to be applied to the image
    pass

    return LUT


def apply_LUT(img, lut):

    pass

    return img



def rgb2gray(rgb):

    pass

    return gray


if __name__ == "__main__":

    # read img
    im = Image.open("bild01.jpg")
    ref = Image.open("bild02.jpg")

    # convert to numpy array

    # convert to grayscale

    # compute histograms
    # histo_im
    # histo_ref

    # compute mapping function (LUT) for matching histograms

    # compute new image with lut
    # im_new

    # compute new histogram of new image
    # histo_new

    # plot information
    N = histo_new.size
    x = range(N)
    width = 1

    # plot histogram of new image
    plt.figure(1)
    plt.subplot(211)
    plt.bar(x, histo_new, width, color="blue")
    plt.xlim([0,N-1])
    # plot new img
    plt.figure(1)
    plt.subplot(212)
    plt.imshow(im_new, cmap = cm.Greys_r)

    # plot reference histogram
    plt.figure(2)
    plt.subplot(211)
    plt.bar(x, histo_ref, width, color="blue")
    plt.xlim([0,N-1])
    # plot reference image
    plt.figure(2)
    plt.subplot(212)
    plt.imshow(ref, cmap = cm.Greys_r)

    plt.show()

