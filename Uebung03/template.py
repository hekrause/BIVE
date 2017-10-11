from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_cumHisto(img, binSize=1):

    return np.cumsum(bin_Histo(img, binSize))


def bin_Histo(img, bin=1):
    intervalls = np.ceil(256 / bin)
    histo = np.zeros(shape=intervalls)

    for x in range(0, img.width):
        for y in range(0, img.height):
            brightness = round(img.getpixel((x, y)), 0)
            index = (brightness * intervalls) / 256
            histo[index] += 1

    return histo


def match_Histo(img_histo, ref_histo):
    #img_histo . . . original histogram
    #ref_histo . . . reference histogram
    #returns the mapping function LUT to be applied to the image

    LUT = np.zeros(shape=256, dtype=np.int)

    for a in range(0, 256):
        j = 256 - 1
        while (j >= 0 and img_histo[a] <= ref_histo[j]):
            LUT[a] = j
            j -= 1

    return LUT
'''
    for i in range(0, 256):
        P_i = img_histo[i] / img_histo[255]
        for j in range(0, 256):
            P_j = ref_histo[j] / ref_histo[255]

            if j < 255:
                P_j_next = ref_histo[j + 1] / ref_histo[255]
            else:
                P_j_next = 1

            if P_i > P_j and P_i <= P_j_next:
                difference_i_j = P_i - P_j
                difference_i_j_next = P_j_next - P_i
                if difference_i_j < difference_i_j_next:
                    LUT[i] = j
                else:
                    LUT[i] = clamping(j + 1)
                    
    return LUT                
'''

def apply_LUT(img, lut):

    for x in range(0, img.width):
        for y in range(0, img.height):
            edit = img.getpixel((x, y))
            lut_edit = lut[edit].item()
            img.putpixel((x, y), lut_edit)

    return img


def clamping(value):
    if value > 255:
        return 255
    else:
        return value

def rgb2gray(rgb):
    # convert to grayscale image (only one channel)
    return rgb.convert('L')


if __name__ == "__main__":

    # read img
    img = Image.open("bild01.jpg")
    ref = Image.open("bild02.jpg")

    # convert to grayscale
    img = rgb2gray(img)
    ref = rgb2gray(ref)

    # compute histograms
    img_histo = compute_cumHisto(img, 1)
    ref_histo = compute_cumHisto(ref, 1)

    # compute mapping function (LUT) for matching histograms
    LUT = match_Histo(img_histo, ref_histo)

    # compute new image with lut
    img_new = apply_LUT(img, LUT)

    # compute new histogram of new image
    histo_new = compute_cumHisto(img_new, 1)

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
    plt.imshow(img_new, cmap = cm.Greys_r)

    # plot reference histogram
    plt.figure(2)
    plt.subplot(211)
    plt.bar(x, ref_histo, width, color="blue")
    plt.xlim([0,N-1])
    # plot reference image
    plt.figure(2)
    plt.subplot(212)
    plt.imshow(ref, cmap = cm.Greys_r)
    
    plt.show()