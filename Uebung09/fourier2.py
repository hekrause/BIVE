from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal


def ibv_fft2(img, windowing):
    # Lab exercise 'Industrial Image Processing'
    ViewMode = 2

    # build 2d window
    img_x = img.shape[1]
    img_y = img.shape[0]

    if windowing == 'hanning':
        u_vector = np.hanning(img_x) #single row
        v_vector = np.hanning(img_y) #single column

    elif windowing == 'bartlett':
        u_vector = np.bartlett(img_x) #single row
        v_vector = np.bartlett(img_y) #single column

    elif windowing == 'parzen':
        u_vector = signal.parzen(img_x) #single row
        v_vector = signal.parzen(img_y) #single column

    elif windowing == 'gaussian':
        u_vector = signal.gaussian(img_x, std=img_x/6) #single row
        v_vector = signal.gaussian(img_y, std=img_y/6) #single column

    else:
        u_vector = np.ones(img_x) #single row
        v_vector = np.ones(img_y) #single column

    window = np.zeros(img.shape)
    img_windowed = np.zeros(img.shape)

    # 2d window and windowed image
    for u in range(img_x):
        for v in range(img_y):
            window[v,u] = np.multiply(u_vector[u], v_vector[v])*255
            img_windowed[v,u] = np.multiply(np.float64(img[v,u]), window[v,u])/255

    ## fft of img_windowed
    spectrum = np.fft.fft2(img_windowed)
    # centered spectrum
    if ViewMode == 2:
        spectrum_shifted = np.fft.fftshift(spectrum)
    else:
        spectrum_shifted = spectrum # not shifted

    # magnitude, real and imaginary part
    magnitude_spectrum = np.log(np.absolute(spectrum_shifted))#20*np.log(np.abs(spectrum_shifted))
    real_spectrum = np.log(np.abs(spectrum_shifted.real))
    imag_spectrum = np.log(np.abs(spectrum_shifted.imag))

    if ViewMode == 1:
        x_start = 0
        x_end = img_x
        y_start = 0
        y_end = img_y

    elif ViewMode == 2:
        x_start = - img_x/2
        x_end = img_x/2
        y_start = - img_y/2
        y_end = img_y/2


    plt.figure(1)
    # number of values plotted at the x- and y-axis
    max_xticks = 5
    xloc = plt.MaxNLocator(max_xticks)
    max_yticks = 8
    yloc = plt.MaxNLocator(max_yticks)

    # signal
    ax = plt.subplot(221)
    ax.set_title("signal")
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(yloc)
    ax.imshow(img_windowed, cmap = cm.Greys_r)

    # amplitude spectrum
    ax = plt.subplot(222)
    ax.set_title("amplitude spectrum")
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(yloc)
    ax.imshow(magnitude_spectrum, cmap = cm.Greys_r,extent=(x_start,x_end,y_start,y_end))

    # real spectrum
    ax = plt.subplot(223)
    ax.set_title("real spectrum")
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(yloc)
    ax.imshow(real_spectrum, cmap = cm.Greys_r,extent=(x_start,x_end,y_start,y_end))

    # imaginary spectrum
    ax = plt.subplot(224)
    ax.set_title("imaginary spectrum")
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(yloc)
    ax.imshow(imag_spectrum, cmap = cm.Greys_r,extent=(x_start,x_end,y_start,y_end))

    plt.show()


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


if __name__ == "__main__":

    # read image
    img = Image.open("shutter.jpg")

    # convert to numpy array
    img = np.array(img)

    # convert to grayscale - use for print.jpg and bild01.jpg
    #img = rgb2gray(img)

    ibv_fft2(img, None)

    print("Finish.")
