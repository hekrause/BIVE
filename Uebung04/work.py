from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def filter_img(img, mask, offset):

    copy_img = img
    convolution_mask = get_convolution_mask(mask)

    for x in range(0, 2):
        for y in range(0, 1):
            points = get_points(x, y, convolution_mask)
            valid_points, count = validate_points(img.height, img.width, points)


    return img


def get_convolution_mask(mask):
    y, x = mask.shape

    convolution_mask = np.zeros(shape=mask.shape, dtype=object)

    for i in range(0, x):
        for j in range(0, y):
            new_row = (((y - 1) / -2) + i)
            new_column = (((x - 1) / -2) + j)
            convolution_mask[j, i] = (new_row, new_column)

    #print("Convolution Mask", convolution_mask)
    return convolution_mask


def get_points(x, y, convolution):
    y_c, x_c = convolution.shape
    points = np.zeros(shape=(x_c * y_c), dtype=object)
    count = 0

    for i in range(0, x_c):
        for j in range(0, y_c):
            points[count] = (x + convolution[j, i][0], y + convolution[j, i][1])
            count += 1

    #print("Points", points)
    return points


def validate_points(height, width, points):
    valid_points = np.zeros(shape=points.size, dtype=object)
    count = 0

    for i in range(0, points.size):
        if points[i][0] >= 0 and points[i][0] < height and points[i][1] >= 0 and points[i][1] < width:
            valid_points[count] = points[i];
            count += 1

    print("Valid Points", valid_points)
    return valid_points, count


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

    # convert to grayscale
    img = rgb2gray(img)

    mask = np.array([[1, 1, 1], [1, 3, 1], [1, 1, 1]])

    filter_img(img, mask, 0)

'''
    # read img
    img = Image.open("bild01.jpg")

    # convert to grayscale
    img = rgb2gray(img)

    # plot new img
    plt.figure(1)
    plt.subplot(212)
    plt.imshow(img_new, cmap=cm.Greys_r)

    plt.show()
'''