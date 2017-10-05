from PIL import Image, ImageOps

import numpy as np


def print_info(img):
    print(img.format)
    print(img.mode)
    print(img.size)
    print(img.width)
    print(img.height)
    print(img.palette)
    print(img.info)


def load_image(filename):
    img = Image.open(filename)
    img.load()

    return img


def make_array(img):
    return np.array(img)


def split_channels(np_array):
    np_array_red = np_array.copy()
    np_array_blue = np_array.copy()
    np_array_green = np_array.copy()

    np_array_blue[:, :, 0] *= 0
    np_array_blue[:, :, 1] *= 0
    blue = Image.fromarray(np_array_blue)

    np_array_green[:, :, 1] *= 0
    np_array_green[:, :, 2] *= 0
    green = Image.fromarray(np_array_green)

    np_array_red[:, :, 0] *= 0
    np_array_red[:, :, 2] *= 0
    red = Image.fromarray(np_array_red)

    return blue, green, red


def transform(img, mode):
    if mode is "vertical":
        return ImageOps.flip(img)
    elif mode is "horizontal":
        return ImageOps.mirror(img)
    else:
        return img

if __name__ == "__main__":
    # 2.Bild einlesen
    image = load_image("hidden.png")

    # 3.Bild zu Numpy-Array konvertieren
    array = make_array(image)

    # 4. Bild nach Farb-Kanälen getrennt ausgeben
    blue, green, red = split_channels(array)

    # 5.Funktion zum vertikal/horizontal spiegeln
    t_image = transform(image, "vertical")

    t_image.show()
    blue.show()
    green.show()
    red.show()
