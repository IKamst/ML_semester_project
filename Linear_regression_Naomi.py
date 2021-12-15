# Translated version of Matlab code basicDemo.m by Naomï Broersma.
#
import re
import numpy as np
from matplotlib import pyplot as plt

# Read the mfeat-pix data and format it nicely.
def read_data():
    with open('mfeat-pix.txt', 'r') as file:
        # Each line corresponds to one digit.
        lines = file.read().splitlines()

    digit_array = []
    for line in lines:
        elem = re.split(r'\s', line)
        elem = list(filter(None, elem))
        digit_array.append(elem)
    return digit_array


# Plots one digit.
def plot_one_digit(digit):
    print(digit)
    array_picture = np.zeros((16, 15))
    cnt = 0
    for column in range(16):
        for row in range(15):
            array_picture[column][row] = digit[cnt]
            cnt = cnt + 1
    plt.imshow(array_picture)
    plt.show()
    return


# Plot a figure of the first 10 digits of each class.
def plot_pixel_data(lines):
    array_pictures = np.zeros((100, 16, 15))
    cnt_index = 0
    for i in range(10):
        for j in range(10):
            picture = lines[200 * i + j]
            cnt = 0
            for column in range(16):
                for row in range(15):
                    array_pictures[cnt_index][column][row] = picture[cnt]
                    cnt = cnt + 1
            print(array_pictures[cnt_index])
            cnt_index = cnt_index + 1
    fig, axes = plt.subplots(10, 10, figsize=(16, 15))
    cnt = 0
    for row in range(10):
        for column in range(10):
            axes[row, column].imshow(array_pictures[cnt])
            cnt = cnt + 1
    plt.show()
    return


# Split the data into a training and testing set.
def split_data(lines):
    return


if __name__ == "__main__":
    digits = read_data()
    plot_one_digit(digits[0])
    plot_pixel_data(digits)
    split_data(digits)
