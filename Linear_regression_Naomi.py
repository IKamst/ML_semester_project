# Translated version of Matlab code basicDemo.m by Naomï Broersma.
import re
import numpy as np


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
    print(digit_array)
    return digit_array


# Plot a figure of the first 10 digits of each class.
def plot_pixel_data(lines):
    return


# Split the data into a training and testing set.
def split_data(lines):
    return


if __name__ == "__main__":
    digits = read_data()
    plot_pixel_data(digits)
    split_data(digits)
