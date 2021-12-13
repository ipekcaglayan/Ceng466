import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from pprint import pprint


def original_histogram(input_img_path, output_path):
    original_histogram_path = os.path.join(output_path, "original_histogram.png")
    img_grayscale = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    row_num, col_num = img_grayscale.shape

    d = defaultdict(int)
    for r in range(row_num):
        for c in range(col_num):
            value = img_grayscale[r][c]
            d[value] += 1

    pixel_values = list(d.keys())
    number_of_pixels = list(d.values())

    plt.bar(pixel_values, number_of_pixels, color='green')

    plt.xlabel("Intensity Values(rk)")
    plt.ylabel("Number of pixels in the image with intensity rk")
    plt.title("Original Histogram")

    plt.savefig(original_histogram_path)

    return d


def gaussian(m, s, x):
    mu1 = m[0]
    mu2 = m[1]
    sigma1 = s[0]
    sigma2 = s[1]
    g1 = 1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(- (x - mu1) ** 2 / (2 * (sigma1 ** 2)))
    g2 = 1 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(- (x - mu2) ** 2 / (2 * (sigma2 ** 2)))

    return g1+g2


def gaussian_histogram(input_img_path, output_path, m, s):
    gaussian_histogram_path = os.path.join(output_path, "gaussian_histogram.png")
    img_grayscale = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    row_num, col_num = img_grayscale.shape
    pixel_num = row_num * col_num

    total = 0
    for i in range(256):
        p = gaussian(m, s, i)
        total += p
        # print(p)

    gaussian_samples = [gaussian(m, s, i)/total * pixel_num for i in range(256)]
    # total = 0

    new_total = 0
    for i in range(256):
        p = gaussian(m, s, i)/total
        new_total += p


    print(new_total, "probabilities")
    print(sum(gaussian_samples), 2048*2048)


    plt.bar(list(range(256)), gaussian_samples, color='green')

    plt.savefig(gaussian_histogram_path)
    return gaussian_samples


def part1(input_img_path, output_path, m, s):

    exist = os.path.exists(output_path)
    if not exist:
        os.makedirs(output_path)

    original_histogram_dict = original_histogram(input_img_path, output_path)
    plt.clf()
    gaussian_samples = gaussian_histogram(input_img_path, output_path, m, s)

    # Matching

    img_grayscale = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    row_num, col_num = img_grayscale.shape
    pixel_num = row_num * col_num

    cumulative_original_histogram_dict = dict()

    cumulative_desired_histogram_dict = dict()

    cumulative_original_histogram_dict[0] = original_histogram_dict[0]
    cumulative_desired_histogram_dict[0] = gaussian_samples[0]

    for key in range(1, 256):
        cumulative_desired_histogram_dict[key] = gaussian_samples[key] + \
                                                  cumulative_desired_histogram_dict[key-1]
        cumulative_original_histogram_dict[key] = original_histogram_dict[key] + cumulative_original_histogram_dict[key-1]

    original_sk = dict()
    desired_sk = dict()

    for k in range(256):
        o_sk = (255/pixel_num)*cumulative_original_histogram_dict[k]
        original_sk[k] = o_sk
        d_sk = (255/pixel_num)*cumulative_desired_histogram_dict[k]
        desired_sk[k] = d_sk

    desired_values = dict()
    for k in range(256):
        old_val = None
        old_index = None
        sk = original_sk[k]
        for z in range(256):
            val = desired_sk[z]
            if old_val and abs(old_val-sk)> abs(val-sk):
                old_val = val
                old_index = z
            if not old_val:
                old_val = val
                old_index = z

        desired_values[k] = old_index

    processed_img = []

    matched_histogram = [0 for i in range(256)]

    for r in range(row_num):
        new_row = []
        for c in range(col_num):
            val = desired_values[img_grayscale[r][c]]
            matched_histogram[val] += 1
            new_row.append(val)

        processed_img.append(new_row)

    plt.clf()
    plt.bar(list(range(256)), matched_histogram, color='green')

    matched_image_histogram_path = os.path.join(output_path, "matched_image_histogram.png")

    plt.savefig(matched_image_histogram_path)

    processed_img = np.array(processed_img)

    cv2.imwrite(os.path.join(output_path, "matched_image.png"), np.array(processed_img))

    return processed_img


def the1_convolution(input_img_path, filter):
    img_grayscale = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)

    processed_img = []
    row_num, col_num = img_grayscale.shape
    f_row = len(filter)
    f_col = len(filter[0])
    row_offset = list(range(-int(f_row/2), int(f_row/2) + 1))
    col_offset = list(range(-int(f_col/2), int(f_col/2) + 1))
    offset = list(itertools.product(row_offset, col_offset))
    f_center_row = int(f_row/2)
    f_center_col = int(f_col/2)

    for r in range(row_num):
        new_row = []
        for c in range(col_num):
            val = 0
            for x, y in offset:
                if r+x >=0 and r+x < row_num and c+y >=0 and c+y < col_num:
                    val += img_grayscale[r+x][c+y]*filter[f_center_row+x][f_center_col+y]

            new_row.append(val)

        processed_img.append(new_row)

    processed_img = np.array(processed_img)
    # status = cv2.imwrite('/Users/ipekcaglayan/Desktop/Ceng466/THE/THE1/outputs/conv-result.png', np.array(processed_img))

    # print("Image written to file-system : ", status)

    return processed_img









