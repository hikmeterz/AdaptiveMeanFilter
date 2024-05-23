import math
import sys

import cv2
import statistics
import numpy as np
import matplotlib.pyplot as plt
def question_1():

    img = cv2.imread('./noisyImage_Gaussian.jpg',0)

    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    adp_mean_filter_img = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT) #5x5 filter kernel size padding 2

    #print(img[4][4])
    #print(adp_mean_filter_img[6][6])
    variance_of_noise = 0.006

    #original imagein x-2,y-2. indexindeki pixel adp_mean_filterin x,y indexine denk gelmektedir.Paddingten dolayi.

    h, w = len(adp_mean_filter_img), len(adp_mean_filter_img[0])
    #print(h,w)
    for x in range(2,h-2):
        for y in range(2,w-2):
            kernel=[
                adp_mean_filter_img[x-2][y-2],
                adp_mean_filter_img[x-2][y-1],
                adp_mean_filter_img[x-2][y],
                adp_mean_filter_img[x-2][y+1],
                adp_mean_filter_img[x-2][y+2],

                adp_mean_filter_img[x-1][y-2],
                adp_mean_filter_img[x-1][y-1],
                adp_mean_filter_img[x-1][y],
                adp_mean_filter_img[x-1][y+1],
                adp_mean_filter_img[x-1][y+2],

                adp_mean_filter_img[x][y-2],
                adp_mean_filter_img[x][y-1],
                adp_mean_filter_img[x][y],
                adp_mean_filter_img[x][y+1],
                adp_mean_filter_img[x][y+2],

                adp_mean_filter_img[x+1][y-2],
                adp_mean_filter_img[x+1][y-1],
                adp_mean_filter_img[x+1][y],
                adp_mean_filter_img[x+1][y+1],
                adp_mean_filter_img[x+1][y+2],

                adp_mean_filter_img[x+2][y-2],
                adp_mean_filter_img[x+2][y-1],
                adp_mean_filter_img[x+2][y],
                adp_mean_filter_img[x+2][y+1],
                adp_mean_filter_img[x+2][y+2],
            ]
            local_variance_of_kernel = statistics.variance(kernel)
            local_avg_of_kernel = sum(kernel) / len(kernel)
            adp_mean_filter_img[x][y] = img[x-2][y-2] - (variance_of_noise / local_variance_of_kernel) * ( (img[x-2][y-2]) - local_avg_of_kernel )


    output_1_1 = adp_mean_filter_img#cv2.copyMakeBorder(adp_mean_filter_img, 0, 0, 0, 0, borderType=cv2.BORDER_CONSTANT)


    output_1_2  = cv2.blur(output_1_1, (5, 5))

    output_1_3 = cv2.GaussianBlur(output_1_1, (5, 5), 0)


    original_img = cv2.imread('./lena_grayscale_hq.jpg', 0)
    original_img = cv2.copyMakeBorder(original_img, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT)
    original_img = cv2.normalize(original_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    x1 = cv2.PSNR(original_img, output_1_1)
    x2 = cv2.PSNR(original_img, output_1_2)
    x3 = cv2.PSNR(original_img, output_1_3)

    print(x1)
    print(x2)
    print(x3)


def local_average_intensity_in_kernel(img,x,y):
    kernel = []
    for i in range(x-2,x+3):
        for j in range(y-2,y+3):
            kernel.append(img[x][y])


    avg = sum(kernel) / len(kernel)
    return avg
def varience_of_intensities_in_kernel(img,x,y):

    #kernel= img[x-2:x,y-2:y]

    #print(type(img[x][y]))

    kernel = []
    for i in range(x-2,x+3):
        for j in range(y-2,y+3):
            kernel.append(img[i][j])
            #print(img[i][j] , i , j )

    var = statistics.variance(kernel)
    return var



def question_2():


    img = cv2.imread('./noisyImage.jpg', 0)
    adp_mean_filter_img = cv2.copyMakeBorder(img,3,3,3,3,borderType=cv2.BORDER_CONSTANT)
    h, w = len(adp_mean_filter_img), len(adp_mean_filter_img[0])
    # print(h,w)
    kernel_size=3
    for x in range(3, h - 3):
        for y in range(3, w - 3):
            z_x_y = adp_mean_filter_img[x][y]
            z_min = find_min(adp_mean_filter_img,x,y,kernel_size)
            z_max = find_max(adp_mean_filter_img,x,y,kernel_size)
            z_med = find_median(adp_mean_filter_img,x,y,kernel_size)
            zxys = [z_min ,z_med,z_max]
            zxys_sorted = np.sort(zxys)
            if(np.array_equal(zxys,zxys_sorted) ):#level A
                #levelB
                zxys1 = [z_min,z_x_y,z_max]
                zxys_sorted1 = np.sort(zxys1)
                if (np.array_equal(zxys1, zxys_sorted1)):
                    adp_mean_filter_img[x][y] = z_x_y
                else:
                    adp_mean_filter_img[x][y] = z_med
            else:
                #level A nin devami
                kernel_size+=2
                if(kernel_size < 7):
                    y-=1 # repeat level A
                else:
                    adp_mean_filter_img[x][y] = z_med
                    kernel_size=3


    output_2_1 = adp_mean_filter_img
    output_2_2 = cv2.medianBlur(output_2_1,3)
    output_2_3 = cv2.medianBlur(output_2_1, 5)
    output_2_4 = cv2.medianBlur(output_2_1, 7)
    output_2_5 = center_weighted_median_filter(output_2_1,3)
    output_2_6 = center_weighted_median_filter(output_2_1, 5)
    output_2_7 = center_weighted_median_filter(output_2_1, 7)

    original_img_wb = cv2.imread('./lena_grayscale_hq.jpg', 0)
    original_img = cv2.copyMakeBorder(original_img_wb, 3, 3, 3, 3, borderType=cv2.BORDER_CONSTANT)
    x1 = cv2.PSNR(original_img, output_2_1)
    x2 = cv2.PSNR(original_img, output_2_2)
    x3 = cv2.PSNR(original_img, output_2_3)
    x4 = cv2.PSNR(original_img, output_2_4)
    x5 = cv2.PSNR(original_img, output_2_5)
    x6 = cv2.PSNR(original_img, output_2_6)
    x7 = cv2.PSNR(original_img, output_2_7)

    print(x1)
    print(x2)
    print(x3)
    print(x4)
    print(x5)
    print(x6)
    print(x7)


def center_weighted_median_filter(img,kernel_size):
    center_weighted_median_img = img.copy()  #cv2.copyMakeBorder(img, 3, 3, 3, 3, borderType=cv2.BORDER_CONSTANT)
    h, w = len(center_weighted_median_img), len(center_weighted_median_img[0])
    for x in range(3, h - 3):
        for y in range(3, w - 3):
            centered_wm=w_median_calculate(center_weighted_median_img,x,y,kernel_size)
            center_weighted_median_img[x][y]=centered_wm
            #print(center_weighted_median_img[x][y])

    return center_weighted_median_img

def w_median_calculate(adp_mean_filter_img,x,y,kernel_size):

    if (kernel_size == 3):
        kernel = [
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],

            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],

            adp_mean_filter_img[x + 1][y - 1],
            adp_mean_filter_img[x + 1][y],
            adp_mean_filter_img[x + 1][y + 1],

        ]

    elif (kernel_size == 5):
        kernel = [
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],

            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],

            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],

            adp_mean_filter_img[x + 1][y - 2],
            adp_mean_filter_img[x + 1][y - 1],
            adp_mean_filter_img[x + 1][y],
            adp_mean_filter_img[x + 1][y + 1],
            adp_mean_filter_img[x + 1][y + 2],

            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
        ]
    elif (kernel_size == 7):
        kernel = [
            adp_mean_filter_img[x - 3][y - 3],
            adp_mean_filter_img[x - 3][y - 2],
            adp_mean_filter_img[x - 3][y - 1],
            adp_mean_filter_img[x - 3][y],
            adp_mean_filter_img[x - 3][y + 1],
            adp_mean_filter_img[x - 3][y + 2],
            adp_mean_filter_img[x - 3][y - 3],

            adp_mean_filter_img[x - 2][y - 3],
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],
            adp_mean_filter_img[x - 2][y - 3],

            adp_mean_filter_img[x - 1][y - 3],
            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],
            adp_mean_filter_img[x - 1][y - 3],

            adp_mean_filter_img[x][y - 3],
            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],
            adp_mean_filter_img[x][y - 3],

            adp_mean_filter_img[x+1][y - 3],
            adp_mean_filter_img[x+1][y - 2],
            adp_mean_filter_img[x+1][y - 1],
            adp_mean_filter_img[x+1][y],
            adp_mean_filter_img[x+1][y + 1],
            adp_mean_filter_img[x+1][y + 2],
            adp_mean_filter_img[x+1][y - 3],

            adp_mean_filter_img[x + 2][y - 3],
            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
            adp_mean_filter_img[x + 2][y - 3],

            adp_mean_filter_img[x + 3][y - 3],
            adp_mean_filter_img[x + 3][y - 2],
            adp_mean_filter_img[x + 3][y - 1],
            adp_mean_filter_img[x + 3][y],
            adp_mean_filter_img[x + 3][y + 1],
            adp_mean_filter_img[x + 3][y + 2],
            adp_mean_filter_img[x + 3][y - 3],

        ]
    median = np.median(kernel)
    return median

def find_median(adp_mean_filter_img,x,y,kernel_size):
    median=0
    if(kernel_size==3):
        kernel = [
            adp_mean_filter_img[x - 1][y-1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y+1],

            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x ][y + 1],

            adp_mean_filter_img[x +1][y-1],
            adp_mean_filter_img[x +1][y],
            adp_mean_filter_img[x +1][y+1],

        ]
        median = np.median(kernel)
    elif(kernel_size==5):
        kernel = [
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],

            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],

            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],

            adp_mean_filter_img[x + 1][y - 2],
            adp_mean_filter_img[x + 1][y - 1],
            adp_mean_filter_img[x + 1][y],
            adp_mean_filter_img[x + 1][y + 1],
            adp_mean_filter_img[x + 1][y + 2],

            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
        ]
        median = np.median(kernel)
    elif (kernel_size == 7):
        kernel = [
            adp_mean_filter_img[x - 3][y - 3],
            adp_mean_filter_img[x - 3][y - 2],
            adp_mean_filter_img[x - 3][y - 1],
            adp_mean_filter_img[x - 3][y],
            adp_mean_filter_img[x - 3][y + 1],
            adp_mean_filter_img[x - 3][y + 2],
            adp_mean_filter_img[x - 3][y - 3],

            adp_mean_filter_img[x - 2][y - 3],
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],
            adp_mean_filter_img[x - 2][y - 3],

            adp_mean_filter_img[x - 1][y - 3],
            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],
            adp_mean_filter_img[x - 1][y - 3],

            adp_mean_filter_img[x][y - 3],
            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],
            adp_mean_filter_img[x][y - 3],

            adp_mean_filter_img[x+1][y - 3],
            adp_mean_filter_img[x+1][y - 2],
            adp_mean_filter_img[x+1][y - 1],
            adp_mean_filter_img[x+1][y],
            adp_mean_filter_img[x+1][y + 1],
            adp_mean_filter_img[x+1][y + 2],
            adp_mean_filter_img[x+1][y - 3],

            adp_mean_filter_img[x + 2][y - 3],
            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
            adp_mean_filter_img[x + 2][y - 3],

            adp_mean_filter_img[x + 3][y - 3],
            adp_mean_filter_img[x + 3][y - 2],
            adp_mean_filter_img[x + 3][y - 1],
            adp_mean_filter_img[x + 3][y],
            adp_mean_filter_img[x + 3][y + 1],
            adp_mean_filter_img[x + 3][y + 2],
            adp_mean_filter_img[x + 3][y - 3],

        ]
        median = np.median(kernel)

    return median

def find_max(adp_mean_filter_img,x,y,kernel_size):
    maximum=0
    if(kernel_size==3):
        kernel = [
            adp_mean_filter_img[x - 1][y-1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y+1],

            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x ][y + 1],

            adp_mean_filter_img[x +1][y-1],
            adp_mean_filter_img[x +1][y],
            adp_mean_filter_img[x +1][y+1],

        ]
        maximum = max(kernel)
    elif(kernel_size==5):
        kernel = [
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],

            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],

            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],

            adp_mean_filter_img[x + 1][y - 2],
            adp_mean_filter_img[x + 1][y - 1],
            adp_mean_filter_img[x + 1][y],
            adp_mean_filter_img[x + 1][y + 1],
            adp_mean_filter_img[x + 1][y + 2],

            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
        ]
        maximum = max(kernel)
    elif (kernel_size == 7):
        kernel = [
            adp_mean_filter_img[x - 3][y - 3],
            adp_mean_filter_img[x - 3][y - 2],
            adp_mean_filter_img[x - 3][y - 1],
            adp_mean_filter_img[x - 3][y],
            adp_mean_filter_img[x - 3][y + 1],
            adp_mean_filter_img[x - 3][y + 2],
            adp_mean_filter_img[x - 3][y - 3],

            adp_mean_filter_img[x - 2][y - 3],
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],
            adp_mean_filter_img[x - 2][y - 3],

            adp_mean_filter_img[x - 1][y - 3],
            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],
            adp_mean_filter_img[x - 1][y - 3],

            adp_mean_filter_img[x][y - 3],
            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],
            adp_mean_filter_img[x][y - 3],

            adp_mean_filter_img[x+1][y - 3],
            adp_mean_filter_img[x+1][y - 2],
            adp_mean_filter_img[x+1][y - 1],
            adp_mean_filter_img[x+1][y],
            adp_mean_filter_img[x+1][y + 1],
            adp_mean_filter_img[x+1][y + 2],
            adp_mean_filter_img[x+1][y - 3],

            adp_mean_filter_img[x + 2][y - 3],
            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
            adp_mean_filter_img[x + 2][y - 3],

            adp_mean_filter_img[x + 3][y - 3],
            adp_mean_filter_img[x + 3][y - 2],
            adp_mean_filter_img[x + 3][y - 1],
            adp_mean_filter_img[x + 3][y],
            adp_mean_filter_img[x + 3][y + 1],
            adp_mean_filter_img[x + 3][y + 2],
            adp_mean_filter_img[x + 3][y - 3],

        ]
        maximum = max(kernel)

    return maximum


def find_min(adp_mean_filter_img,x,y,kernel_size):
    minimum=0
    if(kernel_size==3):
        kernel = [
            adp_mean_filter_img[x - 1][y-1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y+1],

            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x ][y + 1],

            adp_mean_filter_img[x +1][y-1],
            adp_mean_filter_img[x +1][y],
            adp_mean_filter_img[x +1][y+1],

        ]
        minimum = min(kernel)
    elif(kernel_size==5):
        kernel = [
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],

            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],

            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],

            adp_mean_filter_img[x + 1][y - 2],
            adp_mean_filter_img[x + 1][y - 1],
            adp_mean_filter_img[x + 1][y],
            adp_mean_filter_img[x + 1][y + 1],
            adp_mean_filter_img[x + 1][y + 2],

            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
        ]
        minimum = min(kernel)
    elif (kernel_size == 7):
        kernel = [
            adp_mean_filter_img[x - 3][y - 3],
            adp_mean_filter_img[x - 3][y - 2],
            adp_mean_filter_img[x - 3][y - 1],
            adp_mean_filter_img[x - 3][y],
            adp_mean_filter_img[x - 3][y + 1],
            adp_mean_filter_img[x - 3][y + 2],
            adp_mean_filter_img[x - 3][y - 3],

            adp_mean_filter_img[x - 2][y - 3],
            adp_mean_filter_img[x - 2][y - 2],
            adp_mean_filter_img[x - 2][y - 1],
            adp_mean_filter_img[x - 2][y],
            adp_mean_filter_img[x - 2][y + 1],
            adp_mean_filter_img[x - 2][y + 2],
            adp_mean_filter_img[x - 2][y - 3],

            adp_mean_filter_img[x - 1][y - 3],
            adp_mean_filter_img[x - 1][y - 2],
            adp_mean_filter_img[x - 1][y - 1],
            adp_mean_filter_img[x - 1][y],
            adp_mean_filter_img[x - 1][y + 1],
            adp_mean_filter_img[x - 1][y + 2],
            adp_mean_filter_img[x - 1][y - 3],

            adp_mean_filter_img[x][y - 3],
            adp_mean_filter_img[x][y - 2],
            adp_mean_filter_img[x][y - 1],
            adp_mean_filter_img[x][y],
            adp_mean_filter_img[x][y + 1],
            adp_mean_filter_img[x][y + 2],
            adp_mean_filter_img[x][y - 3],

            adp_mean_filter_img[x+1][y - 3],
            adp_mean_filter_img[x+1][y - 2],
            adp_mean_filter_img[x+1][y - 1],
            adp_mean_filter_img[x+1][y],
            adp_mean_filter_img[x+1][y + 1],
            adp_mean_filter_img[x+1][y + 2],
            adp_mean_filter_img[x+1][y - 3],

            adp_mean_filter_img[x + 2][y - 3],
            adp_mean_filter_img[x + 2][y - 2],
            adp_mean_filter_img[x + 2][y - 1],
            adp_mean_filter_img[x + 2][y],
            adp_mean_filter_img[x + 2][y + 1],
            adp_mean_filter_img[x + 2][y + 2],
            adp_mean_filter_img[x + 2][y - 3],

            adp_mean_filter_img[x + 3][y - 3],
            adp_mean_filter_img[x + 3][y - 2],
            adp_mean_filter_img[x + 3][y - 1],
            adp_mean_filter_img[x + 3][y],
            adp_mean_filter_img[x + 3][y + 1],
            adp_mean_filter_img[x + 3][y + 2],
            adp_mean_filter_img[x + 3][y - 3],

        ]
        minimum = min(kernel)

    return minimum



if __name__ == '__main__':
    question_1()
    question_2()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
