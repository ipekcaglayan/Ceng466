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
        

    gaussian_samples = [gaussian(m, s, i)/total * pixel_num for i in range(256)]
    # total = 0

    new_total = 0
    for i in range(256):
        p = gaussian(m, s, i)/total
        new_total += p


    # print(new_total, "probabilities")
    # print(sum(gaussian_samples), 2048*2048)


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


def convoRGB(img_rgb, filter,row_num,col_num):
    
    processed_img = []
    
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
            val = [0,0,0]
            for x, y in offset:
                if r+x >=0 and r+x < row_num and c+y >=0 and c+y < col_num:
                   
                        val[2] += img_rgb[r+x][c+y][2]* filter[f_center_row+x][f_center_col+y]
                   
                        val[1] += img_rgb[r+x][c+y][1]* filter[f_center_row+x][f_center_col+y]
                    
                        val[0] += img_rgb[r+x][c+y][0]* filter[f_center_row+x][f_center_col+y]



            new_row.append(val)

        processed_img.append(new_row)
    processed_img = np.array(processed_img)

    return processed_img
def convoGray(img_grayscale,filter):
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
                    val += img_grayscale[r+x][c+y]* filter[f_center_row+x][f_center_col+y]
                  

            new_row.append(val)
          

        processed_img.append(new_row)

   
    processed_img = np.array(processed_img)
 

    return processed_img

def the1_convolution(input_img_path, filter):
    img_grayscale = cv2.imread(input_img_path, 0)
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
                    val += img_grayscale[r+x][c+y]* filter[f_center_row+x][f_center_col+y]
                  

            new_row.append(val)
          

        processed_img.append(new_row)

   
    processed_img = np.array(processed_img)

    return processed_img


def part2(input_img_path , output_path ) :
    img_grayscale = cv2.imread(input_img_path, 0)
    laplacian=[[1,1,1],[1,-8,1],[1,1,1]]
    processed_img=convoGray(img_grayscale,laplacian)

    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
   
    cv2.imwrite(os.path.join(output_path, "edges.png"), np.array(processed_img))
   
    return processed_img
def medianFilter(img_rgb,f_row,f_col):
    
    processed_img = []
    (row_num, col_num,renkler) = img_rgb.shape
    
    
    row_offset = list(range(-int(f_row/2), int(f_row/2) + 1))
    col_offset = list(range(-int(f_col/2), int(f_col/2) + 1))
    offset = list(itertools.product(row_offset, col_offset))

    f_center_row = int(f_row/2)
    f_center_col = int(f_col/2)

    for r in range(row_num):
        new_row = []
        for c in range(col_num):
            val = [[],[],[]]
            valresult=[0,0,0]
            for x, y in offset:
                
                if r+x >=0 and r+x < row_num and c+y >=0 and c+y < col_num:
                        val[0].append(img_rgb[r+x][c+y][0])
                        val[1].append(img_rgb[r+x][c+y][1])
                        val[2].append(img_rgb[r+x][c+y][2])
     

            for i in range(3):
                val[i].sort()
                valresult[i]=val[i][int(len(val[i])/2)-1]
            

            new_row.append(valresult)

        processed_img.append(new_row)
    

    return processed_img
def clip(channel):

    row=len(channel)
    column=len(channel[0])
    for i in range(row):
        for j in range(column):
            if channel[i][j]<0:
                channel[i][j]=0
            elif channel[i][j]>255:
                channel[i][j]=255
    return channel
    
def scale(channel):
   
    row=len(channel)
    column=len(channel[0])
    
    minx=np.amin(channel)
    maxx=np.amax(channel)  
    
    for i in range(row):
        for j in range(column):
            channel[i][j]=int(((channel[i][j]-minx)/(maxx-minx))*255)
          
    return channel
def enhance_3(path_to_3 , output_path ) :
    input_img = cv2.imread(path_to_3)
    
    (row_num, col_num,renkler) = input_img.shape
   
    input_img=medianFilter(input_img,11,11)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(os.path.join(output_path, "enhanced.png"), np.array(input_img))
    return input_img
def enhance_4(path_to_4 , output_path ) :
    input_img = cv2.imread(path_to_4)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    (row_num, col_num,renkler) = input_img.shape
    input_img=medianFilter(input_img,9,9)
   
    cv2.imwrite(os.path.join(output_path, "enhanced2.png"), np.array(input_img))

    laplacian=[[1,1,1],[1,-8,1],[1,1,1]]
    edges_img=convoRGB(input_img,laplacian,row_num,col_num)
  
    red = edges_img[:,:,2]
    green=edges_img[:,:,1]
    blue=edges_img[:,:,0]
    clip(red)
    clip(green)
    clip(blue)
    # scale(red)
    # scale(green)
    # scale(blue)
 
   
   
    
  
    for i in range(row_num):
        for j in range(col_num):
          
            input_img[i][j][0]=input_img[i][j][0]-edges_img[i][j][0]
            input_img[i][j][1]=input_img[i][j][1]-edges_img[i][j][1]
            input_img[i][j][2]=input_img[i][j][2]-edges_img[i][j][2]
    
   
    cv2.imwrite(os.path.join(output_path, "enhanced1.png"), np.array(input_img))
    

   
  
    return input_img

#enhance_4("./THE1-Images/4.png","./THE1_Outputs/")
# enhance_3("./THE1-Images/3.png","./enhance3/")
# part2("./THE1-Images/2.png","./part2/edges.png")
# part1("./THE1-Images/1.png","./Outputs/1/ex1",[45,200],[45,45])
#the1_convolution("./THE1-Images/1.png",[[-1,0,1],[-2,0,2],[-1,0,1]])
