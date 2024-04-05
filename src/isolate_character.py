import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils.logger import Logger


def draw_isolated_character(src_img, contours):
    blank = np.zeros(src_img.shape, dtype=np.uint8) 
    cv.drawContours(blank, [contours], -1, (255, 255, 255), -1)
    return blank


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def filter_contours(contours, src_img_w, src_img_h):
    filtered_contours = []
    DISTANCE_THRESHOLD = 50
    PERIMETER_THRESHOLD = 20
    image_center = (src_img_w // 2, src_img_h // 2)
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        if perimeter < PERIMETER_THRESHOLD:
            continue
        is_close_to_center = True
        
        for point in contour:
            distance = calculate_distance(point[0], image_center)
            
            if distance > DISTANCE_THRESHOLD:
                is_close_to_center = False
                break
        
        if is_close_to_center:
            filtered_contours.append(contour)
            
    return filtered_contours 

import numpy as np


def find_color_thing(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Use KMeans clustering to quantize the image
    # Adjust n_clusters for different quantization levels
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    
    # Find the most frequent cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    most_frequent_cluster_index = unique[np.argmax(counts)]
    
    # Find the color of the most frequent cluster
    most_frequent_color = kmeans.cluster_centers_[most_frequent_cluster_index]
    
    # Convert the color to integer values
    most_frequent_color = most_frequent_color.round().astype(int)
    
    print("HELLOOW#*(&$(*&$%(*&%(*&(*&*(&: ", most_frequent_color)
    return most_frequent_color


def crop_the_thing(image, size):
    height, width = image.shape[:2]

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the coordinates of the top left corner of the 20x20 patch
    top_left_x = center_x - (size // 2) 
    top_left_y = center_y - (size // 2) 

    # Calculate the coordinates of the bottom right corner of the 20x20 patch
    bottom_right_x = center_x + (size // 2) 
    bottom_right_y = center_y + (size // 2) 

    # Extract the 20x20 patch
    patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return patch 


def isolate_character_exp(src_image):
    # TODO: with the most frequent color voncert to black and convert grey values to black as well
    # Convert the image from BGR to HSV
    blur = cv.GaussianBlur(src_image, (5, 5), 0)
    #most_freq_color = find_color_thing(blur)

    cropped = crop_the_thing(blur, 40) 
    #hsv_image = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    hsv_image = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)

    # Quantize to 16 colors for simplification
    #hsv_image_quantized = hsv_image.copy()
    hsv_image_quantized = hsv_image.copy()
    hsv_image_quantized[:, :, 0] = (hsv_image_quantized[:, :, 0] // 16) * 16  # Hue
    hsv_image_quantized[:, :, 1] = (hsv_image_quantized[:, :, 1] // 64) * 64  # Saturation
    hsv_image_quantized[:, :, 2] = (hsv_image_quantized[:, :, 2] // 64) * 64  # Value
    
    # Convert back to BGR color space
    simplified_image = cv.cvtColor(hsv_image_quantized, cv.COLOR_HSV2BGR)
    most_freq_color = find_color_thing(simplified_image)
    
    # Find unique colors
    unique_colors = np.unique(simplified_image.reshape(-1, simplified_image.shape[2]), axis=0)
    print("ISDY(F*^*&^T%$#$%^&*(*&&&&&&&&&&&&&)): ", unique_colors)
    
    # Convert unique colors back to HSV to filter by saturation
    unique_colors_hsv = cv.cvtColor(unique_colors.reshape(-1, 1, 3), cv.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Filter out colors with low saturation (e.g., gray colors)
    # Adjust the saturation threshold as needed
    saturation_threshold = 25  # This can be adjusted
    colored_unique_colors = unique_colors[np.where(unique_colors_hsv[:, 1] > saturation_threshold)]
    
    # Create masks for each non-gray unique color and visualize
    height, width, _ = blur.shape
    mask_images = np.zeros((len(colored_unique_colors), height, width), dtype=np.uint8)

    for i, color in enumerate(colored_unique_colors):
        # Create a mask for each unique color
        #mask = np.all(simplified_image == color, axis=-1)
        mask = np.all(blur == color, axis=-1)
        
        # Visualize the mask
        mask_images[i][mask] = 255  # White where the color matches

    thing = [img for img in mask_images]
    thing.insert(0, simplified_image)

    return 0, thing 



def bak2_isolate_character_exp(src_image):
    # TODO: remove gray

    src_image = cv.GaussianBlur(src_image, (5, 5), 0)

    # Convert the image from BGR to HSV
    hsv_image = cv.cvtColor(src_image, cv.COLOR_BGR2HSV)
    
    # Quantize to 16 colors for simplification
    # Note: Adjust these bins to change color precision
    hsv_image_quantized = hsv_image.copy()
    hsv_image_quantized[:, :, 0] = (hsv_image_quantized[:, :, 0] // 16) * 16
    hsv_image_quantized[:, :, 1] = (hsv_image_quantized[:, :, 1] // 64) * 64
    hsv_image_quantized[:, :, 2] = (hsv_image_quantized[:, :, 2] // 64) * 64
    
    # Convert back to BGR color space
    simplified_image = cv.cvtColor(hsv_image_quantized, cv.COLOR_HSV2BGR)
    
    # Find unique colors
    unique_colors = np.unique(simplified_image.reshape(-1, simplified_image.shape[2]), axis=0)
    
    # Create masks for each unique color and visualize
    height, width, _ = src_image.shape
    mask_images = np.zeros((len(unique_colors), height, width), dtype=np.uint8)

    for i, color in enumerate(unique_colors):
        # Create a mask for each unique color
        mask = np.all(simplified_image == color, axis=-1)
        
        # Visualize the mask
        mask_images[i][mask] = 255  # White where the color matches

    return 0, mask_images
        

def bak_isolate_character_exp(src_image):
    #cropped_image = src_image[20:100, 20:100]

    gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    #adjusted = cv.convertScaleAbs(gray, alpha=2, beta=0)
    #adjusted = np.clip(adjusted, 0, 255).astype('uint8')

    #adjusted = cv.convertScaleAbs(src_image, alpha=2, beta=0) # up contrast 
    #adjusted = np.clip(adjusted, 0, 255).astype('uint8')
    #gray = cv.cvtColor(adjusted, cv.COLOR_BGR2GRAY)
    
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #clahe_image = clahe.apply(gray)
    #blur = cv.GaussianBlur(clahe_image, (5, 5), 0)

    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # TODO: do the contours to find the shape then run the thresholding on just that shape 
    # "Otsu's method performs well when the histogram has a bimodal distribution with a deep and 
    # sharp valley between the two peaks"

    #_, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    #thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 3)
    #_, thresh = cv.threshold(blur, 0, 255, cv.THRESH_OTSU)

    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 3) 

    #cv.imwrite("_srcimagething.jpg", src_image)
    #cv.imwrite("_thresholdthing.jpg", thresh) 
    #cv.imwrite("_graything.jpg", gray) 
    #cv.imwrite("_blurthing.jpg", blur) 
    #cv.imwrite("_contrastthing.jpg", adjusted) 

    #ctrs, hier = cv.findContours(thresh.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #ctrs, hier = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #ctrs, hier = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    #ctrs, hier = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    ctrs, hier = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    src_w, src_h = src_image.shape[:2]
    filtered_contours = filter_contours(ctrs, src_w, src_h)
    if len(filtered_contours) == 0:
        return -1, [gray, blur, thresh, "filtered_contours length is 0"]

    contours = sorted(filtered_contours, key=cv.contourArea, reverse=False)
    small_ctr = contours[0]
    
    ctr_pic = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.drawContours(ctr_pic, ctrs, -1, (255, 0, 255), 1)
    cv.drawContours(ctr_pic, [small_ctr], -1, (0, 255, 0), 1)
    #cv.imwrite("_contoursthing.jpg", gray)

    #blank_image = np.zeros_like(src_image)
    #for ctr in ctrs:
        #cv.drawContours(blank_image, [ctr], -1, (255, 255, 255), thickness=cv.FILLED)

    isolated_character_img = draw_isolated_character(src_image, small_ctr) 

    return 0, [gray, blur, ctr_pic, thresh] 


def isolate_character(src_image):
    image = cv.cvtColor(src_image, cv.COLOR_BGR2RGB)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (1, 1, image.shape[1]-1, image.shape[0]-1)

    cv.grabCut(image, mask, rect, bgd_model, fgd_model, 7, cv.GC_INIT_WITH_RECT)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask[:, :, np.newaxis]
    pixels = result.reshape(-1, 3)

    num_clusters = 3 # shape, letter, and bg colors
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)
    labels = kmeans.predict(pixels)
    labels = labels.reshape(image.shape[:2])

    clustered_images = [np.zeros_like(result) for _ in range(num_clusters)]
    for i in range(num_clusters):
        clustered_images[i][labels == i] = result[labels == i]

    #fig, axes = plt.subplots(1, 3)
    #axes[0].imshow(image)
    #axes[0].set_title("Original")
    #axes[1].imshow(result)
    #axes[1].set_title("Background Removed")
    #axes[2].imshow(clustered_images[2])
    #axes[2].set_title("Final")
    
    #plt.savefig('cluster_results.jpg')
    return clustered_images[2]
