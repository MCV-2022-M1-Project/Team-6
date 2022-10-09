import numpy as np
import math
import pickle
from utils import *
from evaluation_funcs import *
from ml_metrics import *
import getopt, sys


# Setup for parameters
argumentList = sys.argv[1:]
options = "c:g:s:1:2:d:o:m:q:p:f:h"
long_options = ['image_colorspace', 'grayscale_method', 'similarity_method', 'input_folder_q1', 'input_folder_q2', 'input_folder_bd', 'output', 'mask_colorspace', 'only_qs2', 'folder_method', 'help']

image_colorspace = 'ycrcb'
grayscale_method = 'w'
similarity_method = 'x'
input_folder_q1 = '../Data/qsd1_w1/'
input_folder_q2 = '../Data/qsd2_w1/'
input_folder_bd = '../Data/bbdd/'
output = 10
mask_colorspace = 'hsv'
only_qs2 = False
folder_method = 1

# Parsing over arguments, if not found they have the default values from above
try:
    arguments, values = getopt.getopt(argumentList, options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument == "-c" or currentArgument == "--image_colorspace":
            image_colorspace = currentValue
        elif currentArgument == "-g" or currentArgument == "--grayscale_method":
            grayscale_method = currentValue
        elif currentArgument == "-s" or currentArgument == "--similarity_method":
            similarity_method = currentValue
        elif currentArgument == "-1" or currentArgument == "--input_folder_q1":
            input_folder_q1 = currentValue
        elif currentArgument == "-2" or currentArgument == "--input_folder_q2":
            input_folder_q2 = currentValue
        elif currentArgument == "-d" or currentArgument == "--input_folder_bd":
            input_folder_bd = currentValue
        elif currentArgument == "-o" or currentArgument == "--output":
            output = currentValue
        elif currentArgument == "-m" or currentArgument == "--mask_colorspace":
            mask_colorspace = currentValue
        elif currentArgument == "-q" or currentArgument == "--only_qs2":
            only_qs2 = currentValue
        elif currentArgument == "-f" or currentArgument == "--folder_method":
            folder_method = currentValue
        elif currentArgument == "-h" or currentArgument == "--help":
            print_command_line_help()
except getopt.error as err:
    # output error, and return with an error code
    print(str(err))


# Global variables
query_image_paths = []
bd_image_paths = []
sorted_similarities = []
histograms = {}
predicted_results = []

# Get data base image paths for reading the images
bd_image_paths = [img for img in list(paths.list_images(input_folder_bd)) if (img[-4:] == JPG_EXTENSION)]


### QS 1 ###
output_path_QS1 = OUTPUT_FOLDER_QS1.format(6, 1, folder_method)
create_output_folders(output_path_QS1)
# Get query image paths for reading the images
query_image_paths = get_images(input_folder_q1, JPG_EXTENSION)

# Calculate histogram on all data base images
for bd_image_path in bd_image_paths:
    # Read and convert image to given colorspace
    bgr_image = cv2.imread(bd_image_path)
    image = get_image_in_colorspace(bgr_image, image_colorspace, grayscale_method)

    # Calculate histogram for current image (in order to compare it to the other histograms)
    histogram = get_histogram(image, False)
    histograms[bd_image_path] = cv2.normalize(histogram, histogram)

if only_qs2 == 'False':
    for i, query_image_path in enumerate(query_image_paths):
        # Read and convert image to given colorspace
        bgr_image = cv2.imread(query_image_path)
        image = get_image_in_colorspace(bgr_image, image_colorspace, grayscale_method)

        # Calculate histogram for current image (in order to compare it to the other histograms)
        histogram = get_histogram(image, False)
        histograms[query_image_path] = histogram

        # Calculate similarities
        similarities_q1 = calculate_similarities(similarity_method, histograms, bd_image_paths, query_image_path, i)

        # Keep only given number of results for current image
        predicted_results.append([tup[0] for tup in similarities_q1[:output]])

    # Pickle predicted results
    pickle.dump(predicted_results, open(output_path_QS1 + '/result.pkl', 'wb'))

    # Evaluate map@k
    print('Map@K evaluation for QS1, with parameters: image colorspace: {}, grayscale method: {}, similarity method: {}'.format(image_colorspace, grayscale_method, similarity_method))
    actual_results = get_actual_results(input_folder_q1)
    evaluate_mapk_1_5(actual_results, predicted_results)


### QS 2 ###
if only_qs2 == 'True':
    output_path_QS2 = OUTPUT_FOLDER_QS2.format(6, 1, folder_method)
    create_output_folders(output_path_QS2)

    # Get query image paths for reading the images
    query_image_paths = get_images(input_folder_q2, JPG_EXTENSION)
    # Reset global variables, keep histograms in order to avoid recalculating them for data base images
    similarities = []
    sorted_similarities = []
    predicted_results = []
    predicted_masks = []
    actual_masks = []

    min_max = get_min_max_pixel_per_channel(query_image_paths, mask_colorspace, grayscale_method)
    for i, query_image_path in enumerate(query_image_paths):
        # Read image and convert colorspace
        bgr_image = cv2.imread(query_image_path)
        image = get_image_in_colorspace(bgr_image, mask_colorspace, grayscale_method)

        # Get mask and refine it by closing and opening it
        print('Generating mask for image {}'.format(query_image_path))
        mask = get_mask(image, min_max)

        closing_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
        opening_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))

        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_structure)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, opening_structure)

        # Save mask image
        masked_path = get_mask_file_path(query_image_path, folder_method)
        cv2.imwrite(masked_path, opening)

        # Apply mask and convert images back to RGB
        predicted_masks.append(opening)
        masked = cv2.bitwise_or(image, image, mask=opening)

        # Calculate histogram for masked image
        histogram = get_histogram(masked, True)
        histograms[masked_path] = cv2.normalize(histogram, histogram)

        # Calculate similarities
        similarities_q2 = calculate_similarities(similarity_method, histograms, bd_image_paths, masked_path, i)

        # Keep only given number of results for current image
        predicted_results.append([tup[0] for tup in similarities_q2[:output]])

    # Pickle predicted results
    pickle.dump(predicted_results, open(output_path_QS2 + '/result.pkl', 'wb'))

    # Evaluate map@k
    print('Map@K evaluation for QS2, with parameters: mask colorspace: {}, grayscale method: {}, similarity method: {}'.format(
            mask_colorspace, grayscale_method, similarity_method))
    actual_results = get_actual_results(input_folder_q1)
    evaluate_mapk_1_5(actual_results, predicted_results)

    actual_masks_paths = get_images(input_folder_q2, PNG_EXTENSION)
    for i, actual_masks_path in enumerate(actual_masks_paths):
        print('Evaluation for: {}'.format(actual_masks_path))
        actual_mask = cv2.imread(actual_masks_path)
        actual_mask_binary = get_image_in_colorspace(actual_mask, COLORSPACE_GRAYSCALE, grayscale_method)
        evaluate_masks(predicted_masks[i], actual_mask_binary)