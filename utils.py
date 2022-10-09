from imutils import paths
import os
import cv2
import numpy as np
import math
import pickle
from evaluation_funcs import *
from ml_metrics import *

# Constants
JPG_EXTENSION = ".jpg"
PNG_EXTENSION = '.png'
ACTUAL_RESULTS_FILE = "gt_corresps.pkl"
COLORSPACE_GRAYSCALE = 'gray'
COLORSPACE_RGB = 'rgb'
COLORSPACE_YCRCB = 'ycrcb'
COLORSPACE_HSV = 'hsv'
OUTPUT_FOLDER_QS1 = 'Team{}/week{}/QST1/method{}'
OUTPUT_FOLDER_QS2 = 'Team{}/week{}/QST2/method{}'


"""
    Gets the image file name from the path in the format (without extension)

    Parameters
    ----------
    img_number : int
            The number of the image from which we construct the file name

    Returns: Name of the file after filling the number with 0s and adding the extension
    -------
"""
def get_image_name(image_path):
    print()
    return os.path.basename(image_path[:len(image_path) - len(JPG_EXTENSION)])


"""
    Constructs an array with all images of the given extension from the given path

    Parameters
    ----------
    input_folder: numpy.ndarray
            Path of the folder from which the images should be extracted
    
    extension: string
            Extension of the images to be retrieved (including the '.', ex: '.jpg')

    Returns: An array with all images with the given extension under given path
    -------
"""
def get_images(input_folder, extension):
    all_images = list(paths.list_images(input_folder))
    return [img for img in all_images if (img[-4:] == extension)]


"""
    Convert given image to given colorspace 

    Parameters
    ----------
    image : numpy.ndarray
            Image object which should be converted

    colorspace: string
            Colorspace to which the image should be converted to
    
    grayscale_method : char
        Method for calculating grayscale ('a' = average of channels, 'w' = weighted calculation)

    Returns: Converted image
    -------
"""
def get_image_in_colorspace(image, colorspace, grayscale_method):
    result = image
    if colorspace == COLORSPACE_RGB:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif colorspace == COLORSPACE_YCRCB:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif colorspace == COLORSPACE_HSV:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif colorspace == COLORSPACE_GRAYSCALE:
        result = get_grayscale(grayscale_method, image)
    return result


"""
    Converts given image to grayscale using one of two methods depending on the given parameter
    (parameter name: grayscale_method)

    Parameters
    ----------
    grayscale_method : char
            Method for calculating grayscale ('a' = average of channels, 'w' = weighted calculation)

    image : numpy.ndarray
            Image object which should be converted

    Returns: Image converted to grayscale
    -------
"""
def get_grayscale(grayscale_method, image):
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    if grayscale_method == 'a':
        img_gray = (R + G + B) / 3 
    else:
        img_gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return np.float32(img_gray)


"""
    Calculates the histogram for a given image
    If image is not grayscale, it calculates the average between the histograms of each channel
    If a mask has been applied to the image, we ignore the background (black) from the histogram

    Parameters
    ----------
    image : numpy.ndarray
            Image object for which we calculate the histogram

    has_mask_applied: boolean
            Indicates if the image has had a mask applied
            

    Returns: Histogram of an image
    -------
"""
def get_histogram(image, has_mask_applied):
    if has_mask_applied == True:
        histogram = cv2.calcHist(image,[0],None,[255],[0,255])
    else:
        histogram = cv2.calcHist(image,[0],None,[255],[0,255])
            
    if len(histogram) == 3:
        return np.sum(histogram, axis = 0)/3
    return histogram


"""
    Calculates the similarity between two given histograms depending on the method chosen with similarity_method
    The intersection comparison uses a method from OpenCv, while the others do the calculations themselves

    Parameters
    ----------
    similarity_method : char
            Similarity method that should be used for the calculation

    query_histogram : array[int]
            Histogram of the query image

    bd_histogram: array[int]
            Histogram of the data base image

    Returns: Similarity between two histograms
    -------
"""
def get_similarity(similarity_method, query_histogram, bd_histogram):
    distance = 0
    if similarity_method == 'e':
        distance = euclidian_distance_comparison(query_histogram, bd_histogram)
    elif similarity_method == 'l':
        distance = l1_distance_comparison(query_histogram, bd_histogram)
    elif similarity_method == 'x':
        distance = x_square_distance_comparison(query_histogram, bd_histogram)
    return distance


"""
    Calculates euclidian distance between the two given histograms

    Parameters
    ----------
    query_histogram : array[int]
            Histogram of the query image

    bd_histogram: array[int]
            Histogram of the data base image

    Returns: Euclidian distance
    -------
"""
def euclidian_distance_comparison(query_histogram, bd_histogram):
    dist = 0
    for i in range(0,len(query_histogram)):
        dist += (query_histogram[i] - bd_histogram[i])**2
    return math.sqrt(dist)


"""
    Calculates L1 distance between the two given histograms

    Parameters
    ----------
    query_histogram : array[int]
            Histogram of the query image

    bd_histogram: array[int]
            Histogram of the data base image

    Returns: L1 distance
    -------
"""
def l1_distance_comparison(query_histogram, bd_histogram):
    dist = 0
    for i in range(0,len(query_histogram)):
        dist += abs(query_histogram[i] - bd_histogram[i])
    return dist


"""
    Calculates X^2 distance between the two given histograms

    Parameters
    ----------
    query_histogram : array[int]
            Histogram of the query image

    bd_histogram: array[int]
            Histogram of the data base image

    Returns: X^2 distance
    -------
"""
def x_square_distance_comparison(query_histogram, bd_histogram):
    dist = 0
    for i in range(0,len(query_histogram)):
        if (query_histogram[i] + bd_histogram[i] != 0):
            dist += ((query_histogram[i] - bd_histogram[i])**2)/(query_histogram[i] + bd_histogram[i])
    return dist


"""
    Minimim and maximum value for each channel from all images
    Gets the first pixel from an image as a tuple
    If image is grayscale, first channel will contain the value and the others are filled with 0

    Parameters
    ----------
    image_paths : String
            Path to images to extract first pixel from
            
    colorspace: string
            Colorspace to which the image should be converted to
    
    grayscale_method : char
        Method for calculating grayscale ('a' = average of channels, 'w' = weighted calculation)
    
    Returns: Minimum and maximum values of first pixel from all images
    -------
"""
def get_min_max_pixel_per_channel(image_paths, colorspace, grayscale_method):
    first_pixels = []

    for image_path in image_paths:
        bgr_image = cv2.imread(image_path)
        image = get_image_in_colorspace(bgr_image, colorspace, grayscale_method)

        # Append first pixel of every image to an array
        if colorspace == COLORSPACE_GRAYSCALE and tuple(image[0][0]) not in first_pixels:
            first_pixels.append((image[0][0], 0, 0))
        elif tuple(image[0][0]) not in first_pixels:
            first_pixels.append(tuple(image[0][0]))

    print(f'Colorspce: {colorspace}')
    r, g, b = zip(*first_pixels)
    print(f'First channel {min(r)}-{max(r)}')
    print(f'Second channel {min(g)}-{max(g)}')
    print(f'Third channel {min(b)}-{max(b)}')
    print('-----------')

    return [(min(r), max(r)), (min(g), max(g)), (min(b), max(b))]


"""
    Creates output folders from given path under the current path
    ex: first_folder/last_folder --> ./first_folder is created then ./first_folder/last_folder

    Parameters
    ----------
    path : String
            Path of the last folder to be created (starting from first folder to be created)

    Returns: Nothing, just creates the folders
    ----------
"""
def create_output_folders(path):
    current_path = ""
    for folder in path.split('/'):
        current_path += folder
        if not os.path.exists(current_path):
            os.mkdir(os.path.abspath(current_path))
        current_path += '/'


"""
    Compute similarities for a given image with all data base images

    Parameters
    ----------
    similarity_method: char
            Similarity method used to compare the histograms

    histograms : dictionary of string:arrays
            Dictionary containinig all histograms - path of the image as key and histogram as value

    query_image_path, : String
            Path of the query image for which we calculate the similarities
    i: int
            Number which indicates the name and position(number) of the image

    Returns: Similarities array, sorted, ascending
    ----------
"""
def calculate_similarities(similarity_method, histograms, bd_image_paths, query_image_path, i):
    # Calculate similarities for current image and all data base images
    current_similarity = []

    for j, bd_image_path in enumerate(bd_image_paths):
        similarity = get_similarity(similarity_method, histograms[query_image_path], histograms[bd_image_path])
        current_similarity.append((j, similarity))
    # Sort similarities array by the second value(similarity) in the tuples
    sorted_similarities = sorted(current_similarity, key=lambda tup: tup[1])

    return sorted_similarities


"""
    Unpickle pickled file containing the actual results

    Parameters
    ----------
    input_folder: String
            Path to the folder containg the pickled file

    Returns: Unpickled file
    ----------
"""
def get_actual_results(input_folder):
    file = open(input_folder + '/' + ACTUAL_RESULTS_FILE, 'rb')
    return pickle.load(file)


"""
    Calculate map@1 and map@5

    Parameters
    ----------
    actual_results: array[array[int]]
            Array containing the ground truth

    predicted_results: array[array[int]]
            Array containing the results of the system

    Returns: Nothing, only prints results
    ----------
"""
def evaluate_mapk_1_5(actual_results, predicted_results):
    mapk1 = mapk(actual_results,predicted_results,1)
    mapk5 = mapk(actual_results,predicted_results,5)
    print("{} ({}/{})".format(mapk1, int(len(predicted_results) * mapk1), len(predicted_results)))
    print("{} ({}/{})".format(mapk5, int(len(predicted_results) * mapk5), len(predicted_results)))
    

"""
    Construct mask for given image

    Parameters
    ----------
    image: numpy.ndarray
            Image object from which we construct the mask

    Returns: numpy.ndarray
            Mask image object
    ----------
"""
def get_mask(image, min_max):
    dimensions = image.shape
    mask = np.zeros((dimensions[0], dimensions[1]), dtype="uint8")
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            R, G, B = image[i,j]
            if min_max[0][0] <= R <= min_max[0][1] and min_max[1][0] <= G <= min_max[1][1] and min_max[2][0] <= B <= min_max[2][1]:
                mask[i,j] = 0
            else:
                mask[i,j] = 255
    return mask


"""
    Constructs mask file path at which they should be saved

    Parameters
    ----------
    image_path: string
            Path to image from which the mask has been constructed (needed for file name)
    
    folder_method: int
            Number needed to construct output folder

    Returns: string
            Path where the mask should be saved (including name and file extension)
    ----------
"""
def get_mask_file_path(image_path, folder_method):
    return OUTPUT_FOLDER_QS2.format(6, 1, folder_method) + '/' + get_image_name(image_path) + PNG_EXTENSION


"""
    Evaluation of masks - comparison betweek predicted masks and ground truth

    Parameters
    ----------
    predicted: numpy.ndarray
            Predicted mask image object

    actual: numpy.ndarray
            Ground truth mask image object

    Returns: Nothing, only prints results
    ----------
"""
def evaluate_masks(predicted, actual):
    pixelTP, pixelFP, pixelFN, pixelTN = performance_accumulation_pixel(predicted, actual)
    pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
    print('Pixel Precision {}'.format(pixel_precision))
    print('Pixel Accuracy {}'.format(pixel_precision))
    print('Pixel Specificity {}'.format(pixel_specificity))
    print('Pixel Sensitivity {}'.format(pixel_sensitivity))


"""
    Print command line help

    Parameters
    ----------
    None

    Returns: Nothing, only prints help
    ----------
"""
def print_command_line_help():
    print('-c or --image_colorspace    -- Sets the value for all query image colorspaces.')
    print('                            -- Values: gray, rgb, ycrcb, hsv')
    print('                            -- Default: ycrcb')
    print('-g or --grayscale_method    -- Sets the value of the grayscale method used when converting image to grayscale')
    print('                            -- Values: a = average, w = weighted')
    print('                            -- Default: w')
    print('-s or --similarity_method   -- Sets the value of the similarity method used when comparing 2 histograms')
    print('                            -- Values: e = euclidean distance, l = L1 distance, x = X square distance')
    print('                            -- Default: x')
    print('-1 or --input_folder_q1     -- Sets the value of the input folder for the first image query set')
    print('                            -- Values: path to the input folder - should be relative to current folder, add / at the end')
    print('                            -- Default: ../Data/qsd1_w1/')
    print('-2 or --input_folder_q2     -- Sets the value of the input folder for the second image query set')
    print('                            -- Values: path to the input folder - should be relative to current folder, add / at the end')
    print('                            -- Default: ../Data/qsd1_w2/')
    print('-d or --input_folder_bd     -- Sets the value of the input folder for the image data base set')
    print('                            -- Values: path to the input folder - should be relative to current folder, add / at the end')
    print('                            -- Default: ../Data/qsd1_w2/')
    print('-o or --output              -- Sets the value of the number of results to be returned')
    print('                            -- Values: integers (smaller than image sets)')
    print('                            -- Default: 10')
    print('-m or --mask_colorspace     -- Sets the colorspace for the query image before calculating the mask')
    print('                            -- Values: gray, rgb, ycrcb, hsv')
    print('                            -- Default: hsv')
    print('-q or --run_qs2             -- If the system should run on the second query set (takes a few minutes)')
    print('                            -- Values: boolean')
    print('                            -- Default: false')
    print('-f or --folder_method       -- Number to be appended to the method folder (in the output)')
    print('                            -- Values: integer')
    print('                            -- Default: 1')