import numpy as np
from scipy.stats import skew, entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from tqdm import tqdm
import cv2

def extract_color_features(images, color):
    means = []
    stds = []
    entropies = []
    skews = []
    channel_dict = {'red': 0, 'green': 1, 'blue': 2}
    channel = channel_dict[color]

    for image in tqdm(images):
        image = image[:, :, channel]
        means.append(np.mean(image.flatten()))
        stds.append(np.std(image.flatten()))
        entropies.append(entropy(image.flatten()))
        skews.append(skew(image.flatten()))

    return np.array([means, stds, entropies, skews])

def extract_texture_features(images):
    means = []
    stds = []
    contrasts = []
    correlations = []
    energies = []
    homogeneities = []

    for image in tqdm(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        glcm = graycomatrix(image, distances=[1], angles=[0])

        mean = np.mean(glcm.flatten())
        std = np.std(glcm.flatten())
        contrast = graycoprops(glcm, 'contrast').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()

        means.append(mean)
        stds.append(std)
        contrasts.append(contrast[0])
        correlations.append(correlation[0])
        energies.append(energy[0])
        homogeneities.append(homogeneity[0])

    return np.array([means, stds, contrasts, correlations, energies, homogeneities])

def extract_shape_features(images):
    areas = []
    perimeters = []
    major_axis_lengths = []
    minor_axis_lengths = []
    eccentricities = []

    for image in tqdm(images):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = image / 255

        label_image = label(image)
        regions = regionprops(label_image)

        largest_area_index = np.argmax([region.area for region in regions])

        area = regions[largest_area_index].area
        perimeter = regions[largest_area_index].perimeter
        major_axis_length = regions[largest_area_index].major_axis_length
        minor_axis_length = regions[largest_area_index].minor_axis_length
        eccentricity = regions[largest_area_index].eccentricity

        areas.append(area)
        perimeters.append(perimeter)
        major_axis_lengths.append(major_axis_length)
        minor_axis_lengths.append(minor_axis_length)
        eccentricities.append(eccentricity)

    return np.array([areas, perimeters, major_axis_lengths, minor_axis_lengths, eccentricities])
