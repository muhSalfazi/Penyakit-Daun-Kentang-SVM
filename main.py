import numpy as np  # Add this import statement for numpy
import pandas as pd
from load_images import load
from feature_extraction import extract_color_features, extract_texture_features, extract_shape_features
from svm_classifier import svm_train_test
import cv2
import matplotlib.pyplot as plt

# Load images
early_images = load('./data/Test/Potato___Early_blight')
healthy_images = load('./data/Test/Potato___Healthy')
late_images = load('./data/Test/Potato___Late_blight')

all_images = np.append(early_images, late_images, axis=0)
all_images = np.append(all_images, healthy_images, axis=0)

# Extract features
color_features_red = extract_color_features(all_images, 'red')
color_features_green = extract_color_features(all_images, 'green')
color_features_blue = extract_color_features(all_images, 'blue')

color_features_red = color_features_red.T
color_features_green = color_features_green.T
color_features_blue = color_features_blue.T

color_features = np.hstack((color_features_red, color_features_green, color_features_blue))

texture_features = extract_texture_features(all_images)
shape_features = extract_shape_features(all_images)

# Create DataFrame
df_color = pd.DataFrame(color_features, columns=['red_mean', 'red_std', 'red_entropy', 'red_skew',
                                                 'green_mean', 'green_std', 'green_entropy', 'green_skew',
                                                 'blue_mean', 'blue_std', 'blue_entropy', 'blue_skew'])
df_texture = pd.DataFrame(texture_features.T, columns=['texture_mean', 'texture_std', 'contrast',
                                                       'correlation', 'energy', 'homogeneity'])
df_shape = pd.DataFrame(shape_features.T, columns=['area', 'perimeter', 'major_axis_length',
                                                   'minor_axis_length', 'eccentricity'])

df = pd.concat([df_color, df_texture, df_shape], axis=1)

# Create label
df['label'] = np.array([0] * 100 + [1] * 100 + [2] * 100)

# Train and test the SVM classifier
svm_train_test(df)

# Footnotes - Display image for testing
# Check if the index is within the range of all_images
index = 299  # Valid index range for 300 images is from 0 to 299
testes = all_images[index]  # Using a valid index

# Display the image
plt.imshow(testes)
plt.xticks([])  # Disable x ticks
plt.yticks([])  # Disable y ticks
plt.show()


# Check if the index is within the range of all_images
index = 299  # Valid index range for 300 images is from 0 to 299
testes = all_images[index]  # Using a valid index

# Convert to grayscale
testes = cv2.cvtColor(testes, cv2.COLOR_RGB2GRAY)
plt.imshow(testes, cmap='gray')
plt.xticks([])  # Disable x ticks
plt.yticks([])  # Disable y ticks
plt.show()

# After thresholding, you get two values: _, and the thresholded image.
_, testes = cv2.threshold(testes, 127, 255, cv2.THRESH_BINARY)

# Normalize to 0-1 by dividing by 255
testes = testes / 255

# Display the image
plt.imshow(testes, cmap='gray')
plt.xticks([])  # Disable x ticks
plt.yticks([])  # Disable y ticks
plt.show()
