import os
import cv2
from tqdm import tqdm
import numpy as np

def load(path):
    files = os.listdir(path)
    images = []
    for file in tqdm(files):
        bgr_image = cv2.imread(path + '/'+file)
        # Convert to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        images.append(rgb_image)
    return np.array(images)
