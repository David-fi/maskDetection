from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# extracting HOG Features 
def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    hog_features = []
    for img in images:
        # Convert to grayscale for HOG
        gray_img = np.mean(img, axis=2) #convert to grayscale
        #calc HOG descriptor for the image 
        features = hog(
            gray_img,
            orientations=orientations, #number of orientation bins 
            pixels_per_cell=pixels_per_cell, #size of the cell
            cells_per_block=cells_per_block, #griup for local contrast
            block_norm='L2-Hys', #method for normalization
            feature_vector=True #make sure i output a flat vector 
        )
        #store the descriptor
        hog_features.append(features) 
    return np.array(hog_features) #2d array, each row is a HOG feature
