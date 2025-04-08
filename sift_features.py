import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def extract_sift_descriptors(images, max_features_per_image=100):
   
    sift = cv2.SIFT_create() #feature extracture, takes keypoints and descriptors 
    descriptors_list = []
    valid_indices = []

    for idx, img in enumerate(tqdm(images, desc="Extracting SIFT descriptors")):
        #converting the previously normalised float back to 8-bit with unit8 for OpenCV
        gray = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY) #convert to grayscale 
        _, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            #limit descriptors to contorl computational time
            descriptors_list.append(descriptors[:max_features_per_image]) 
            valid_indices.append(idx)  # record that image worked

    return descriptors_list, valid_indices #return descriptors and indices that worked 

def build_visual_vocabulary(descriptor_list, vocab_size=100):
    all_descriptors = np.vstack(descriptor_list) #all the descriptors from the images into a single array

    #using kMeans group descriptors inot clusters
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size, #numb of visual words
        batch_size=1000, #batch size, 1000 is a balance between fast and scalable
        random_state=42#reproducibility seed 
        ) 
    kmeans.fit(all_descriptors)#fit the model to all the descriptors 
    return kmeans #returning the trained vocab model

def compute_bovw_histograms(descriptor_list, kmeans_model):
    vocab_size = kmeans_model.n_clusters #number of bins in the bag of visual words histogram
    histograms = []

    for descriptors in descriptor_list:
        histogram = np.zeros(vocab_size) #initialise histogram with zero
        if descriptors is not None:
            words = kmeans_model.predict(descriptors) #each descriptor is assigned to a visual word
            for w in words:
                histogram[w] += 1 #increment the count in corresponding histogram bin
        histograms.append(histogram) # add the histogram to the list

    return np.array(histograms) #list into array