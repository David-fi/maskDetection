import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from evaluator import ModelEvaluator
import os
import joblib

# Set path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR,  '..','..', 'Models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'sift_mlp_model.joblib')
KMEANS_PATH = os.path.join(MODEL_DIR, 'sift_kmeans.joblib')
evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])
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

def train_sift_mlp(X_train, y_train, vocab_size=100):
    print("Extracting SIFT features...")
    #retrieve the usable descriptors and indicies
    descriptors_list, valid_indices = extract_sift_descriptors(X_train)

    #filter the labels corresponding to valid images
    y_train_filtered = y_train[valid_indices]

    print("Building visual vocabulary...")
    kmeans_model = build_visual_vocabulary(descriptors_list, vocab_size=vocab_size)

    #features converted to fixed length histograms
    print("Building BoVW histograms...")
    bovw_train = compute_bovw_histograms(descriptors_list, kmeans_model)

    print("Training MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128,),#one hidden layer with 128 neuronr
        max_iter=500, # 500 training iterations
        random_state=42 #for reproducibility
        )
    clf.fit(bovw_train, y_train_filtered) #training the MLP on BoVW features and label
    # Save models
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(kmeans_model, KMEANS_PATH)

    return clf, kmeans_model  # return both for reuse on evaluation

def evaluate_sift_mlp(clf, kmeans_model, X_val, y_val):
    descriptors_val, val_indices = extract_sift_descriptors(X_val) #extract descriptors from the validation set
    y_val_filtered = y_val[val_indices] #filter out the labels for usable validation images

    bovw_val = compute_bovw_histograms(descriptors_val, kmeans_model) #convert the data in the validation files to BoVW vectors
    y_pred = clf.predict(bovw_val) #predict

    evaluator.evaluate(y_val_filtered, y_pred, model_name="SIFT + BoVW + MLP") 
    evaluator.plot_confusion_matrix(y_val_filtered, y_pred, title="SIFT + BoVW + MLP - Validation Confusion Matrix")

    return y_pred

def show_predictions(X_val, y_val, y_pred, n_samples=4):
    #visualise some examples
    evaluator.visualize_predictions(X_val, y_val, y_pred, n_samples=n_samples)