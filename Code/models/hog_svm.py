from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from evaluator import ModelEvaluator
import numpy as np
import joblib
import os

# Path to save the model
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Models', 'hog_svm_model.joblib')
MODEL_SAVE_PATH = os.path.abspath(MODEL_SAVE_PATH)

evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])

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
            cells_per_block=cells_per_block, #group for local contrast
            block_norm='L2-Hys', #method for normalization
            feature_vector=True #make sure we output a flat vector 
        )
        #store the descriptor
        hog_features.append(features) 
    return np.array(hog_features) #2d array, each row is a HOG feature

# training
def train_hog_svm(X_train, y_train):
    print("Extracting HOG features for training...")
    hog_train = extract_hog_features(X_train)

    print("Training SVM classifier...")
    #rbf kernel handles non linear decision boundaries 
    clf = SVC(kernel='rbf', C=10, gamma=0.01)
    clf.fit(hog_train, y_train)

    # Save model to the defined path
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    return clf #the trained model

# evaluate the svm 
def evaluate_hog_svm(clf, X, y, split_name="Validation"):
    #using the validation set for now so we can see how it performs on unseen data
    print(f"\n Extracting HOG features for {split_name} set...")
    hog_features = extract_hog_features(X)

    print(f"Predicting {split_name} set...")
    y_pred = clf.predict(hog_features)

    # Use evaluator to get and print detailed metrics
    evaluator.evaluate(y, y_pred, model_name=f"HOG + SVM ({split_name})")
    evaluator.plot_confusion_matrix(y, y_pred, title=f"HOG + SVM - {split_name} Confusion Matrix")

    return y_pred #the labels our model predicted

# visualise some results which were predicted
def show_predictions(X, y_true, y_pred, n_samples=4):
    evaluator.visualize_predictions(X, y_true, y_pred, n_samples=n_samples)
