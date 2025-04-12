import os
from Code.data_loader import prepare_datasets

# Load dataset once
BASE_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CW_Dataset'))

train_image_path = os.path.join(BASE_DATASET_PATH, 'train', 'images')
train_label_path = os.path.join(BASE_DATASET_PATH, 'train', 'labels')
test_image_path = os.path.join(BASE_DATASET_PATH, 'test', 'images')
test_label_path = os.path.join(BASE_DATASET_PATH, 'test', 'labels')

X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(
    train_image_path, train_label_path,
    test_image_path, test_label_path,
    image_size=(128, 128)
)

# --- Model Functions ---

def run_hog_svm():
    from models.hog_svm import train_hog_svm, evaluate_hog_svm, show_predictions
    hog_model = train_hog_svm(X_train, y_train)
    predictions = evaluate_hog_svm(hog_model, X_val, y_val)
    show_predictions(X_val, y_val, predictions)

def run_sift_mlp():
    from models.sift_mlp import train_sift_mlp, evaluate_sift_mlp, show_predictions
    clf, kmeans = train_sift_mlp(X_train, y_train)
    predictions = evaluate_sift_mlp(clf, kmeans, X_val, y_val)
    show_predictions(X_val, y_val, predictions)



def run_cnn_gridsearch():
    from models.cnn import CNNTrainer
    trainer = CNNTrainer()
    param_grid = [
        ((32, 64, 128), 128, 0.3, 0.001, 32, 15),
        ((32, 64, 128, 256), 128, 0.3, 0.0001, 32, 15),
        ((64, 128, 256), 256, 0.2, 0.001, 64, 20),
    ]
    best_model, _ = trainer.run_grid_search(X_train, y_train, X_val, y_val, param_grid)
    trainer.evaluate_model(best_model, X_val, y_val)
    trainer.visualize_predictions(best_model, X_val, y_val)

def run_cnn_best():
    from models.cnn import CNNTrainer
    trainer = CNNTrainer()
    model = trainer.train_best_model(X_train, y_train, X_val, y_val)
    trainer.evaluate_model(model, X_val, y_val)
    trainer.visualize_predictions(model, X_val, y_val)

import os
import time
import joblib
import numpy as np
from keras.api.models import load_model
from evaluator import ModelEvaluator

def run_all_saved_models_on_test():
    evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])

    MAIN_DIR = os.getcwd()
    PROJECT_DIR = os.path.abspath(os.path.join(MAIN_DIR))
    MODEL_DIR = os.path.join(PROJECT_DIR, 'Models')
    print(MODEL_DIR)

    print("\nTesting HOG + SVM Model")
    from models.hog_svm import extract_hog_features
    hog_model_path = os.path.join(MODEL_DIR, 'hog_svm_model.joblib')
    if os.path.exists(hog_model_path):
        hog_model = joblib.load(hog_model_path)
        X_test_hog = extract_hog_features(X_test)

        start = time.time()
        y_pred_hog = hog_model.predict(X_test_hog)
        duration = time.time() - start

        size = os.path.getsize(hog_model_path) / 1024  # KB
        print(f"HOG+SVM inference time: {duration:.4f}s")
        print(f"HOG+SVM model size: {size:.2f} KB")

        evaluator.evaluate(y_test, y_pred_hog, model_name="HOG + SVM (Test)")
        evaluator.plot_confusion_matrix(y_test, y_pred_hog)
    else:
        print("HOG SVM model not found.")

    print("\nTesting SIFT + MLP Model")
    from models.sift_mlp import extract_sift_descriptors, compute_bovw_histograms
    sift_model_path = os.path.join(MODEL_DIR, 'sift_mlp_model.joblib')
    kmeans_path = os.path.join(MODEL_DIR, 'sift_kmeans.joblib')
    if os.path.exists(sift_model_path) and os.path.exists(kmeans_path):
        sift_model = joblib.load(sift_model_path)
        kmeans_model = joblib.load(kmeans_path)

        descriptors_test, test_indices = extract_sift_descriptors(X_test)
        X_test_bovw = compute_bovw_histograms(descriptors_test, kmeans_model)
        y_test_filtered = y_test[test_indices]

        start = time.time()
        y_pred_sift = sift_model.predict(X_test_bovw)
        duration = time.time() - start

        size = os.path.getsize(sift_model_path) / 1024
        print(f"SIFT+MLP inference time: {duration:.4f}s")
        print(f"SIFT+MLP model size: {size:.2f} KB")

        evaluator.evaluate(y_test_filtered, y_pred_sift, model_name="SIFT + MLP (Test)")
        evaluator.plot_confusion_matrix(y_test_filtered, y_pred_sift)
    else:
        print("SIFT model or KMeans model not found.")

    print("\nTesting best CNN Model")
    cnn_model_path = os.path.join(MODEL_DIR, 'best_CNN_model.keras')
    if os.path.exists(cnn_model_path):
        cnn_model = load_model(cnn_model_path)

        start = time.time()
        y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
        duration = time.time() - start

        size = os.path.getsize(cnn_model_path) / 1024 / 1024  # MB
        print(f"CNN inference time: {duration:.4f}s")
        print(f"CNN model size: {size:.2f} MB")

        evaluator.evaluate(y_test, y_pred_cnn, model_name="CNN (Test)")
        evaluator.plot_confusion_matrix(y_test, y_pred_cnn)
    else:
        print("CNN model not found.")

#run_hog_svm()
#run_sift_mlp()

#run_cnn_gridsearch()
#run_cnn_best()
run_all_saved_models_on_test()