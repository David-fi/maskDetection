from hog_features import extract_hog_features
from evaluator import ModelEvaluator 
from sklearn.svm import SVC

#initialise evaluator
evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])

def train_hog_svm(X_train, y_train):
    print("Extracting HOG features for training...")
    hog_train = extract_hog_features(X_train)

    print("Training SVM classifier...")
    #rbf kernel handles non linea decision boundaries 
    clf = SVC(kernel='rbf', C=10, gamma=0.01)
    clf.fit(hog_train, y_train)
    return clf #the trained model

# evaalujate the svm 
def evaluate_hog_svm(clf, X, y, split_name="Validation"):
    #using the validation set for now so i can see how it performs on unseen data
    print(f"\n Extracting HOG features for {split_name} set...")
    hog_features = extract_hog_features(X)

    print(f"Predicting {split_name} set...")
    y_pred = clf.predict(hog_features)

    # Use evaluator to get and print detailed metrics
    evaluator.evaluate(y, y_pred, model_name=f"HOG + SVM ({split_name})")
    evaluator.plot_confusion_matrix(y, y_pred, title=f"HOG + SVM - {split_name} Confusion Matrix")

    return y_pred #the labels our model predicted

# visualise some results that were predicted
def show_predictions(X, y_true, y_pred, n_samples=4):
    evaluator.visualize_predictions(X, y_true, y_pred, n_samples=n_samples)