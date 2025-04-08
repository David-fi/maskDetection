from sklearn.neural_network import MLPClassifier
from evaluator import ModelEvaluator
from sift_features import extract_sift_descriptors, build_visual_vocabulary, compute_bovw_histograms

evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])

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

    print("🧠 Training MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128,),#one hidden layer with 128 neuronr
        max_iter=500, # 500 training iterations
        random_state=42 #for reproducibility
        )
    clf.fit(bovw_train, y_train_filtered) #training the MLP on BoVW features and label

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