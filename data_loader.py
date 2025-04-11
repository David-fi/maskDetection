import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter


def load_images_and_labels(image_dir, label_dir, image_size=(128, 128)):
    images = [] 
    labels = []

    #go through the image files in the folder
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpeg'):
            #make the path for the image and that images label
            img_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpeg', '.txt'))

            # Load and resize image 
            img = imread(img_path) 
            #images might come in many sizes, making them consitent makes feature extraction easier 
            #also normalises to help in our model development
            img_resized = resize(img, image_size, anti_aliasing=True) 
            images.append(img_resized)  # add to the list 

            # Read label
            with open(label_path, 'r') as f:
                label = int(f.read().strip()) #read the label and convert it to an interger
            labels.append(label) # add to the list 

    #converts the two lists, image and label into numpy arrays 
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

def plot_class_distribution(labels, title):
    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = [counter[c] for c in classes]
    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts, tick_label=[f"Class {c}" for c in classes])
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
def prepare_datasets(
    train_image_path, train_label_path,
    test_image_path, test_label_path,
    image_size=(128, 128),
    val_split=0.2,
    seed=42
):
    # Load the whole training set both labels and images, they are resized and normalised 
    X_train_full, y_train_full = load_images_and_labels(train_image_path, train_label_path, image_size)
    
    #same with the test set 
    X_test, y_test = load_images_and_labels(test_image_path, test_label_path, image_size)


    # Show distribution before split
    plot_class_distribution(y_train_full, "Train Full Distribution")
    plot_class_distribution(y_test, "Test Distribution")

    # Split training data into train validation, standard split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_split, random_state=seed, stratify=y_train_full #ensure smae label distribution with stratify
    )

    # Show distribution after split
    plot_class_distribution(y_train, "Train Split Distribution")
    plot_class_distribution(y_val, "Validation Split Distribution")

    #sanity check
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Validation: {X_val.shape}, {y_val.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test