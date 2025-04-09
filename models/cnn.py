import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.utils import to_categorical
from keras.api.callbacks import EarlyStopping
import numpy as np
from evaluator import ModelEvaluator 

evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])

def build_cnn(input_shape=(128, 128, 3), num_classes=3):
    model = Sequential() #using a sequential model
    #initialise a stack of layers 

    # First conv layer, apply 32 filters of size 3x3
    # ReLU activation introduces non-linearity
    # first layer where input shape is defined
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

    #reduces spatial dimensions by 2x, makes feature map smaller and more abstract
    model.add(MaxPooling2D(pool_size=(2, 2))) 

    #randomly turn off 1/4 of the neurons in training 
    model.add(Dropout(0.25))  # prevent overfitting by forcing robustness 

    #the second layer now has double the filters for deeper feature learning 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))    

    #once again double the filters for even deeper learning 
    model.add(Flatten())  # Flatten ouutputs of previous layers to 1D 
    model.add(Dense(128, activation='relu'))  # Fully connected layer, relu helps the network learn non linear boundaries 
    model.add(Dropout(0.5)) #stronger regularization to prevent overfitting before output layer
    model.add(Dense(num_classes, activation='softmax'))  # final output layer has a neuron ofr each class, softmax converts scores to class probs

    model.compile(
        optimizer='adam', #adaptive learning rate optimizer
        loss='categorical_crossentropy', #good for mulit class classification
        metrics=['accuracy'] #keep track during training and evaluation
    )

    return model

def train_cnn(X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=3) #buliding the cnn with the input data, and the numb of classes from the data

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) #stops training early if val loss doesnt improve in 3 epochs, avoid overfitting

    # Convert labels to categorical (1 -> [0, 1, 0])
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)

    history = model.fit(
        X_train, y_train_cat, 
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size, #how many samples are processed at a time
        verbose=1, #progress bar 
        callbacks=[early_stop] #early stopping
    )

    return model, history

def evaluate_cnn(model, X_val, y_val):
    y_pred_prob = model.predict(X_val)  # probabilities for each class
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert softmax outputs to class indices

    # Use evaluator class for metrics and confusion matrix
    evaluator.evaluate(y_val, y_pred, model_name="Simple CNN")
    evaluator.plot_confusion_matrix(y_val, y_pred, title="Simple CNN - Validation Confusion Matrix")

    return y_pred

def show_predictions(X_val, y_val, y_pred, n_samples=4):
    #preview of some predictions 
    evaluator.visualize_predictions(X_val, y_val, y_pred, n_samples=n_samples)