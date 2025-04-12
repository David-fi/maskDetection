import tensorflow as tf
from keras.api.models import Sequential, load_model
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.api.utils import to_categorical
from keras.api.callbacks import EarlyStopping
from keras.api.optimizers import Adam
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import numpy as np, random, tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
from evaluator import ModelEvaluator
evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])

MODEL_SAVE_PATH_GRID = os.path.join(os.path.dirname(__file__),  '..','..', 'Models', 'best_gridsearch_model.keras')
MODEL_SAVE_PATH_GRID = os.path.abspath(MODEL_SAVE_PATH_GRID)

MODEL_SAVE_PATH_FINAL = os.path.join(os.path.dirname(__file__),  '..','..', 'Models', 'best_CNN_model.keras')
MODEL_SAVE_PATH_FINAL = os.path.abspath(MODEL_SAVE_PATH_FINAL)

def build_cnn(input_shape=(128, 128, 3), num_classes=3, filters=(32, 64, 128), dense_units=128, dropout=0.5):
    model = Sequential() #using a sequential model
    model.add(Input(shape=input_shape)) #defining the input shape (it's 128 by 128 RBG img)

    # mulitple conv to max pooling to dropout blocks 
    for i, f in enumerate(filters):
        model.add(Conv2D(f, (3, 3), activation='relu', padding='same')) #layers where features are detected
        model.add(MaxPooling2D(pool_size=(2, 2)))  #reduces spatial dimensions by 2x, makes feature map smaller and more abstract
        #randomly drop .2 then .3 then .4... neurons in training, the more the deeper the layers
        model.add(Dropout(0.2 + i * 0.1))#dropout increases with layers to prevenet overfitting and help with regularization

    model.add(Flatten())# Flatten ouutputs of previous layers to 1D 
    model.add(Dense(dense_units, activation='relu')) # Fully connected layer, relu helps the network learn non linear boundaries 
    model.add(Dropout(dropout)) #stronger regularization to prevent overfitting before output layer
    model.add(Dense(num_classes, activation='softmax')) # final output layer has a neuron ofr each class, softmax converts scores to class probs

    return model

#compiles model with an optimizer and a loss function
def compile_model(model, lr):
    model.compile(
        optimizer=Adam(learning_rate=lr),  #adaptive learning rate optimizer
        loss='categorical_crossentropy',#good for mulit class classification
        metrics=['accuracy']#keep track during training and evaluation
    )
    return model

#the class where we train, eval, and tune the cnn
class CNNTrainer:
    #call the training of a model with the best parameters we found
    def train_best_model(self, X_train, y_train, X_val, y_val):
        #input_shape = X_train.shape[1:] allows the model to be flexible to different resolutions 
          
          
        model_path = os.path.join(MODEL_SAVE_PATH_GRID)
        model = load_model(model_path)  # Load the best model found from grid search
        print("Loaded model parameters:")
        print(model.get_config())
        
        # #  # Define fixed best hyperparameters
        # # filters = (64, 128, 256)
        # # dense_units = 128
        # # dropout = 0.3
        # # lr = 0.001
        # # batch_size = 32
        # # epochs = 15

        # # Build and compile model with those params
        # model = build_cnn(input_shape=X_train.shape[1:], num_classes=3,
        #                 filters=filters, dense_units=dense_units, dropout=dropout)
        # model = compile_model(model, lr)

        # print("Using manually selected best config:")
        # print((filters, dense_units, dropout, lr, batch_size, epochs))
        # print(model.get_config())

        # Convert labels to categorical (1 -> [0, 1, 0])
        y_train_cat = to_categorical(y_train, num_classes=3)
        y_val_cat = to_categorical(y_val, num_classes=3)
        
        #to handle the imbalances in the dataset im using the class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: class_weights[i] for i in range(3)}

        #prevent overfitting in accordance to validation loss
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        #now we train the model
        model.fit(X_train, y_train_cat,
                  validation_data=(X_val, y_val_cat),
                  epochs=20,
                  batch_size=32,
                  class_weight=class_weights_dict, #prevent bias to majority class like correct
                  callbacks=[early_stop],
                  verbose=1)
        #save the model for reuse
        # Using model.save() for CNN as it's a Keras model (saves full model + weights).
        # joblib is used for simpler models like SVM/MLP that don't need architecture saved.    
        model.save(MODEL_SAVE_PATH_FINAL)
        return model

#testing hyperparameter combinations to find the best ones
#i went for a manual gridsearch to have full control over the specificic combinations
#as the classes are imbalanced f1 score is the best to determine the models performance
    def run_grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        best_f1 = -1 #track the best f1 score
        best_model = None
        best_config = None

        #loop through the parameter grid to check which combination is the best perfoming
        for filters, dense_units, dropout, lr, batch_size, epochs in param_grid:
            print(f"\n\U0001F50D Training model with filters={filters}, dense_units={dense_units}, "
                  f"dropout={dropout}, lr={lr}, batch_size={batch_size}, epochs={epochs}")

            #build model with a flexible input in case of different resolution
            model = build_cnn(input_shape=X_train.shape[1:], num_classes=3,
                              filters=filters, dense_units=dense_units, dropout=dropout)
            #compile 
            model = compile_model(model, lr)
            # Convert labels to categorical (1 -> [0, 1, 0])
            y_train_cat = to_categorical(y_train, num_classes=3)
            y_val_cat = to_categorical(y_val, num_classes=3)
            #to handle the imbalances in the dataset im using the class weight
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
            class_weights_dict = dict(enumerate(class_weights))

             #prevent overfitting in accordance to validation loss
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            #train the model
            model.fit(X_train, y_train_cat,
                      validation_data=(X_val, y_val_cat),
                      epochs=epochs,
                      batch_size=batch_size,
                      class_weight=class_weights_dict,
                      callbacks=[early_stop],
                      verbose=1)

            y_pred = np.argmax(model.predict(X_val), axis=1) #convert the softmax output, to the class labels
            results = evaluator.evaluate(y_val, y_pred, model_name=f"GridCNN {filters}-{dense_units}-{dropout}")
            evaluator.plot_confusion_matrix(y_val, y_pred) #plot the confusion matrix

            if results['f1_macro'] > best_f1: #using marco f1 puts a foucs on the perfomance across all classes
                best_f1 = results['f1_macro']
                best_model = model
                best_config = (filters, dense_units, dropout, lr, batch_size, epochs)

        print(f"\n\U0001F3C6 Best Configuration: {best_config}, Macro F1: {best_f1:.4f}")
        # Using model.save() for CNN as it's a Keras model (saves full model + weights).
        # joblib is used for simpler models like SVM/MLP that don't need architecture saved.
        best_model.save(MODEL_SAVE_PATH_GRID)
        return best_model, best_config

    #evalute the final model with the metrics and confusion matrix from the evaluator class
    def evaluate_model(self, model, X_val, y_val):
        y_pred = np.argmax(model.predict(X_val), axis=1)
        evaluator.evaluate(y_val, y_pred, model_name="Final Model")
        evaluator.plot_confusion_matrix(y_val, y_pred)
        return y_pred

    #visualize some results for the report
    def visualize_predictions(self, model, X_val, y_val, n_samples=4):
        y_pred = np.argmax(model.predict(X_val), axis=1)
        evaluator.visualize_predictions(X_val, y_val, y_pred, n_samples=n_samples)
