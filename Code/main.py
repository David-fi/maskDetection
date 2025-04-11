from Code.data_loader import prepare_datasets
from models.hog_svm import train_hog_svm, evaluate_hog_svm, show_predictions
from Code.evaluator import ModelEvaluator
from models.cnn import CNNTrainer

train_image_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/train/images'
train_label_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/train/labels'
test_image_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/test/images'
test_label_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/test/labels'

X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(
    train_image_path, train_label_path,
    test_image_path, test_label_path,
    image_size=(128, 128)
)

'''
# train on training set
hog_svm_model = train_hog_svm(X_train, y_train)
# Evaluate only on validation set
val_preds = evaluate_hog_svm(hog_svm_model, X_val, y_val)
# Visualize some results from validation set
show_predictions(X_val, y_val, val_preds)
'''
'''
from models.sift_mlp import train_sift_mlp, evaluate_sift_mlp, show_predictions

# Train
clf_sift, kmeans_sift = train_sift_mlp(X_train, y_train)
# Evaluate
val_preds_sift = evaluate_sift_mlp(clf_sift, kmeans_sift, X_val, y_val)
# Visualize
show_predictions(X_val, y_val, val_preds_sift)
'''
'''
from models.cnn import train_cnn, evaluate_cnn, show_predictions

# Train CNN
cnn_model, cnn_history = train_cnn(X_train, y_train, X_val, y_val)

# Evaluate
y_pred_cnn = evaluate_cnn(cnn_model, X_val, y_val)

# Visualize
show_predictions(X_val, y_val, y_pred_cnn)

'''

param_grid = [
    ((32, 64, 128), 128, 0.3, 0.001, 32, 15),
    #((32, 64, 128), 256, 0.2, 0.0001, 64, 15),
    ((32, 64, 128, 256), 128, 0.3, 0.0001, 32, 15),
    ((64, 128, 256), 256, 0.2, 0.001, 64, 20),
]

trainer = CNNTrainer()
best_model, best_config = trainer.run_grid_search(X_train, y_train, X_val, y_val, param_grid)
trainer.evaluate_model(best_model, X_val, y_val)
trainer.visualize_predictions(best_model, X_val, y_val)

#best parameters from saved model:
trainer = CNNTrainer()

model = trainer.train_best_model(X_train, y_train, X_val, y_val)

trainer.evaluate_model(model, X_val, y_val)
trainer.visualize_predictions(model, X_val, y_val, n_samples=4)

#test on the test set
trainer.evaluate_model(model, X_test, y_test)
trainer.visualize_predictions(model, X_test, y_test, n_samples=4)

#mask detection in the wild


MaskDetection("/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/masks in the wild database/images")