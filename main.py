from dataLoader import prepare_datasets
from models.hogSvm import train_hog_svm, evaluate_hog_svm, show_predictions
from evaluator import ModelEvaluator

train_image_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/train/images'
train_label_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/train/labels'
test_image_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/test/images'
test_label_path = '/Users/david/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents – David’s MacBook Pro/university/year 3/Computer vision/cw/CV2024_CW_Dataset/test/labels'

X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(
    train_image_path, train_label_path,
    test_image_path, test_label_path,
    image_size=(128, 128)
)


# train on training set
hog_svm_model = train_hog_svm(X_train, y_train)
# Evaluate only on validation set
val_preds = evaluate_hog_svm(hog_svm_model, X_val, y_val)
# Visualize some results from validation set
show_predictions(X_val, y_val, val_preds)
