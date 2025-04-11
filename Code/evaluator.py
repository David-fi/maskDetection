from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

class ModelEvaluator:
    def __init__(self, class_names=["No Mask", "Mask", "Incorrect"]):
        # instead of having the 0,1,2 labels usd actual descriptors for clarity 
        self.class_names = class_names

    def evaluate(self, y_true, y_pred, model_name="Model", verbose=True):
      
        results = {}

        # accuracy score 
        acc = accuracy_score(y_true, y_pred)
        results['accuracy'] = acc

        # Macro = equally weighs all classes (sensitive to minority classes)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Micro = globally computed over all samples for class imbalance)
        results['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        results['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # Weighted = like macro but adjusts for class imbalance by using class frequencies
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        if verbose:
            print(f"\nEvaluation results for {model_name}:")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score (macro): {results['f1_macro']:.4f}")
            print(f"F1 Score (weighted): {results['f1_weighted']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))

        return results

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        #plot a confusion matrix 

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True label')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def visualize_predictions(self, images, y_true, y_pred, n_samples=4):
        
        #Display somerandom image samples with ground truth and predicted labels, this will be useful for examples if i use them in the report
        indices = random.sample(range(len(images)), n_samples)
        plt.figure(figsize=(12, 6))
        for i, idx in enumerate(indices):
            img = images[idx]
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            plt.subplot(1, n_samples, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.tight_layout()
        plt.show()
#initialise evaluator
evaluator = ModelEvaluator(class_names=["No Mask", "Mask", "Incorrect"])