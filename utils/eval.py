from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
from imblearn.metrics import geometric_mean_score, specificity_score

import matplotlib.pyplot as plt
import seaborn as sns

def score(X_test, y_test, model):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.6f}")
    print(f"Error Rate: {1-acc}")
    
    # calculate micro/macro-average precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    
    # calculate selectivity
    selectivity = specificity_score(y_test, y_pred, average='macro')
    print(f"Selectivity: {selectivity:.6f}")

    # calculate balanced accuracy
    bacc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {bacc:.6f}")

    # calculate geometric mean
    gmean = geometric_mean_score(y_test, y_pred, average='macro')
    print(f"Geometric Mean: {gmean:.6f}")

    # calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # plot confusion matrix as heatmap
    plt.figure(figsize=(3,2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # # calculate AUROC score
    # auroc = roc_auc_score(y_test, y_pred, multi_class='ova')
    # print(f"AUROC score: {auroc:.6f}")
    
    # # plot AUROC curve
    # fpr, tpr, _ = roc_curve(y_test, y_pred, multi_class='ova')
    # plt.figure(figsize=(4,4))
    # plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], '--', color='grey')
    # plt.title('ROC Curve')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()
