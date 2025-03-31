import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# 5) Per-class and mean accuracy
cm = confusion_matrix(all_labels, all_preds)
per_class_acc = cm.diagonal() / cm.sum(axis=1)  # diag / row_sum
mean_acc = per_class_acc.mean()

# 6) Precision, recall, F1 for each class
precision_cls = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall_cls = recall_score(all_labels, all_preds, average=None, zero_division=0)
f1_cls = f1_score(all_labels, all_preds, average=None, zero_division=0)

# 7) Macro-averaged precision, recall, and F1
precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

# 8) Super-classes: map each class ID to a broader label
# Example: classes 0,1 -> 'A'; classes 2,3 -> 'B'; class 4 -> 'C'
superclass_map = {0:'A', 1:'A', 2:'B', 3:'B', 4:'C'}
# Extend this mapping if you have more than 5 classes

super_true = np.array([superclass_map[c] for c in all_labels])
super_pred = np.array([superclass_map[c] for c in all_preds])
unique_super = np.unique(list(superclass_map.values()))

# Repeat metrics for super-classes
super_precision = precision_score(super_true, super_pred, labels=unique_super, average=None, zero_division=0)
super_recall    = recall_score(super_true, super_pred, labels=unique_super, average=None, zero_division=0)
super_f1        = f1_score(super_true, super_pred, labels=unique_super, average=None, zero_division=0)

# 9) Report results
print("\n--- Per-Class Accuracy ---")
for idx, acc in enumerate(per_class_acc):
    print(f"Class {idx}: {acc*100:.2f}%")
print(f"Mean Accuracy Across Classes: {mean_acc*100:.2f}%")

print("\n--- Precision, Recall, F1 by Class ---")
num_classes = len(per_class_acc)
for i in range(num_classes):
    print(f"Class {i} | "
          f"Precision: {precision_cls[i]*100:.2f}%, "
          f"Recall: {recall_cls[i]*100:.2f}%, "
          f"F1: {f1_cls[i]*100:.2f}%")

print("\n--- Macro-Averaged Metrics ---")
print(f"Macro Precision: {precision_macro*100:.2f}%")
print(f"Macro Recall:    {recall_macro*100:.2f}%")
print(f"Macro F1:        {f1_macro*100:.2f}%")

print("\n--- Super-Class Metrics ---")
for s_label, sp, sr, sf in zip(unique_super, super_precision, super_recall, super_f1):
    print(f"Super-Class {s_label} | "
          f"Precision: {sp*100:.2f}%, "
          f"Recall: {sr*100:.2f}%, "
          f"F1: {sf*100:.2f}%")

