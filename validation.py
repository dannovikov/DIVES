"""

    # 1. Per subtype sensitivity and specificity.
    #   - Sensitivity: TP / (TP + FN) (aka recall)
    #   - Specificity: TN / (TN + FP)
    #   - TP: True Positive means that the model correctly predicted the subtype.
    #   - TN: True Negative means that the model correctly predicted that the sequence is not of the subtype, so a different subtype.
    #   - FP: False Positive means that the model incorrectly predicted that the sequence is of the subtype.
    #   - FN: False Negative means that the model incorrectly predicted that the sequence is not of the subtype.
    #   - Sensitivity is the proportion of actual positives that are correctly identified as such 
    #   - Specificity is the proportion of actual negatives that are correctly identified as such
    # 2. Per subtype accuracy.
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
            - The numerator is the number of correct predictions for the subtype.
            - The denominator is the total number of predictions for the subtype.
            - Accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined.
    # 3. Per subtype precision.
        - Precision: TP / (TP + FP)
            - The numerator is the number of correct predictions for the subtype.
            - The denominator is the total number of predictions for the subtype.
            - Precision is the proportion of true positives among the total number of positive predictions (both true positives and false positives).

"""

import torch
from tqdm import tqdm
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_model_statistics(model, dataloader, map_label_to_subtype):
    # statistics data structures
    subtype_tp = {}
    subtype_tn = {}
    subtype_fp = {}
    subtype_fn = {}
    subtype_accuracy = {}
    subtype_precision = {}
    subtype_sensitivity = {}
    subtype_specificity = {}

    # initialize data structures
    for label in map_label_to_subtype:
        subtype = map_label_to_subtype[label]
        subtype_tp[subtype] = 0
        subtype_tn[subtype] = 0
        subtype_fp[subtype] = 0
        subtype_fn[subtype] = 0

    # compute statistics
    model.eval()
    with torch.no_grad():
        for x, y, _, _, _, _, _ in tqdm(dataloader, desc="Computing model statistics"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, _, classification = model(x)
            _, predicted = torch.max(classification.data, 1)


            for i in range(len(y)):
                correct_label = y[i].item()
                correct_subtype = map_label_to_subtype[correct_label] 
                prediction = predicted[i].item()
                predicted_subtype = map_label_to_subtype[prediction]

                if prediction == correct_label:
                    subtype_tp[correct_subtype] += 1 # count a true positive
                    for other_label in map_label_to_subtype:
                        if other_label != correct_label:
                            other_subtype = map_label_to_subtype[other_label]
                            subtype_tn[other_subtype] += 1 # count a true negative for all other subtypes

                else:
                    subtype_fn[correct_subtype] += 1 # count a false negative 
                    subtype_fp[predicted_subtype] += 1 # count a false positive
                    for other_label in map_label_to_subtype:
                        if other_label != correct_label and other_label != prediction:
                            other_subtype = map_label_to_subtype[other_label]
                            subtype_tn[other_subtype] += 1 # count a true negative for all other subtypes

    for label in map_label_to_subtype:
        subtype = map_label_to_subtype[label]
        TP, TN, FP, FN = subtype_tp[subtype], subtype_tn[subtype], subtype_fp[subtype], subtype_fn[subtype]
        if FN + TP == 0:
            print("Warning: no positive examples for subtype", subtype)
        subtype_accuracy[subtype] = (TP + TN) / (TP + TN + FP + FN)
        subtype_precision[subtype] = TP / (TP + FP) if TP + FP > 0 else 0
        subtype_sensitivity[subtype] = TP / (TP + FN) if TP + FN > 0 else 0
        subtype_specificity[subtype] = TN / (TN + FP) if TN + FP > 0 else 0

    return subtype_accuracy, subtype_precision, subtype_sensitivity, subtype_specificity





                            




