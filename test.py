import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
loss_fn = F.cross_entropy


def calculate_performance_measures(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return TPR, TNR, FPR, FNR

def calculate_performance_measures_2(y_actual, y_hat):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    """
    We can simply get measures from confusion matrix since we are doing binary classification.
    However, this is a better way which can be used when there are more than 2 classes.
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return [TP, FP, TN, FN]



def evaluate_model(model, test_dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # test on gpu
    model.cuda()
    model.eval()

    result = {"accuracy": 0., "performance_measures": {}, "confusion_matrix": []}

    total_num_corrects = 0
    total_instances = 0
    total_epoch_loss = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    cm_y_true = []
    cm_y_pred = []
    cm = np.zeros((2, 2))
    TP_FP_TN_FN = [0, 0, 0, 0]
    prec_rec_fscore_sup = np.zeros((4,2))

    last_idx = math.floor(len(test_dataset) / batch_size)
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            title = batch[0]
            abstract = batch[1]
            target = batch[2]
            target = torch.autograd.Variable(target).long()

            if torch.cuda.is_available():
                title = title.cuda()
                abstract = abstract.cuda()
                target = target.cuda()

            # convert to float tensor
            title = title.float()
            abstract = abstract.float()

            # make predictions
            predictions_raw = model(title, abstract)
            predictions = torch.max(predictions_raw, 1)[1]

            # calculate loss
            loss = loss_fn(predictions_raw, target)

            # confusion matrix
            cm += confusion_matrix(target.cpu(), predictions.cpu())
            TP_FP_TN_FN = np.add(TP_FP_TN_FN, calculate_performance_measures_2(target.cpu(), predictions.cpu()))

            num_corrects = (predictions.view(target.size()).data == target.data).sum()
            total_num_corrects += num_corrects
            total_instances += len(batch[0])  # same as len(dataloader), but easier to read this way
            total_epoch_loss += loss.item()

            accuracy = total_num_corrects / total_instances * 100
            # print(f"Total Correct: {total_num_corrects}, Total Instance: {total_instances} Accuracy: {accuracy}")

    TPR, TNR, FPR, FNR = TP_FP_TN_FN[0], TP_FP_TN_FN[2], TP_FP_TN_FN[1], TP_FP_TN_FN[3]
    result["accuracy"] = accuracy
    result["confusion_matrix"] = cm
    result["performance_measures"] = {"true_positives": TPR, "true_negatives": TNR, "false_positives": FPR, "false_negatives": FNR,}

    return result




