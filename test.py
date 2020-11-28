import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as nnf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd


def evaluate_model(model, test_dataset, batch_size):
    """
    :param model:
    :param test_dataset: WOSDataset object
    :param batch_size:
    :return: results in dictionary form
    """
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # test on gpu
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    result = {"accuracy": 0.,
              "labels": [],
              "predictions": [],
              "probabilities": [],
              "performance_measures": {},
              "confusion_matrix": []
              }

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_num_corrects = 0
    total_instances = 0

    cm = np.zeros((2, 2))
    tp_fp_tn_fn = [0, 0, 0, 0]

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            title = batch[0]
            abstract = batch[1]
            target = batch[2]

            if torch.cuda.is_available():
                title = title.cuda()
                abstract = abstract.cuda()
                target = target.cuda()

            # convert to float-long tensor
            title = title.float()
            abstract = abstract.float()
            target = torch.autograd.Variable(target).long()

            # make predictions
            predictions_raw = model(title, abstract)
            predictions = torch.max(predictions_raw, 1)[1]

            # get probabilities (for ROC)
            probabilities = nnf.softmax(predictions_raw, dim=1)
            # all_probabilities = np.concatenate((all_probabilities, probabilities.data.tolist()))
            [all_probabilities.append(x.data.tolist()) for x in probabilities]
            # calculate loss
            loss = F.cross_entropy(predictions_raw, target)

            # confusion matrix
            cm += confusion_matrix(target.cpu(), predictions.cpu())
            tp_fp_tn_fn = np.add(tp_fp_tn_fn, calculate_performance_measures(target.cpu(), predictions.cpu()))

            all_labels = np.concatenate((all_labels, target.data.tolist()))
            all_predictions = np.concatenate((all_predictions, predictions.tolist()))

            num_corrects = (predictions.view(target.size()).data == target.data).sum()
            total_num_corrects += num_corrects
            total_instances += len(batch[0])  # same as len(dataloader), but easier to read this way

            accuracy = total_num_corrects / total_instances * 100
            # print(f"Total Correct: {total_num_corrects}, Total Instance: {total_instances} Accuracy: {accuracy}")

    result["accuracy"] = accuracy
    result["labels"] = np.array(all_labels, dtype=np.int)
    result["predictions"] = np.array(all_predictions, dtype=np.int)
    result["probabilities"] = np.array(all_probabilities)
    result["confusion_matrix"] = cm
    result["performance_measures"] = {"true_positives": tp_fp_tn_fn[0],
                                      "true_negatives": tp_fp_tn_fn[2],
                                      "false_positives": tp_fp_tn_fn[1],
                                      "false_negatives": tp_fp_tn_fn[3]
                                      }

    return result


def calculate_performance_measures(labels, predictions):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    """
    We can simply get measures from confusion matrix since we are doing binary classification.
    However, this is a better way which can be used when there are more than 2 classes.
    """
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(predictions)):
        if labels[i] == predictions[i] == 1:
            true_positive += 1
        if predictions[i] == 1 and labels[i] != predictions[i]:
            false_positive += 1
        if labels[i] == predictions[i] == 0:
            true_negative += 1
        if predictions[i] == 0 and labels[i] != predictions[i]:
            false_negative += 1

    return [true_positive, false_positive, true_negative, false_negative]


def draw_roc(result):
    """
    Draws ROC curve using the results obtained from evaluate_model method
    :param result: result dictionary object obtained from evaluate_model method
    :return: None
    """
    labels = result["labels"]
    probs = result["probabilities"]

    # keep probabilities for the positive outcome only
    probs = probs[:, 1]

    # calculate scores
    lr_auc = roc_auc_score(labels, probs)

    # summarize scores
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(labels, probs)
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker='.')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def draw_precision_recall_curve(result):
    """
    :param result: result dictionary object obtained from evaluate_model method
    :return: None
    """

    labels = result["labels"]
    probs = result["probabilities"]
    probs = probs[:, 1]

    # calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(labels, probs)

    plt.plot(recall, precision, marker='.')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def calculate_f1_score(result):
    """
    :param result: result dictionary object obtained from evaluate_model method
    :return: float, f1 score
    """
    labels = result["labels"]
    predictions = result["predictions"]

    # calculate F1 score
    f1 = f1_score(labels, predictions)

    return f1

def draw_confusion_matrix(cm):
    """
    Visualize confusion matrix using values obtained from evaluate_model method
    :param cm: Confusion matrix 2D array
    :return: None
    """

    array = np.array(cm, dtype=np.int)
    df_cm = pd.DataFrame(array, range(2), range(2))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)  # for label size
    labels = ["M. Science", "Chemistry"]
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, xticklabels=labels, yticklabels=labels)  # font size

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()

def plot_accuracies(train_accs, test_accs):
    """
    Plot
    :param train_accs: train accuracies for each epoch
    :param test_accs: test accuracies for each epoch
    :return: None
    """

    # plt.plot(range(len(self.train_losses)), self.train_losses, label="line 1")
    plt.plot(range(len(train_accs)), train_accs, label="Train Accuracy")
    plt.plot(range(len(test_accs)), test_accs, label="Test Accuracy")
    plt.xlabel('Epoch')
    # Set the y axis label of the current axis.
    plt.ylabel('Accuracy')
    # Set a title of the current axes.
    plt.title('Accuracy over epochs on train and test datasets.')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

    # plt.plot('x', 'y1', data=self.train_accs, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    # plt.plot('x', 'y2', data=self.train_losses, marker='', color='olive', linewidth=2)
    # plt.legend()
    # plt.show()