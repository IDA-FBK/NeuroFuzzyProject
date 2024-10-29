import argparse
import os.path

import numpy as np
from numpy.linalg import pinv
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.plots import plot_class_confusion_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", type=str, default="iris", help="specify the dataset to use"
    )
    parser.add_argument(
        "-path_to_conf",
        type=str,
        default="./experiments/configurations/iris/conf-01.json",
        help="configuration file for the current experiment",
    )
    parser.add_argument(
        "-path_to_results",
        type=str,
        default="./experiments/results/baseline-1/iris/",
        help="directory where to store the results",
    )

    args = parser.parse_args()

    dataset = args.dataset
    path_to_conf = args.path_to_conf
    path_to_results = args.path_to_results + f"/{dataset}/"
    conf = get_configuration(path_to_conf)

    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results, exist_ok=True)

    # experiment setting
    optimizer = conf["optimizer"]
    data_encoding = conf["data_encoding"]
    pred_method = conf["pred_method"]

    print("EXPERIMENT:\n")
    print(f"---- Optimizer = {optimizer}\n")

    data_train, data_test, map_class_dict = get_data(dataset, data_encoding)
    # this store the results of each run
    results_df = pd.DataFrame(
        columns=[
            "Accuracy",
            "Fscore",
            "Recall",
            "Precision",
        ]
    )

    x_train, y_train = data_train[0], data_train[1]
    x_test, y_test = data_test[0], data_test[1]

    V = np.dot(pinv(x_train), y_train)

    output_v = np.dot(x_test, V)

    if data_encoding == "no-encoding" and pred_method == "sign":
        y_pred = np.sign(output_v)

        # show again original classes
        y_test[y_test == -1] = map_class_dict[-1]
        y_pred[y_pred == -1] = map_class_dict[-1]

    elif data_encoding == "one-hot-encoding" and pred_method == "argmax":
        y_pred = np.argmax(output_v, 1)
        y_test = np.argmax(y_test, 1)
        if map_class_dict:
            # show again original classes
            y_pred[y_pred == 1] = map_class_dict[1]
            y_pred[y_pred == 0] = map_class_dict[0]

            y_test[y_test == 1] = map_class_dict[1]
            y_test[y_test == 0] = map_class_dict[0]

    n_class = len(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred)

    if n_class == 2:
        average_metrics = "binary"

        # Unravel confusion matrix elements
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        average_metrics = "macro"
        # For multi-class, we need to calculate specificity for each class in a one-vs-rest way
        specificities = []
        for i in range(n_class):
            # True positives for class i
            tp = cm[i, i]
            # All non-class i elements
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            # False positives for class i
            fp = np.sum(cm[:, i]) - cm[i, i]
            # False negatives for class i
            fn = np.sum(cm[i, :]) - cm[i, i]

            specificity_i = tn / (tn + fp)  # Specificity for class i
            specificities.append(specificity_i)

        specificity = np.mean(specificities)  # Average specificity for multi-class

    accuracy = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred, average=average_metrics), 3)
    recall = round(recall_score(y_test, y_pred, average=average_metrics), 3)
    f1 = round(f1_score(y_test, y_pred, average=average_metrics))

    # Plot confusion matrix of class prediction
    plot_class_confusion_matrix(cm, np.unique(y_test), path_to_results)

    # Prints the metrics
    print(f"\nAccuracy: {accuracy}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-Score: {f1}")

    # Save metrics to a file named after the dataset
    metrics_filename = path_to_results+f"{dataset}_metrics.txt"
    with open(metrics_filename, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Specificity: {specificity}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F-Score: {f1}\n")

    print(f"\nMetrics saved in {metrics_filename}")
