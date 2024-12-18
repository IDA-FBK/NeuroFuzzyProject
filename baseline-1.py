import argparse
import os.path

import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.plots import plot_class_confusion_matrix

#### This could be considered a baseline where the pseudoinverse is computed on the normalized features. ####

def compute_metrics(y_true, output_v):
    if data_encoding == "no-encoding" and pred_method == "sign":
        y_pred = np.sign(output_v)

        # show again original classes
        y_true[y_true == -1] = map_class_dict[-1]
        y_pred[y_pred == -1] = map_class_dict[-1]
    elif data_encoding == "one-hot-encoding" and pred_method == "argmax":
        y_pred = np.argmax(output_v, 1)
        y_true = np.argmax(y_true, 1)
        if map_class_dict:
            # show again original classes
            y_pred[y_pred == 1] = map_class_dict[1]
            y_pred[y_pred == 0] = map_class_dict[0]
            y_true[y_true == 1] = map_class_dict[1]
            y_true[y_true == 0] = map_class_dict[0]

    unique_classes = np.unique(y_true)

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    n_class = len(np.unique(y_true))
    if n_class == 2:
        average_metrics = "binary"
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        average_metrics = "macro"
        specificities = []
        for i in range(n_class):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(cm[:, i]) - cm[i, i]

            specificity_i = tn / (tn + fp)
            specificities.append(specificity_i)
        specificity = np.mean(specificities)

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, average=average_metrics), 3),
        "recall": round(recall_score(y_true, y_pred, average=average_metrics), 3),
        "f1": round(f1_score(y_true, y_pred, average=average_metrics), 3),
        "specificity": specificity
    }
    return metrics, cm, unique_classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", type=str, default="diabetes", help="specify the dataset to use"
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
    path_to_results = args.path_to_results
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

    # Split train and test data
    x_train, y_train = data_train[0], data_train[1]
    x_test, y_test = data_test[0], data_test[1]

    # Model training
    V = np.dot(pinv(x_train), y_train)

    # Predictions on train and test sets
    output_train = np.dot(x_train, V)
    output_test = np.dot(x_test, V)


    # Get metrics for train and test data
    train_metrics, cm_train, unique_classes = compute_metrics(y_train.copy(), output_train)
    test_metrics, cm_test, _ = compute_metrics(y_test.copy(), output_test)

    # Plot confusion matrix for train and test
    plot_class_confusion_matrix("TRAIN", cm_train, unique_classes, path_to_results)
    plot_class_confusion_matrix("TEST", cm_test, unique_classes, path_to_results)

    # Save metrics
    metrics_filename = path_to_results + f"{dataset}_metrics.txt"
    with open(metrics_filename, "w") as f:
        f.write("TRAIN METRICS:\n")
        for metric, value in train_metrics.items():
            f.write(f"{metric.capitalize()}: {value}\n")
        f.write("\nTEST METRICS:\n")
        for metric, value in test_metrics.items():
            f.write(f"{metric.capitalize()}: {value}\n")

    print(f"\nMetrics saved in {metrics_filename}")
