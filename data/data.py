import numpy as np
import pandas as pd  # Import pandas to load and process the CSV file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris


def get_one_encoding(labels):
    # Set sparse_output=False to get dense array
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(labels.reshape(-1, 1))

    return one_hot_encoded


def get_data(dataset, data_encoding, seed=0, test_size=0.3):
    """
    Get data of the required dataset

    Parameters:
        - dataset (str): Name of the dataset.
        - data_encoding (str) : Specifies the data encoding method to be used. This parameter affects how data (i.e., target labels)
          is processed within the model. Example: 'no-encoding', 'one-hot-encoding'.
        - seed (int): Seed for random number generation (used only for synthetic exp).
        - test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        - data_train (tuple): Tuple containing features and labels for the training set.
        - data_test (tuple): Tuple containing features and labels for the testing set.
        - map_class_dict (dict): A dictionary that maps the predicted class values (used internally by the model)
          to their original dataset class values.
    """

    map_class_dict = {}
    if dataset == "synthetic":
        rng = np.random.default_rng(seed)
        # Synthetic dataset generation code here

    elif dataset == "iris":
        # Load the Iris dataset
        iris = load_iris()
        x = iris.data
        y = iris.target

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            # Convert to binary classification problem
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y[y > 0] = 1
            y.reshape(-1, 1)

    elif dataset == "diabetes":
        # Assuming the diabetes dataset is stored in 'diabetes.csv'
        file_path = "data/datasets/diabetes.csv"

        # Lendo o arquivo
        df = pd.read_csv(file_path)
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
            map_class_dict = {c-1:c for c in np.unique(y)}
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "liver":
        # Assuming the diabetes dataset is stored in 'liver_data.txt'
        file_path = "data/datasets/liver_data.txt"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
            map_class_dict[0] = 1
            map_class_dict[1] = 2
        elif data_encoding == "no-encoding":
            y[y == 2] = -1
            map_class_dict[-1] = 2
            y = y.reshape(-1, 1)

    elif dataset == "mammography":
        # Assuming the diabetes dataset is stored in 'data/mammographic_masses.txt'
        file_path = "data/datasets/mammographic_masses.txt"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "diabetes":
        # Assuming the diabetes dataset is stored in 'data/diabetes.csv'
        file_path = "data/datasets/diabetes.csv"

        df = pd.read_csv(file_path)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "gestacional":
        # Assuming the gestacional dataset is stored in 'data/gestacionaldiabetes.csv'
        file_path = "data/datasets/gestacionaldiabetes.csv"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "sepsis":
        # Assuming the sepsis dataset is stored in 'data/sepsis.csv'
        file_path = "data/datasets/sepsis.csv"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "preeclampsia":
        # Assuming the preclampsia dataset is stored in 'data/preeclampsia.csv'
        file_path = "data/datasets/preeclampsia.csv"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "transfusion":
        # Assuming the transfusion dataset is stored in 'data/transfusion.txt'
        file_path = "data/datasets/Transfusion.txt"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
        elif data_encoding == "no-encoding":
            y[y == 0] = -1
            map_class_dict[-1] = 0
            y = y.reshape(-1, 1)

    elif dataset == "haberman":
        # Assuming the haberman dataset is stored in 'data/haberman'
        file_path = "data/datasets/haberman.data"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
            map_class_dict[0] = 1
            map_class_dict[1] = 2
        elif data_encoding == "no-encoding":
            y[y == 2] = -1
            map_class_dict[-1] = 2
            y = y.reshape(-1, 1)

    elif dataset == "heart":
        # Assuming the heart dataset is stored in 'data/heart.txt'
        file_path = "data/datasets/heart.txt"

        df = pd.read_csv(file_path, sep=",")
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
            map_class_dict[0] = 1
            map_class_dict[1] = 2
        elif data_encoding == "no-encoding":
            y[y == 2] = -1
            map_class_dict[-1] = 2
            y = y.reshape(-1, 1)

    # Data normalization
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_normalized, y, test_size=test_size, random_state=seed
    )

    data_train = (x_train, y_train)
    data_test = (x_test, y_test)

    return data_train, data_test, map_class_dict
