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
        # DESCRIPTION:
        # -------------------------
        # The objective of the dataset is to diagnostically predict whether or not a patient has diabetes,
        # based on certain diagnostic measurements included in the dataset. Several constraints were placed on the
        # selection of these instances from a larger database. In particular, all patients here are females at least
        # 21 years old of Pima Indian heritage.
        # -------------------------
        # NUMBER OF INSTANCES: 768
        # NUMBER OF ATTRIBUTES:  8 plus class (all numeric-valued)
        #       1. Number of times pregnant
        #       2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
        #       3. Diastolic blood pressure (mm Hg)
        #       4. Triceps skin fold thickness (mm)
        #       5. 2-Hour serum insulin (mu U/ml)
        #       6. Body mass index (weight in kg/(height in m)^2)
        #       7. Diabetes pedigree function
        #       8. Age (years)
        #       9. Class variable (0 or 1)
        # -------------------------
        # KAGGLE: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
        # MORE INFO: https://www.openml.org/search?type=data&sort=runs&id=37&status=active
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: https://link.springer.com/article/10.1186/s12911-024-02582-4
        # - code: https://github.com/ChristelSirocchi/medical-informed-ML
        # DOUBTS: Are the missing values? How are features scaled in other approaches?

        # Assuming the diabetes dataset is stored in 'data/datasets/diabetes.csv'
        file_path = "data/datasets/diabetes.csv"

        df = pd.read_csv(file_path)
        x = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values  # Target variable

        if data_encoding == "one-hot-encoding":
            y = get_one_encoding(y)
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
        # DESCRIPTION:
        # -------------------------
        # This data set can be used to predict the severity (benign or malignant) of a mammographic mass lesion
        # from BI-RADS attributes and the patient's age. It contains a BI-RADS assessment, the patient's age and three
        # BI-RADS attributes together with the ground truth (the severity field) for 516 benign and 445 malignant masses
        # that have been identified on full field digital mammograms collected at the Institute of
        # Radiology of the University Erlangen-Nuremberg between 2003 and 2006
        # -------------------------
        # NUMBER OF INSTANCES: should be 961 but in Paulo's version is 830
        # NUMBER OF ATTRIBUTES:  5 plus class
        #       1. BI-RADS assessment: 1 to 5 (ordinal, non-predictive!)
        #       2. Age: patient's age in years (integer)
        #       3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
        #       4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
        #       5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
        #       6. Class variable: Severity: benign=0 or malignant=1 (binominal, goal field!)
        # MISSING VALUES:
        #     - BI-RADS assessment:    2
        #     - Age:                   5
        #     - Shape:                31
        #     - Margin:               48
        #     - Density:              76
        #     - Severity:              0
        # -------------------------
        # UCI: https://archive.ics.uci.edu/dataset/161/mammographic+mass
        # POSSIBLE COMPARISON (TO INVESTIGATE):
        # - paper: https://link.springer.com/article/10.1007/s10552-024-01942-9
        # - code: ?
        # DOUBTS: How are missing values handled and features scaled in other approaches?

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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    x_train_normalized = scaler.fit_transform(x_train)
    x_test_normalized = scaler.transform(x_test)

    data_train = (x_train_normalized, y_train)
    data_test = (x_test_normalized, y_test)

    return data_train, data_test, map_class_dict
