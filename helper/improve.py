import warnings
warnings.filterwarnings("ignore")

import time
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def preprocess(data, feature_selector, target_selector, split_size):

    features = data[feature_selector]
    target = data[target_selector]
    xTrain, xTest, yTrain, yTest = train_test_split(features,
                                                    target,
                                                    test_size=split_size,
                                                    random_state=1)
    return xTrain, xTest, yTrain, yTest


def ranking(Actual, Prediction):
    return accuracy_score(Actual, Prediction) * 100


def pipeline(xTrain, xTest, yTrain, yTest, model, weights_name):

    model.fit(xTrain, yTrain)
    joblib.dump(model, weights_name)
    prediction = model.predict(xTest)

    return ranking(yTest, prediction)


def run_v1(data):

    response = {}
    feature_selector = ["X-Axis", "Y-Axis", "Z-Axis"]
    target_selector = ["Activity"]
    split_size = 0.3
    weights_location = "./knn/"

    xTrain, xTest, yTrain, yTest = preprocess(data, feature_selector,
                                              target_selector, split_size)
    for i in tqdm(range(1, 50)):
        model = KNeighborsClassifier(n_neighbors=i)
        response["KNN_K_" + str(i)] = pipeline(
            xTrain, xTest, yTrain, yTest, model,
            (weights_location + "KNN_K_" + str(i) + ".joblib"))

    return {
        k: v
        for k, v in sorted(response.items(), key=lambda item: item[1])
    }
