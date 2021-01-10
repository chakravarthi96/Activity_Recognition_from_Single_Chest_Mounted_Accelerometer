import warnings
warnings.filterwarnings("ignore")

import time
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier

# Logistic
Logistic_Regression = LogisticRegression(multi_class='multinomial',
                                         random_state=1)
# Neighbors Algorithms
K_Neighbors_Classifier = KNeighborsClassifier(n_neighbors=14)

# Naive Bayes Algorithms
Bernoulli_Naive_Bayes = BernoulliNB()
Complement_Naive_Bayes = ComplementNB()
Gaussian_Naive_Bayes = GaussianNB()
Multinomial_Naive_Bayes = MultinomialNB()

# Support Vector Algorithms
Linear_Support_Vector_Classifier = LinearSVC(random_state=0, tol=1e-5)
Support_Vector_Classifier = SVC()

# Decision Tree Algorithms
Decision_Tree_Classifier = DecisionTreeClassifier(random_state=0)
Extra_Tree_Classifier = BaggingClassifier(ExtraTreeClassifier(random_state=0),
                                          random_state=0)

# Simple Neural Network
Multi_layer_Perceptron_Classifier = MLPClassifier(random_state=1, max_iter=300)

# Ensemble Methods
Ada_Boost_Classifier = AdaBoostClassifier(
    base_estimator=RandomForestClassifier(),
    n_estimators=7,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=1)
Bagging_Classifier = BaggingClassifier(base_estimator=SVC(),
                                       n_estimators=10,
                                       random_state=0)
ExtraTrees_Classifier = ExtraTreesClassifier(n_estimators=100, random_state=0)
Histogram_based_Gradient_Boosting_Classifier = HistGradientBoostingClassifier()
Random_Forest_Classifier = RandomForestClassifier(max_depth=2, random_state=0)

Voting_Classifier = VotingClassifier(estimators=[
    ('lr', LogisticRegression(multi_class='multinomial', random_state=1)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=1)),
    ('gnb', GaussianNB())
],
                                     voting='hard')

models = {
    "Logistic_Regression": Logistic_Regression,
    "K_Neighbors": K_Neighbors_Classifier,
    "BernoulliNB": Bernoulli_Naive_Bayes,
    "ComplementNB": Complement_Naive_Bayes,
    "GaussianNB": Gaussian_Naive_Bayes,
    "MultinomialNB": Multinomial_Naive_Bayes,
    "Linear_Support_Vector": Linear_Support_Vector_Classifier,
    "Support_Vector": Support_Vector_Classifier,
    "Decision_Tree": Decision_Tree_Classifier,
    "Extra_Tree": Extra_Tree_Classifier,
    "Multi_layer_Perceptron": Multi_layer_Perceptron_Classifier,
    "Ada_Boost": Ada_Boost_Classifier,
    "Bagging": Bagging_Classifier,
    "ExtraTrees": ExtraTrees_Classifier,
    "Histogram_based_Gradient_Boosting":
    Histogram_based_Gradient_Boosting_Classifier,
    "Random_Forest": Random_Forest_Classifier,
    "Voting": Voting_Classifier,
}


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


def run(data):

    response = {}
    feature_selector = ["X-Axis", "Y-Axis", "Z-Axis"]
    target_selector = ["Activity"]
    split_size = 0.2
    weights_location = "./weights/"

    xTrain, xTest, yTrain, yTest = preprocess(data, feature_selector,
                                              target_selector, split_size)
    for model in tqdm(models):
        response[model] = pipeline(xTrain, xTest, yTrain, yTest, models[model],
                                   (weights_location + model + ".joblib"))

    return {
        k: v
        for k, v in sorted(response.items(), key=lambda item: item[1])
    }