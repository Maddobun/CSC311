from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from typing import List


def load_data() -> tuple:
    """
    parse text file to generate a list, preprocess it using sklearn feature extraction,
    and split it into 70% training, 15% validation, and 15% test sets
    """
    data = []
    labels = []

    path = ["data/clean_real.txt", "data/clean_fake.txt"]

    with open(path[0], "r") as f:
        for line in f:
            data.append(line)
            labels.append([1])

    with open(path[1], "r") as f:
        for line in f:
            data.append(line)
            labels.append([-1])

    data, labels = np.array(data), np.array(labels)

    vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(data)

    # split data
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    data_val, data_test, label_val, label_test = train_test_split(data_test, label_test, test_size=0.5, random_state=42)

    return [data_train, label_train], [data_val, label_val], [data_test, label_test]


def select_model(train: list, val: list) -> None:
    """
    Train decision tree models with different hyperparameters and print out the accuracy for each model
    """

    trees = []

    # hyperparameters
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    criterion = ["gini", "entropy", "log_loss"]

    # generate all possible combinations of hyperparameters
    parameters = product(max_depth, criterion)

    # train a model for each combination of hyperparameters
    for depth, crit in parameters:
        clf = DecisionTreeClassifier(max_depth=depth, criterion=crit)
        clf.fit(train[0], train[1])
        trees.append(clf)

    # print out the accuracy for each model
    for tree in trees:
        print(f"Accuracy for the model trained with max depth {tree.get_depth()} and criterion {tree.criterion}:"
              f" {tree.score(val[0], val[1])}")
        print()

    # plot the accuracy for each model as function of max depth
    plot_accuracy(trees, val)


def plot_accuracy(trees: List[DecisionTreeClassifier], val: list) -> None:
    """
    Plot the accuracy for each model as function of max depth.

    :param trees: list of decision tree models
    :param val: 2d array where the first column is the validation data and the second column is the labels
    """

    # generate a list of accuracies for each model
    accuracies = [tree.score(val[0], val[1]) for tree in trees]
    depths = [tree.get_depth() for tree in trees]

    # plot the accuracy for each model as function of max depth
    plt.plot(depths, accuracies)
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.title("Accuracy as a function of max depth")
    plt.show()




















