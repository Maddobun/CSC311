import scipy.sparse
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
            labels.append(["real"])

    with open(path[1], "r") as f:
        for line in f:
            data.append(line)
            labels.append(["fake"])

    data, labels = np.array(data), np.array(labels)

    vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(data)

    # split data
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    data_val, data_test, label_val, label_test = train_test_split(data_test, label_test, test_size=0.5, random_state=42)

    return (data_train, label_train), (data_val, label_val), (data_test, label_test), vectorizer.get_feature_names_out()


def select_model(train: list, val: list, feature_names: list) -> DecisionTreeClassifier:
    """
    Train decision tree models with different hyperparameters and print out the accuracy for each model
    """

    trees = []

    # hyperparameters
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    criterion = ["gini", "entropy", "log_loss"]

    # train a model for each combination of hyperparameters
    for crit in criterion:
        crit_trees = []
        for depth in max_depth:
            clf = DecisionTreeClassifier(max_depth=depth, criterion=crit, random_state=42)
            clf.fit(train[0], train[1])
            crit_trees.append((clf, clf.score(val[0], val[1])))
        trees.append(crit_trees)

    # print out accuracy
    for i, crit in enumerate(trees):
        for j, tree in enumerate(crit):
            print(f"Accuracy for {criterion[i]} with max_depth {max_depth[j]}: {tree[1]}")

    # print the hyperparameters for the model with the highest accuracy
    best = max(max(trees, key=lambda x: max(x, key=lambda y: y[1])[1]), key=lambda x: x[1])
    print(f"Best model: {best[0].criterion} with max_depth {best[0].max_depth}")

    # plot the accuracy for each model as function of max depth
    plot_accuracy(trees)

    # visualize the decision tree with the highest accuracy
    visualize_tree(best[0], feature_names)

    # for testing
    return best[0]


def plot_accuracy(trees: list) -> None:
    """
    Plot the accuracy for each model as function of max depth.

    :param trees: list of decision tree models
    """

    # plot one line for each criterion
    # max_depth starts at 1, so we need to add 1 to the index
    for i, crit in enumerate(trees):
        plt.plot(range(1, len(crit)+1), [x[1] for x in crit], label=crit[0][0].criterion)

    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def visualize_tree(tree: DecisionTreeClassifier, feature_names: list) -> None:
    """
    Visualize the decision tree with the highest accuracy.

    :param tree: decision tree model
    :param feature_names: list of feature names
    """

    from sklearn.tree import export_graphviz
    import graphviz

    dot_data = export_graphviz(tree, out_file=None,
                               feature_names=feature_names,
                               class_names=["real", "fake"],
                               max_depth=2)
    graph = graphviz.Source(dot_data)
    graph.render("tree")


def test_model(tree: DecisionTreeClassifier, test: tuple) -> None:
    """
    Test the model on the test set and print out the accuracy.

    :param tree: decision tree model
    :param test: test set
    """

    print(f"Accuracy on test set: {tree.score(test[0], test[1])}")


def compute_information_gain(train_data: np.ndarray,
                             train_label: np.ndarray,
                             feature: (str, float),
                             feature_name_list) -> None:
    """
    Compute the information gain on training set for the given feature

    :param train_data: training set
    :param train_label: training labels
    :param feature: the feature to split, given as the name of the feature
    :param feature_name_list: list of feature names
    """

    # get the column of the feature
    feature_index = np.where(feature_name_list == feature[0])[0]
    train_label = np.where(train_label == "real", 1.0, 0.0)

    # merge the training set and training labels
    data = np.c_[train_data, train_label]

    # count the number of real and fake news in the training set
    real_count = np.count_nonzero(data[:, -1] == 1)
    fake_count = np.count_nonzero(data[:, -1] == 0)

    # compute initial entropy
    initial_entropy = - (real_count / data.shape[0]) * np.log2(real_count / data.shape[0]) - (
            fake_count / data.shape[0]) * np.log2(fake_count / data.shape[0])

    # split the data based on the given feature and its value
    cond = data[:, feature_index] < feature[1]
    left_index, right_index = np.where(cond), np.where(~cond)
    left, right = data[left_index[0]], data[right_index[0]]

    # count the number of real and fake news in the left and right split
    left_real_count = np.count_nonzero(left[:, -1] == 1)
    left_fake_count = np.count_nonzero(left[:, -1] == 0)
    right_real_count = np.count_nonzero(right[:, -1] == 1)
    right_fake_count = np.count_nonzero(right[:, -1] == 0)

    left_entropy = - left_real_count / left.shape[0] * np.log2(left_real_count / left.shape[0]) \
                   - left_fake_count / left.shape[0] * np.log2(left_fake_count / left.shape[0])
    right_entropy = - right_real_count / right.shape[0] * np.log2(right_real_count / right.shape[0]) \
                    - right_fake_count / right.shape[0] * np.log2(right_fake_count / right.shape[0])

    expected_conditional_entropy = left.shape[0] / data.shape[0] * left_entropy + \
                                   right.shape[0] / data.shape[0] * right_entropy

    # compute information gain
    information_gain = initial_entropy - expected_conditional_entropy

    print(f"Information gain for feature {feature[0]} with threshold {feature[1]}: {information_gain}")


if __name__ == "__main__":
    train, val, test, feature_names = load_data()
    best = select_model(train, val, feature_names)
    test_model(best, test)

    # get the index of feature named trump
    top_5 = [("trump", 0.056),
             ("trumps", 0.162),
             ("hillary", 0.069),
             ("trump", 0.051),
             ("the", 0.325)]

    for entry in top_5:
        compute_information_gain(train[0].toarray(), train[1], entry, feature_names)
