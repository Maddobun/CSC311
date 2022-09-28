import models.hm1_code as dtree
from sklearn.tree import DecisionTreeClassifier
from typing import List
import numpy as np

if __name__ == "__main__":
    train, val, test, feature_names = dtree.load_data()
    best = dtree.select_model(train, val, feature_names)
    dtree.test_model(best, test)

    # get the index of feature named trump
    top_5 = [("trump", 0.056),
             ("trumps", 0.162),
             ("hillary", 0.069),
             ("trump", 0.051),
             ("the", 0.325)]

    for entry in top_5:
        dtree.compute_information_gain(train[0].toarray(), train[1], entry, feature_names)

