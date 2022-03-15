import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import csv
from datetime import datetime


# KAGGLE SUBMISSION TEAM NAME: mostpalonen
#
# Takes advantage of gridsearch and cross validation to
# optimize the n_neighbors. For my training, n_neighbors=9
# leads to the best results.

def unpickle(file):
    """
    Helper function that loads a given file from path
    """
    with open(file, 'rb') as pickleFile:
        data = pickle.load(pickleFile)
    return data


def loadData():
    """
    Load input data and transform the data from size (8,8,3) to (8,8,1).
    Transformation reduces the complexity because colorchannels are duplicant. 
    After this the input data is transformed to one dimensional (64,1) vectors.
    """

    x_tr = unpickle("data/teenmagi/training_x.dat")
    x_val = unpickle("data/teenmagi/validation_x.dat")
    y_tr = unpickle("data/teenmagi/training_y.dat")

    for i in range(len(x_tr)):
        x_tr[i] = x_tr[i][:,:,1].flatten()

    for i in range(len(x_val)):
        x_val[i] = x_val[i][:,:,1].flatten()

    return x_tr, x_val, y_tr


def main():
    print(f"\nExecution started at {datetime.now()}")

    # Dictionary of all the values used for finding optimal value for n_neighbors
    param_grid = {'n_neighbors': np.arange(3, 5)}

    # Load data
    x_tr, x_val, y_tr = loadData()
    print(f"Data loaded")

    # Initialize nearest neighbour classifier
    clf = neighbors.KNeighborsClassifier(n_jobs=6)

    # Use gridsearch to test all values for n_neighbors
    clf_gscv = GridSearchCV(clf, param_grid, cv=3, verbose=2)

    # Fit model to data
    clf_gscv.fit(x_tr, y_tr)
    print(f"Fit done")

    # Do predictions with the best performing estimator
    y_pred = clf_gscv.best_estimator_.predict(x_val)
    print(f"Predictions done")

    # Write CSV of predictions
    with open("data/teenmagi/predictions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Class"])
        i = 1
        for item in y_pred:
            writer.writerow([i, item])
            i += 1
    print(f"CSV written \n")

main()
