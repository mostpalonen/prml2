import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


def loadData():
    with open("data/teenmagi/training_x.dat", 'rb') as pickleFile:
        x_tr = pickle.load(pickleFile)

    with open("data/teenmagi/validation_x.dat", 'rb') as pickeFile:
        x_val = pickle.load(pickeFile)

    with open("data/teenmagi/training_y.dat", 'rb') as pickeFile:
        y_tr = pickle.load(pickeFile)

    return x_tr, x_val, y_tr

def main():

    # Parameters
    n_neighbours = 1

    # Load data
    x_tr, x_val, y_tr = loadData()
    print(x_tr[1].shape)

    # Initialize nearest neighbour classifier and validate it
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbours, algorithm='kd_tree')
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_val)

main()
