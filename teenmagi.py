import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split

# KAGGLE SUBMISSION TEAM NAME: mostpalonen

def main():
    print(f"\nExecution started at {datetime.now()}")

    # Initial parameters
    input_shape = (8,8,1)
    num_classes = 1001

    # Load data
    x_tr, x_tst, y_tr, y_tst, x_val = loadData()

    # Define network 
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(16, 1, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes))

    # Compile the model
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
    model.summary()
    
    # Train the model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    model.fit(x_tr, y_tr, epochs=20, callbacks=[callback])

    # Do predictions for test and validation data
    model.evaluate(x_tst,  y_tst, verbose=2)
    y_val = model.predict(x_val)

    # Write CSV of predictions
    with open("data/teenmagi/NN_predictions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Class"])
        for i, item in enumerate(y_val):
            writer.writerow([i+1, np.argmax(item)])

    print(f"\nExecution done at {datetime.now()}")


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
    """

    x_tr = unpickle("data/teenmagi/training_x.dat")
    x_val = unpickle("data/teenmagi/validation_x.dat")
    y_tr = unpickle("data/teenmagi/training_y.dat")

    for i in range(len(x_tr)):
        x_tr[i] = x_tr[i][:,:,1]

    for i in range(len(x_val)):
        x_val[i] = x_val[i][:,:,1]
    
    x_tr, x_tst, y_tr, y_tst = train_test_split(x_tr, y_tr, test_size=0.2, random_state=42)
    x_tr, x_tst, y_tr, y_tst, x_val = np.array(x_tr, np.uint8), np.array(x_tst, np.uint8), np.array(y_tr, np.uint8), np.array(y_tst, np.uint8), np.array(x_val, np.uint8)
    return x_tr, x_tst, y_tr, y_tst, x_val

if __name__ == '__main__':
    main()