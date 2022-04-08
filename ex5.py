import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image

def main():

    # Task1: Load and normalize data
    x_tr, x_tst, y_tr, y_tst = loadData()
    x_tr, x_tst = x_tr / 255.0, x_tst / 255.0

    # Task 2: Define network
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(4096, 3)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Task 3: Compile and train the net
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer='SGD', loss=loss, metrics=['accuracy'])
    model.summary()
    model.fit(x_tr, y_tr, epochs=10)

    # Do predictions for test data
    preds = model.predict(x_tst)
    for i, pred in enumerate(preds):
        if pred > 0.5:
            preds[i] = 1
        if pred <= 0.5:
            preds[i] = 0

    print(f"\nPrediction accuracy: {accuracy(y_tst, preds)}")
    model.evaluate(x_tst,  y_tst, verbose=2)

def loadData():
    """
    Task1: Load GTSRB dataset and split into training and testing sets
    """
    x = np.zeros(shape=(659, 4096, 3))  # Array for storing 660 64x64x3 images
    y = np.zeros(shape=(659))   # Ground truths
    y[450:] = 1

    i = 0
    for c, r in enumerate([450, 209]):  # Iterate class 1 and 2
        for n_img in range(r):    # Iterate pics
            n_img_str = (3-len(str(n_img)))*"0" + str(n_img)  # Parse image name
            img = Image.open(f"./data/GTSRB_subset_2/class{c+1}/{n_img_str}.jpg", "r")
            pix_val = list(img.getdata())
            pix_ar = np.array(pix_val)
            x[i] = pix_ar
            i += 1
    return train_test_split(x, y, test_size=0.2, random_state=69)

def accuracy(prediction, validation):
    correct = 0
    for i, predicted in enumerate(prediction):
        if predicted == validation[i]:
            correct += 1

    return correct/len(prediction)

if __name__ == '__main__':
    main()