import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image

def main():

    # Load and normalize data
    x_tr, x_tst, y_tr, y_tst = loadData()
    x_tr, x_tst = x_tr / 255.0, x_tst / 255.0
    input_shape = (64,64,3)

    # Task 1: Define network
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2, activation='sigmoid'))

    # Task 2: Compile and train the net
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer='SGD', loss=loss, metrics=['accuracy'])
    model.summary()
    model.fit(x_tr, y_tr, epochs=20, batch_size=8) # Batch size of 32 gives really bad results

    # Do predictions for test data
    model.evaluate(x_tst,  y_tst, verbose=2)

def loadData():
    """
    Load GTSRB dataset and split into training and testing sets
    """
    x = np.zeros(shape=(659, 64, 64, 3))  # Array for storing 660 64x64x3 images
    y = np.zeros(shape=(659))   # Ground truths
    y[450:] = int(1)

    # One-hot encode ground-truth labels
    y = tf.keras.utils.to_categorical(y)

    i = 0
    for c, r in enumerate([450, 209]):  # Iterate class 1 and 2
        for n_img in range(r):    # Iterate pics
            n_img_str = (3-len(str(n_img)))*"0" + str(n_img)  # Parse image name
            img = Image.open(f"./data/GTSRB_subset_2/class{c+1}/{n_img_str}.jpg", "r")
            pix_ar = np.array(img)
            x[i] = pix_ar
            i += 1
    return train_test_split(x, y, test_size=0.2, random_state=42)


if __name__ == '__main__':
    main()