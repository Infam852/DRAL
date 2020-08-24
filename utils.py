import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

from settings import NUM_CLASSES


def load_mnist10():
    """ Returns (x_train, y_train), (x_test, y_test) """
    return keras.datasets.mnist.load_data()


def split(x, y, split_idx):
    if len(x) != len(y):
        raise Exception(f'Both x({len(x)}), y({len(y)}) should have \
                        equal length')
    x1, x2 = np.split(x, (split_idx, ))
    y1, y2 = np.split(y, (split_idx, ))
    return (x1, y1), (x2, y2)


def show_img(img_arr, label=None):
    plt.imshow(img_arr, cmap='gray')
    plt.title(f"label: {label}" if label is not None else "")
    plt.axis("off")
    plt.show()


def preprocess_data(data, data_type):
    if data_type == 'x':
        data = data.astype("float32") / 255.0
        data = np.expand_dims(data, -1)
    elif data_type == 'y':
        data = keras.utils.to_categorical(data, NUM_CLASSES)
    else:
        raise Exception(f'Wrong data_type: {data_type}')
    return data


def get_most_uncertain(model, x_arr, k):
    predictions = model.predict(x_arr)
    # print('predictions: {}'.format(predictions))
    probs = tf.nn.softmax(predictions).numpy()
    diffs = np.array([])
    for p in probs:
        p.sort()
        diffs = np.append(diffs, abs(p[-1] - p[-2]))

    smallest_margins_idx = diffs.argsort()[:k]
    # show_img(np.squeeze(sm.x_unl[smallest_margins_idx[0]], axis=(2,)), smallest_margins_idx[0])
    return smallest_margins_idx
