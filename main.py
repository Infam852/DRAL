import numpy as np

from settings import N_EVAL, NUM_CLASSES
from utils import load_mnist10, preprocess_data, split, get_most_uncertain
from datasets_manager import DatasetsManager
from models import CNNModel


def train_loop(train_model, sm, y_train):
    n_queries = 10
    n_oracle = 50
    # active learning
    for k in range(n_queries):
        score = train_model.evaluate(sm.storage_eval.x, sm.storage_eval.y)
        print(f"Number of samples used to trainig: {len(sm.storage_train)}")
        print(f"{k}. Test loss:", score[0])
        print("Test accuracy:", score[1])
        idxs = get_most_uncertain(train_model, sm.x_unl, n_oracle)
        labels = y_train[idxs]  # query an oracle
        # show_img(np.squeeze(sm.x_unl[idxs][0], axis=(2,)), labels[0])
        sm.label_samples(idxs, labels)
        y_train = np.delete(y_train, idxs, axis=0)
        train_model.fit(*sm.storage_train.get_xy())
        print(f"End of {k} query")
        print(sm)


if __name__ == "__main__":
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = load_mnist10()
    X_TRAIN = X_TRAIN[:20000]
    Y_TRAIN = Y_TRAIN[:20000]
    x_train = preprocess_data(X_TRAIN, data_type='x')
    y_train = preprocess_data(Y_TRAIN, data_type='y')
    (x_eval, y_eval), (x_train, y_train) = split(x_train, y_train, N_EVAL)
    x_test = preprocess_data(X_TEST, data_type='x')
    y_test = preprocess_data(Y_TEST, data_type='y')

    sm = DatasetsManager(x_train, x_eval, y_eval, x_test, y_test, NUM_CLASSES)
    train_model = CNNModel(sm.get_x_shape(), NUM_CLASSES, epochs=5)

    train_loop(train_model, sm, y_train)
    final_loss, final_acc = train_model.evaluate(*sm.storage_test.get_xy())
    print(f'Final loss: {final_loss}')
    print(f'Final accuracy: {final_acc}')
