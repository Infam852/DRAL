import numpy as np


class DatasetsManager:
    def __init__(self, x_unlabelled, x_eval, y_eval, x_test, y_test, ncls):
        """
        Samples passed should be preprocessed, especially their
        dimension should be expanded
        """
        if len(x_test) != len(y_test):
            raise Exception("Number of test features must equal \
                             number of test labels")

        # e.g. (28,28,1) important when append to an empty list
        self.x_shape = x_unlabelled.shape[1:]
        self.y_shape = y_eval.shape

        self.x_unl = x_unlabelled
        self.add_storage('storage_train', np.empty((0,) + self.x_shape),
                         np.empty((0, ncls)))
        self.add_storage('storage_eval', x_eval, y_eval)
        self.add_storage('storage_test', x_test, y_test)

    def add_storage(self, name, x, y):
        setattr(self, name, Storage(x, y))

    def label_samples(self, idxs, labels):
        if len(idxs) != len(labels):
            raise Exception(f"Length of x_idxs ({len(idxs)}) must be equal to the \
                              length of labels ({len(labels)})")

        self.storage_train.add_samples(self.x_unl[idxs], labels)
        self.x_unl = np.delete(self.x_unl, idxs, axis=0)

    def get_x_shape(self):
        return self.x_shape

    def get_y_shape(self):
        return self.y_shape

    def get_x_unl(self, n):
        return self.x_unl[n]

    def __str__(self):
        msg = """
        Number of unlabelled samples: {}
        Number of labelled samples: {}
        Number of evaluation samples: {}
        Number of test samples: {}
        """.format(len(self.x_unl), len(self.storage_train),
                   len(self.storage_eval), len(self.storage_test))
        return msg


class Storage:
    # maybe y can be None? - it saves memory for large datasets (unlabelled)
    def __init__(self, x, y=np.array([])):
        """ x, y should be numpy arrays """
        if len(y) and len(x) != len(y):
            raise Exception(f'Number of x samples({len(x)}) does not match \
                            number of y samples({len(y)})')

        self.x = x
        self.y = y

    def add_samples(self, x_new, y_new):
        self.x = np.append(self.x, x_new, axis=0)
        self.y = np.append(self.y, y_new, axis=0)

    def remove_samples(self, idxs):
        self.x = np.delete(self.x, idxs, axis=0)
        self.y = np.delete(self.y, idxs, axis=0)

    def get_xy(self):
        return self.x, self.y

    def get_xy_idxs(self, idxs):
        return self.x[idxs], self.y[idxs]

    def __len__(self):
        return len(self.x)
