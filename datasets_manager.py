import numpy as np


class DatasetsManager:
    def __init__(self, x_unlabelled, x_eval, y_eval, x_test, y_test):
        """
        Samples passed should be preprocessed, especially their
        dimension should be expanded
        """
        self._validate(x_unlabelled, x_eval, y_eval, x_test, y_test)

        self.add_basic_storage('unl', x_unlabelled)
        self.add_storage('eval', x_eval, y_eval)
        self.add_storage('test', x_test, y_test)
        self.add_storage(
            'train', np.empty((0,) + self.test.get_feature_shape()),
            np.empty((0,) + self.test.get_label_shape())
        )

    def add_storage(self, name, x, y):
        setattr(self, name, Storage(x, y, name))

    def add_basic_storage(self, name, x):
        setattr(self, name, BasicStorage(x, name))

    def _validate(self, x_unlabelled, x_eval, y_eval, x_test, y_test):
        self._fail_if_len_mismatch(x_eval, y_eval, 'eval')
        self._fail_if_len_mismatch(x_test, y_test, 'test')
        self._fail_if_shape_mismatch(x_unlabelled, x_eval, x_test)

    def _fail_if_len_mismatch(self, x, y, name):
        if len(x) != len(y):
            raise Exception(f"Number of {name} features must equal \
                              number of {name} labels")

    def _fail_if_shape_mismatch(self, *args):
        it = iter(args)
        shape = next(it).shape[1:]
        if not all(arg.shape[1:] == shape for arg in args):
            raise ValueError(f"Not all args have shape: {shape}")

    def label_samples(self, idxs, labels):
        if len(idxs) != len(labels):
            raise Exception(
                f"""Length of indicies ({len(idxs)}) must be equal to the length of
                labels ({len(labels)})""")

        # !TODO maybe transaction method (from, to, idxs, **kwargs)?
        self.train.add(self.unl.get_x(idxs), labels)
        self.unl.remove(idxs)

    def get_x_shape(self):
        return self.x_shape

    def get_y_shape(self):
        return self.y_shape

    def __str__(self):
        msg = """
        Number of unlabelled samples: {}
        Number of labelled samples: {}
        Number of evaluation samples: {}
        Number of test samples: {}
        """.format(len(self.unl), len(self.train),
                   len(self.eval), len(self.test))
        return msg


class Storage:
    # maybe y can be None? - it saves memory for large datasets (unlabelled)
    def __init__(self, x, y, name):
        """ x, y should be numpy arrays """
        if len(x) != len(y):
            raise Exception(f'Number of x samples({len(x)}) does not match \
                            number of y samples({len(y)})')
        self._x = x
        self._y = y
        self.name = name

    def add(self, x_new, y_new):
        self._x = np.append(self._x, x_new, axis=0)
        self._y = np.append(self._y, y_new, axis=0)

    def remove(self, idxs):
        self._x = np.delete(self._x, idxs, axis=0)
        self._y = np.delete(self._y, idxs, axis=0)

    def get_x(self, idx=-1):
        return self._x[idx] if idx > -1 else self._x

    def get_y(self, idx=-1):
        return self._y[idx] if idx > -1 else self._y

    def get_xy(self, idx=None):
        return (self._x[idx], self._y[idx]) if idx > -1 \
            else (self._x, self._y)

    def get_x_shape(self):
        return self._x.shape

    def get_y_shape(self):
        return self._y.shape

    def get_feature_shape(self):
        return self._x.shape[1:]

    def get_label_shape(self):
        return self._y.shape[1:]

    def __len__(self):
        return len(self._x)

    def __str__(self):
        return f"Number of {self.name} samples in storage: {len(self)}"


class BasicStorage:
    """ Storage for one variable """

    def __init__(self, x, name):
        self._x = x
        self.name = name
        self._counter = 0

    def get_x(self, idx=-1):
        print('idx: ', idx)
        print(self._x[idx].shape if idx > -1 else self._x.shape)
        return self._x[idx] if idx > -1 else self._x

    def __next__(self):
        if self._counter < len(self._x):
            idx = self._counter
            self._counter += 1
            return idx
        else:
            raise StopIteration

    def __iter__(self):
        self._counter = 0
        return self

    def remove(self, idxs):
        self._x = np.delete(self._x, idxs, axis=0)

    def get_x_shape(self):
        return self._x.shape

    def get_feature_shape(self):
        return self._x.shape[1:]

    def __len__(self):
        return len(self._x)
