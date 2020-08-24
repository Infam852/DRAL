from tensorflow import keras


class CNNModel:
    def __init__(self,
                 input_shape,
                 num_classes,
                 epochs,
                 batch_size=128,):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self._model = None
        self._create_model()

    def _create_model(self):
        model = keras.models.Sequential([
            keras.Input(shape=self.input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        self._model = model

    def fit(self, x, y):
        self._model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, x):
        return self._model.predict(x, batch_size=self.batch_size)

    def evaluate(self, x, y):
        """ Returns a tuple (loss, accuracy) """
        return self._model.evaluate(x, y)


class AuxModel:
    def __init__(self, model, layer_idx):
        self._model = keras.Model(
            inputs=model.inputs, outputs=model.outputs +
            [model.layers[layer_idx].output])

    def predict(self, x):
        """ Returns (final_output, intermediate_layer_output) """
        return self._model.predict(x)
