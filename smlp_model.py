from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


class SimpleMLP:
    """
    A Simple Multilayer Perceptron Model
    """

    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
