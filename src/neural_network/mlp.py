from .base import Base
from pandas import DataFrame
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np


class MLP(Base):

    def __init__(self, network=None):
        self.network = self.create_network() if network is None else network

    def train(self, data: tuple) -> None:
        training_input_tensor = tf.convert_to_tensor((data[0].astype('float32')) / 10000)
        training_output_tensor = tf.convert_to_tensor((data[1].astype('float32')) / 10000)
        self.network.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0),
                             loss=tf.keras.losses.MeanSquaredError(),
                             metrics=['accuracy'])
        self.network.fit(training_input_tensor, training_output_tensor, epochs=5)

    def test(self, data: tuple) -> tuple:
        testing_input_tensor = tf.convert_to_tensor((data[0].astype('float32')) / 10000)
        testing_output_tensor = tf.convert_to_tensor((data[1].astype('float32')) / 10000)
        self.network.evaluate(testing_input_tensor, testing_output_tensor)
        weights = self.network.layers[0].get_weights()[0]
        result = self.network.predict(testing_input_tensor) * 10000
        mse = self.calculate_mean_squared_err(DataFrame(result), data[1])
        return mse, weights

    @staticmethod
    def create_network():
        network = tf.keras.models.Sequential()
        network.add(tf.keras.layers.Dense(128, activation='relu'))
        network.add(tf.keras.layers.Dense(64, activation='relu'))
        network.add(tf.keras.layers.Dense(32, activation='relu'))
        network.add(tf.keras.layers.Dense(16, activation='relu'))
        network.add(tf.keras.layers.Dense(8, activation='relu'))
        network.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        return network

    @staticmethod
    def calculate_mean_squared_err(result: DataFrame, test_output: DataFrame):
        mse = []
        for x, y in zip(result[0], result[1]):
            mse.append(mean_squared_error([x, y], [DataFrame(test_output['reference__x']).iloc[0],
                                                   DataFrame(test_output['reference__y']).iloc[0]]))
        return np.sqrt(mse)
