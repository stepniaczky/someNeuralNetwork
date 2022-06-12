from .base import Base
from pandas import DataFrame
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
import math


class MLP(Base):

    def __init__(self, network=None):
        self.network = self.create_network() if network is None else network

    def train(self, data: tuple) -> None:
        measurement = (data[0].astype('float32')) / 10000
        reference = (data[1].astype('float32')) / 10000
        self.network.compile(optimizer=tf.keras.optimizers.Adam(),
                             loss=tf.keras.losses.MeanSquaredError(),
                             metrics=['accuracy'])
        self.network.fit(measurement, reference, epochs=100)

    def test(self, data: tuple) -> tuple:
        measurement = (data[0].astype('float32')) / 10000
        reference = (data[1].astype('float32')) / 10000
        weights = self.network.layers[0].get_weights()[0]
        res = self.network.predict(measurement)
        error_mlp = self.calculate_mean_squared_err(
            res, reference, choice="result")
        error_meas = self.calculate_mean_squared_err(measurement, reference)
        print("Mean square error of the measured values =",
              sum(error_meas) / len(error_meas))
        print("Mean square error of the corrected values =",
              sum(error_mlp) / len(error_mlp))
        print(
            "Arithmetic mean of the input weights [measurement / reference] =", np.mean(weights, axis=1))
        return error_mlp, error_meas

    @staticmethod
    def create_network():
        network = tf.keras.models.Sequential()
        network.add(tf.keras.layers.Dense(512, activation='relu'))
        network.add(tf.keras.layers.Dense(256, activation='relu'))
        network.add(tf.keras.layers.Dense(128, activation='relu'))
        network.add(tf.keras.layers.Dense(64, activation='relu'))
        network.add(tf.keras.layers.Dense(32, activation='relu'))
        network.add(tf.keras.layers.Dense(16, activation='relu'))
        network.add(tf.keras.layers.Dense(8, activation='relu'))
        network.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        return network

    @staticmethod
    def calculate_mean_squared_err(measurement: DataFrame, reference: DataFrame, choice=None):
        reference = reference.values.tolist()
        mse = []
        measurement = measurement.tolist() if choice == "result" else measurement.values.tolist()
        for i in range(len(measurement)):
            value = mean_squared_error([measurement[i][0], measurement[i][1]], [
                                       reference[i][0], reference[i][1]])
            mse.append(math.sqrt(value))
        return np.sort(mse) * 10000
