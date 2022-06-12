from glob import glob
from os.path import normpath
import pandas as pd
from pandas import DataFrame
import tensorflow as tf
from src.neural_network.mlp import MLP


def load_model(filename: str) -> MLP or None:
    try:
        return tf.keras.models.load_model(f"models/{filename}")
    except FileNotFoundError:
        print("File with given name does not exist!")
        return None


def load_data(col_names: list, file_type: str = 'dynamic', dynamic_names: list = None) \
        -> tuple[DataFrame, DataFrame]:
    measurement = DataFrame()
    reference = DataFrame()

    if file_type == 'dynamic':
        files = [f"data/{d}.xlsx" for d in dynamic_names]
    else:
        files = [normpath(i) for i in glob(f"data/*/*{file_type}*.xlsx")]

    for file in files:
        (m_file, r_file) = read_excel(file, col_names)
        measurement = pd.concat([measurement, m_file])
        reference = pd.concat([reference, r_file])

    print(f"{file_type.capitalize()} data has been successfully loaded.")
    return measurement, reference


def read_excel(file, col_names) -> tuple[DataFrame, DataFrame]:
    data = [pd.read_excel(file, sheet_name=0)]
    data = pd.concat(data, ignore_index=True)
    data = data.dropna()
    return data[col_names[:2]], data[col_names[2:]]
