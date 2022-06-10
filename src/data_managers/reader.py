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
    _input = DataFrame()
    _output = DataFrame()

    if file_type == 'dynamic':
        files = [f"data/{d}.xlsx" for d in dynamic_names]
    else:
        files = [normpath(i) for i in glob(f"data/*/*{file_type}*.xlsx")]

    for file in files:
        (in_file, out_file) = read_excel(file, col_names)
        _input = pd.concat([_input, in_file])
        _output = pd.concat([_output, out_file])

    print(f"{file_type.capitalize()} data has been successfully loaded.")
    return _input, _output


def read_excel(file, col_names) -> tuple[DataFrame, DataFrame]:
    data = pd.read_excel(file, sheet_name=0)
    to_drop = []
    for i, row in data.iterrows():
        if row['success'] is False:
            to_drop.append(i)

    data = data.drop(data.index[to_drop])
    return data[col_names[:2]], data[col_names[2:]]
