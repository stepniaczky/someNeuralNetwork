from glob import glob
import pandas as pd
from pandas import DataFrame
import tensorflow as tf
from src.neural_network.mlp import MLP
import re


def load_model(filename: str) -> MLP or None:
    try:
        return tf.keras.models.load_model(f"models/{filename}")
    except FileNotFoundError:
        print("File with given name does not exist!")
        return None


def load_data(col_names: list, file_type: str) -> tuple[DataFrame, DataFrame]:
    if file_type == 'dynamic':
        all_files = glob("data/F*/*.xlsx")
        files = [f for f in all_files if not (re.search("stat", f) or re.search("random", f))]
    else:
        files = glob("data/F*/*_stat_*.xlsx")
    data = [pd.read_excel(file, header=0, usecols=col_names) for file in files]
    data = pd.concat(data, ignore_index=True)
    data = data.dropna()
    measurement = pd.concat([data[col_names[0]], data[col_names[1]]], axis=1)
    reference = pd.concat([data[col_names[2]], data[col_names[3]]], axis=1)
    print(f"{file_type.capitalize()} data has been successfully loaded.")
    return measurement, reference
