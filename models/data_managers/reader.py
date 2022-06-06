import glob
import os
import pandas as pd
from pandas import DataFrame


def read_excel(file, col_names) -> tuple[DataFrame, DataFrame]:
    input_data = pd.read_excel(file, sheet_name=0)
    for i, row in input_data.iterrows():
        if row['success'] is False:
            input_data.drop(i)

    input_data = input_data[col_names]
    return input_data, DataFrame(data=input_data)


def load_data(room: str, col_names: list, file_type: str = 'dynamic', dynamic_names: list = None) \
        -> tuple[DataFrame, DataFrame]:
    _input = DataFrame()
    _output = DataFrame()

    if file_type == 'dynamic':
        files = [f"data/{room}/{d}.xlsx" for d in dynamic_names]
    else:
        files = [os.path.normpath(i) for i in glob.glob(f"data/{room}/*{file_type}*.xlsx")]

    for file in files:
        (in_file, out_file) = read_excel(file, col_names)
        _input = pd.concat([_input, in_file])
        _output = pd.concat([_output, out_file])

    print(f"{file_type.capitalize()} data from room {room} has been successfully loaded.")
    return _input, _output
