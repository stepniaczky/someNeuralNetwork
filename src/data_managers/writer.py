import shutil
from pandas import DataFrame
from os import mkdir
from os.path import exists
import numpy as np
from matplotlib import pyplot as plt


def create_dir(first_dir: str, second_dir: str = None):
    if not exists(first_dir):
        mkdir(first_dir)
    if second_dir is not None:
        path = f"{first_dir}/{second_dir}"
        if exists(path):
            shutil.rmtree(path)
        mkdir(path)


def to_plot(dir_name: str, mse: np.array, meas: np.array):
    mse = np.sort(mse)
    for errors, label in zip([mse, meas], ["corrected", "measured"]):
        y = 1. * np.arange(len(errors)) / (len(errors) - 1)
        plt.plot(errors, y, label=label)
    plt.legend()
    plt.savefig(f"results/{dir_name}/distribution_error.jpg", dpi=500)
    plt.show()


def to_xlsx(dir_name: str, mse: np.array) -> None:
    DataFrame(mse).to_excel(f"results/{dir_name}/distribution_error.xlsx")


def save_results(dir_name: str, mse: DataFrame, meas) -> None:
    create_dir("results", dir_name)
    to_xlsx(dir_name, mse)
    to_plot(dir_name, mse, meas)


def save_model(neural_network, filename: str):
    create_dir("models")
    try:
        neural_network.network.save(f"models/{filename}")
    except FileExistsError:
        return "err"
