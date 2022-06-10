from pandas import DataFrame
from os import mkdir, rmdir
from os.path import exists


def create_dir(first_dir: str, second_dir: str = None):
    if not exists(first_dir):
        mkdir(first_dir)
    if second_dir is not None:
        path = f"{first_dir}/{second_dir}"
        if exists(path):
            rmdir(path)
        else:
            mkdir(path)


def to_plot(dir_name: str, mse: DataFrame, weights: DataFrame):
    ...


def to_xlsx(dir_name: str, mse: DataFrame) -> None:
    mse.to_excel(f"results/{dir_name}/error_distribution.xlsx")


def save_results(dir_name: str, mse: DataFrame, weights: DataFrame) -> None:
    create_dir("results", dir_name)
    to_xlsx(dir_name, mse)
    to_plot(dir_name, mse, weights)


def save_model(neural_network, filename: str):
    create_dir("models")
    try:
        neural_network.network.save(f"models/{filename}")
    except FileExistsError:
        return "err"
