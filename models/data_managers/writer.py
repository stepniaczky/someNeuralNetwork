from pandas import DataFrame


def save_results(file_name: str, data: DataFrame) -> None:
    data.to_excel(file_name)
