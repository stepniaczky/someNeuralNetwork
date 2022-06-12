from src.data_managers.reader import load_model, load_data
from src.data_managers.writer import save_results, save_model
from src.neural_network.mlp import MLP

if __name__ == '__main__':

    choice: int = 0
    while choice not in [1, 2]:
        print("[1] - Training mode",
              "[2] - Testing mode", sep="\n")
        try:
            choice = int(input("Choice: "))
            if choice in [1, 2]:
                break
            print("You have to choose 1 or 2 option!")
        except ValueError:
            print("Invalid value!")

    if choice == 1:
        print("Data is getting loaded...")
        (measurement, reference) = load_data(col_names=['data__coordinates__x', 'data__coordinates__y',
                                                        'reference__x', 'reference__y'], file_type="stat")

        model = MLP()
        print("Model training has started...")
        model.train((measurement, reference))
        print("Saving the learned model...")
        filename = input("Enter a file name: ")
        while save_model(model, filename) == "err":
            filename = input("Model with given file name already exists!")

    if choice == 2:
        model = None
        filename = None
        while model is None:
            filename = input("Enter a file name of trained model: ")
            model = load_model(filename)

        print("Data is getting loaded...")
        (measurement, reference) = load_data(col_names=['data__coordinates__x', 'data__coordinates__y',
                                                        'reference__x', 'reference__y'], file_type="dynamic")

        model = MLP(model)
        (error_mlp, error_meas) = model.test((measurement, reference))
        save_results(filename, error_mlp, error_meas)
