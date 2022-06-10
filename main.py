from src.data_managers.reader import load_data, load_model
from src.data_managers.writer import save_results, save_model
from src.neural_network.mlp import MLP

if __name__ == '__main__':
    dn8 = ['f8/f8_1p', 'f8/f8_1z', 'f8/f8_2p', 'f8/f8_2z', 'f8/f8_3p', 'f8/f8_3z']
    dn10 = ['f10/f10_1p', 'f10/f10_1z', 'f10/f10_2p', 'f10/f10_2z', 'f10/f10_3p', 'f10/f10_3z']

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
        (in_stat, out_stat) = load_data(col_names=['data__coordinates__x', 'data__coordinates__y',
                                                   'reference__x', 'reference__y'], file_type="stat")

        model = MLP()
        print("Model training has started...")
        model.train((in_stat, out_stat))
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

        (in_dyn, out_dyn) = load_data(col_names=['data__coordinates__x', 'data__coordinates__y',
                                                 'reference__x', 'reference__y'], dynamic_names=dn8)

        model8 = MLP(model)
        (mse, weights) = model8.test((in_dyn, out_dyn))
        save_results(filename, mse, weights)
