from models.data_managers.reader import load_data

if __name__ == '__main__':
    dn8 = ['f8_1p', 'f8_1z', 'f8_2p', 'f8_2z', 'f8_3p', 'f8_3z']
    dn10 = ['f10_1p', 'f10_1z', 'f10_2p', 'f10_2z', 'f10_3p', 'f10_3z']

    (in_stat8, out_stat8) = load_data(room="F8", col_names=['reference__x', 'reference__y'], file_type="stat")
    (in_dynamic8, out_dynamic8) = load_data(room="F8", col_names=['reference__x', 'reference__y'], dynamic_names=dn8)

    (in_stat10, out_stat10) = load_data(room="F10", col_names=['reference__x', 'reference__y'], file_type="stat")
    (in_dynamic10, out_dynamic10) = load_data(room="F10", col_names=['reference__x', 'reference__y'], dynamic_names=dn10)
