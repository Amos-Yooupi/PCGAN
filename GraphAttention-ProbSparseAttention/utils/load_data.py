import os.path
import torch
from show_data import show_tensor_data
import json


RootDir = 'tensor_data'
UPDATE_ROOTDIR = 'Pydata_GNN'
SPAN = 3


def init_file_path(span):
    import_root_file = os.path.dirname(__file__)
    Span_dir = f'Span-{span}'
    ChooseDir = os.path.join(RootDir, Span_dir)
    ChooseDir = os.path.join(import_root_file, ChooseDir)

    EarthQuake = 'Earthquake.opt'
    Bridge = f'Span-{span}.opt'
    Train = f'Train.opt'

    FILE_PATH_LIST = [os.path.join(ChooseDir, path) for path in [Train, Bridge, EarthQuake]]
    print(FILE_PATH_LIST)
    return FILE_PATH_LIST


def init_file_path_update(span):
    import_root_file = os.path.dirname(__file__)
    Span_dir = f'{span}'
    ChooseDir = os.path.join(UPDATE_ROOTDIR, Span_dir)
    ChooseDir = os.path.join(import_root_file, ChooseDir)

    EarthQuake = 'Earthquake.pt'
    Bridge = f'Span-{span}.pt'
    Train = f'Train.pt'

    FILE_PATH_LIST = [os.path.join(ChooseDir, path) for path in [Train, Bridge, EarthQuake]]
    print(FILE_PATH_LIST)
    return FILE_PATH_LIST


def read_file(file_path, freq=1):
    if not os.path.exists(file_path):
        print(f"{file_path} is not exist")
        return None
    read_file_data = torch.load(file_path)
    if file_path.split('\\')[-1] == "Earthquake.opt":
        read_file_data = read_file_data.reshape(-1, 3, read_file_data.shape[-1])
    read_file_data = read_file_data[:, :, ::freq]
    print(f"Read {file_path}", ' '*5, f"File shape: {read_file_data.shape}")
    return read_file_data


__all__ = ["init_file_path", "read_file", "init_file_path_update"]

if __name__ == '__main__':
    # 通过发现，可以将采样率降低200倍，减轻数据压力
    # file_path_list = init_file_path(3)
    file_path_list = init_file_path_update(6)

    for file_path in file_path_list:
        read_data = read_file(file_path)
        if read_data is not None:
            show_tensor_data(read_data[0], freq=200, time_interval=5e-5, y_labels=["Acc x", "Acc y", "Acc z"], compare=False, title=["Acc x", "Acc y", "Acc z"])