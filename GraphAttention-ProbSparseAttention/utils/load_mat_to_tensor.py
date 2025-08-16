import os.path

import torch
import scipy.io
import matplotlib.pyplot as plt

ROOTFILE = 'Pydata_GNN'
span_list = [3, 4, 5, 6, 7, 8, 9]
pos_list = ["ori", "mid", "end"]
acc_dis_list = ["Acc", "Dis"]


def get_bridge(n_span):
    file_path = os.path.join(ROOTFILE, f"{n_span}跨数据")
    for pos in pos_list:
        for item in acc_dis_list:
            bridge_file_path = os.path.join(file_path, f"Bri{item}_{pos}_data.mat")
            if not os.path.exists(bridge_file_path):
                print("路径--",bridge_file_path, "不存在！！")
                exit()
            mat = scipy.io.loadmat(bridge_file_path)
            numpy_array = mat[f'Bri{item}_{pos}_data']  # 这里'data'是你要读取的变量名
            # 排列[Accx, Accx, Accx, Accy, Acc]
            tensor_data = torch.tensor(numpy_array)  # [L, n_spanx3]
            node_list = []
            # 读取每个节点的特征 -- 节点和桥跨越一样
            for i in range(n_span):
                node_list.append(tensor_data[:, i::n_span])
            bridge_data = torch.stack(node_list, dim=0)
            save_file_path = os.path.join(ROOTFILE, str(n_span))
            if not os.path.exists(save_file_path):
                os.mkdir(save_file_path)
            save_path = os.path.join(save_file_path, f"bri_{pos}_{item}.pt")
            torch.save(bridge_data, save_path)
            print(f"读取---桥梁---跨数---{n_span}---Bridge--{pos}---{item}--尺寸---{tensor_data.shape}---节点尺寸---{bridge_data.shape}", end='\n')
    print("---"*20)


def get_train(n_span):
    file_path = os.path.join(ROOTFILE, f"{n_span}跨数据")
    for item in acc_dis_list:
        train_file_path = os.path.join(file_path, f"Car{item}_data.mat")
        if not os.path.exists(train_file_path):
            print("路径--", train_file_path, "不存在！！")
            exit()
        mat = scipy.io.loadmat(train_file_path)
        numpy_array = mat[f'Car{item}_data']  # 这里'data'是你要读取的变量名
        # 排列[Accx, Accx, Accx, Accy, Acc]
        tensor_data = torch.tensor(numpy_array)  # [L, n_spanx3]
        node_list = []
        # 读取每个节点的特征 -- 节点和桥跨越一样
        for i in range(3):
            node_list.append(tensor_data[:, i::3])
        train_data = torch.stack(node_list, dim=0)
        save_file_path = os.path.join(ROOTFILE, str(n_span))
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        save_path = os.path.join(save_file_path, f"car_{item}.pt")
        torch.save(train_data, save_path)
        print(f"读取---列车---跨数---{n_span}---Train---{item}--尺寸---{tensor_data.shape}---节点尺寸---{train_data.shape}", end='\n')
    print("---"*20)


def get_earth(n_span):
    file_path = os.path.join(ROOTFILE, f"{n_span}跨数据")
    train_file_path = os.path.join(file_path, f"EqAcc_data.mat")
    if not os.path.exists(train_file_path):
        print("路径--", train_file_path, "不存在！！")
        exit()
    mat = scipy.io.loadmat(train_file_path)
    numpy_array = mat[f'EqAcc_data']  # 这里'data'是你要读取的变量名
    tensor_data = torch.tensor(numpy_array)  # [L, n_spanx3]
    node_list = []
    # 读取每个节点的特征 -- 节点和桥跨越一样
    for i in range(n_span + 1):
        node_list.append(tensor_data[:, i*3:(i+1)*3])
    eq_data = torch.stack(node_list, dim=0)
    save_file_path = os.path.join(ROOTFILE, str(n_span))
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)
    save_path = os.path.join(save_file_path, f"Eq.pt")
    torch.save(eq_data, save_path)
    print(f"读取---地震加速度---跨数---{n_span}---Earth--尺寸---{tensor_data.shape}---节点尺寸---{eq_data.shape}", end='\n')
    print("---" * 20, end='\n'*3)


def fusion_bridge(n_span):
    file_path = os.path.join(ROOTFILE, f"{n_span}")
    if not os.path.exists(file_path):
        print(f"{file_path}不存在，请先读取文件再融合！！！")
        exit()
    # 将相同位置的dis和acc融合
    bridge_pos_list = []
    for pos in pos_list:
        per_node_list = []
        for item in acc_dis_list:
            bridge_file_path = os.path.join(file_path, f"bri_{pos}_{item}.pt")
            per_node_list.append(torch.load(bridge_file_path))
        # 进行特征融合
        per_node = torch.cat(per_node_list, dim=-1)
        bridge_pos_list.append(per_node)
    # 融合位置是ori-mid-end-ori-mid-end的循环
    fusion_list = []
    for i in range(n_span):
        # 按照顺序取出来每个节点的位置
        fusion_list.extend([item[i, :, :] for item in bridge_pos_list])
    # 把时间维度放在最后
    fusion_tensor = torch.stack(fusion_list, dim=0).transpose(1, 2)
    save_file_path = os.path.join(file_path, f'Span-{n_span}.pt')
    torch.save(fusion_tensor, save_file_path)
    print(f"Bridge融合完毕---尺寸---{fusion_tensor.shape}")


def fusion_train(n_span):
    file_path = os.path.join(ROOTFILE, f"{n_span}")
    if not os.path.exists(file_path):
        print(f"{file_path}不存在，请先读取文件再融合！！！")
        exit()
    # 将相同位置的dis和acc融合
    fusion_list = []
    for item in acc_dis_list:
        train_file_path = os.path.join(file_path, f"car_{item}.pt")
        fusion_list.append(torch.load(train_file_path))
    # 进行特征融合
    fusion_tensor = torch.cat(fusion_list, dim=-1).transpose(1, 2)
    save_file_path = os.path.join(file_path, f'Train.pt')
    torch.save(fusion_tensor, save_file_path)
    print(f"Train融合完毕---尺寸---{fusion_tensor.shape}")


def fusion_eq(n_span):
    file_path = os.path.join(ROOTFILE, f"{n_span}")
    if not os.path.exists(file_path):
        print(f"{file_path}不存在，请先读取文件再融合！！！")
        exit()
    # 将相同位置的dis和acc融合
    fusion_list = []
    eq_file_path = os.path.join(file_path, f"Eq.pt")
    fusion_tensor = torch.load(eq_file_path).transpose(1, 2)
    # 进行特征融合
    save_file_path = os.path.join(file_path, f'Earthquake.pt')
    torch.save(fusion_tensor, save_file_path)
    print(f"EarthQuake融合完毕---尺寸---{fusion_tensor.shape}")


if __name__ == '__main__':
    # 读取数据
    # for n_span in span_list:
    #     print(f"读取桥跨数---||{n_span}||")
    #     get_train(n_span)
    #     get_bridge(n_span)
    #     get_earth(n_span)

    # 融合数据
    for n_span in span_list:
        print(f"融合桥跨数---||{n_span}||")
        fusion_bridge(n_span)
        fusion_train(n_span)
        fusion_eq(n_span)