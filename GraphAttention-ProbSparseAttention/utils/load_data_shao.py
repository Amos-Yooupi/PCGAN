import copy
import torch
import os
import matplotlib.pyplot as plt

RootFile = 'GIN_PAPER'
file_name = "model_ver_dis_rail_+"
file_path = os.path.join(RootFile, file_name)


class LoadDataShao(object):
    def __init__(self, text_path, num_train, label_only=False, freq=10):
        self.text_path = text_path
        self.num_train = num_train
        if not os.path.exists(text_path):
            print(text_path, "is not exist")
            exit()
        with open(text_path, encoding='utf-8') as f:
            data = f.readlines()
            self.data = [item.strip('\n') for item in data]
        # 节点数量
        self.node_num = int(data[1])
        # 时间步长
        self.num_t = int(data[0])
        # 每个时间步长图结构的长度
        self.per_data_length = self.node_num * 4 + 1
        # 由于每个文件只有label不一样，所以读取后续文件只需要保存label
        self.label_only = label_only
        # 采样频率
        self.freq = freq

    def get_labels(self, freq):
        # 读取label
        target = self.data[1:][::self.per_data_length]
        target = torch.tensor([float(item) for item in target[1:]])
        t = torch.arange(target.numel())
        plt.plot(t, target)
        plt.plot(t[::freq], target[::freq])
        plt.show()
        return target

    def get_graph_features(self):
        # 读取图结构
        all_graph = []
        for i in range(self.num_t):
            graph = self.data[2+i*self.per_data_length:2+(i+1)*self.per_data_length]
            graph_feature = torch.tensor([float(__) for _ in graph[1::4] for __ in _.split(' ')]).view(-1, 2)
            all_graph.append(graph_feature)
        graph_feature = torch.stack(all_graph, dim=0)
        return graph_feature

    def get_adj(self):
        # 读取邻接矩阵
        edge_idx = []
        for node_idx in range(self.node_num):
            edge_idx.extend([[node_idx, int(neighbor_idx)] for neighbor_idx in self.data[4 + node_idx * 4].split(' ')])
        temp = copy.copy(edge_idx)
        for connection in temp:
            if connection[::-1] not in edge_idx:
                edge_idx.append(connection[::-1])
        edge_idx = torch.tensor(edge_idx).T
        values = torch.ones(edge_idx.shape[-1])
        adj_matrix = torch.sparse_coo_tensor(edge_idx, values).to_dense() + torch.eye(self.node_num)
        return adj_matrix

    def save_to_tensor(self):
        # 加载label，graph_features, adj_matrix
        save_dir = os.path.join(RootFile, str(self.num_train))
        labels_name = f"labels_{'_'.join(file_name.split('_')[-4:-1])}"
        labels_path = os.path.join(save_dir, f"{labels_name}.pt")
        graph_path = os.path.join(save_dir, "graph_features.pt")
        adj_path = os.path.join(save_dir, "adj_matrix.pt")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        lables = self.get_labels(self.freq)
        torch.save(lables, labels_path)
        print(self.text_path, "读取成功！")
        print("列车数量:", self.num_train)
        print(f"{labels_name}:", lables.shape)
        if self.label_only:
            return
        else:
            graph_features = self.get_graph_features()
            adj_matrix = self.get_adj()
            torch.save(graph_features, graph_path)
            torch.save(adj_matrix, adj_path)
            print("graph_features:", graph_features.shape)
            print("adj_matrix:", adj_matrix.shape)


class LabelFusion(object):
    def __init__(self, num_train):
        self.num_train = num_train

    def fusion(self, fusion_label_name="label_acc"):
        labels_list = ["labels_lat_dis_bri", "labels_ver_dis_bri","labels_lat_dis_rail", "labels_ver_dis_rail"]
        labels_path = os.path.join(RootFile, str(self.num_train))
        save_labels_path = os.path.join(labels_path, f"{fusion_label_name}.pt")
        labels = []
        for label_name in labels_list:
            path = os.path.join(labels_path, label_name + ".pt")
            labels.append(torch.load(path))
        labels = torch.stack(labels, dim=-1)
        torch.save(labels, save_labels_path)
        print(labels_path, "保存成功！-----", labels.shape)


def fusion_all_label():
    label_list = ["label_acc.pt", "label_dis.pt"]
    n_train_list = [3, 4, 5, 6, 7]
    for n_train in n_train_list:
        path = os.path.join(RootFile, str(n_train))
        all_label_path = os.path.join(path, "labels.pt")
        all_label = []
        for label_name in label_list:
            label_path = os.path.join(path, label_name)
            patch_label = torch.load(label_path)
            all_label.append(patch_label)
        all_label = torch.cat(all_label, dim=-1)
        torch.save(all_label, all_label_path)
        print(f"{n_train}融合所有label{label_list} -> labels.pt")


def show_labels(labels, time_interval=2e-3):
    assert labels.shape.__len__() == 2
    num_t, D = labels.shape
    labels = labels.detach().cpu().clone()
    labels_list = ["labels_lat_acc_bri", "labels_ver_acc_bri", "labels_lat_acc_rail", "labels_ver_acc_rail",
                   "labels_lat_dis_bri", "labels_ver_dis_bri", "labels_lat_dis_rail", "labels_ver_dis_rail"]
    t = torch.arange(num_t) * time_interval
    fig, axes = plt.subplots(2, D//2)
    for i, axe in enumerate(axes.flatten()):
        axe.plot(t, labels[:, i])
        axe.set_title(labels_list[i], fontsize=20)
        axe.set_xlabel("Time(s)", fontsize=16)
        axe.set_ylabel("Response", fontsize=16)
    plt.show()


def show_predict(labels, predict, time_interval=2e-3):
    assert labels.shape.__len__() == 2
    assert predict.shape.__len__() == 2
    num_t, D = labels.shape
    labels = labels.detach().cpu().clone()
    labels_list = ["labels_lat_acc_bri", "labels_ver_acc_bri", "labels_lat_acc_rail", "labels_ver_acc_rail",
                   "labels_lat_dis_bri", "labels_ver_dis_bri", "labels_lat_dis_rail", "labels_ver_dis_rail"]
    t = torch.arange(num_t) * time_interval
    fig, axes = plt.subplots(2, D//2)
    for i, axe in enumerate(axes.flatten()):
        axe.plot(t, labels[:, i], label="True")
        axe.plot(t, predict[:, i], label="Predict")
        axe.set_title(labels_list[i], fontsize=20)
        axe.set_xlabel("Time(s)", fontsize=16)
        axe.set_ylabel("Response", fontsize=16)
        axe.xaxis.set_major_locator(plt.MaxNLocator(4))
        axe.yaxis.set_major_locator(plt.MaxNLocator(8))
        axe.tick_params(axis='both', which='major', labelsize=20)  # Increase label size
        axe.grid(True)
    plt.show()


if __name__ == '__main__':
    # 先读取成tensor格式数据
    # map_dict = {'Test_record1.txt': 7}
    # for i in range(4):
    #     map_dict[f"Train_record{i + 1}.txt"] = i+3
    # for text_name, num_train in map_dict.items():
    #     text_path = os.path.join(file_path, text_name)
    #     LoadDataShao(text_path, num_train, True).save_to_tensor()

    # 进行label融合, 一定要确保第一步运行了
    for num_train in range(3, 8):
        LabelFusion(num_train).fusion("label_dis")

    # 融合acc和dis
    fusion_all_label()





