import os.path
import torch
from torch.utils.data import DataLoader, Dataset
from load_data_shao import show_labels, show_predict


class CustomerDataSet(Dataset):
    def __init__(self, graph_features, dis, labels, enc_length, dec_length, proportion=1., data_mode='simple'):
        # 图结构数据
        self.graph_features = graph_features
        self.dis = dis
        self.labels = labels
        assert graph_features.shape.__len__() == 3, print("[num_t, node_num, D]")
        assert labels.shape.__len__() == 2, print("[num_t, 9]")

        # enc_length代表输入窗口的尺寸，dec_length用于取出label
        self.enc_length = enc_length
        self.dec_length = dec_length

        # data length
        self.proportion = proportion

        # 训练模式 -> simple的stride为dec_length complex的stride为1
        self.data_mode = data_mode
        if self.data_mode == "simple":
            self.stride = self.dec_length
            self.length = (int(self.graph_features.shape[0] * self.proportion) - self.enc_length) // self.dec_length
        elif self.data_mode == "complex":
            self.stride = 1
            self.length = int(self.graph_features.shape[0] * self.proportion) - self.enc_length - self.dec_length + 1
        else:
            raise Exception("data mode only can be simple or complex")

    def __getitem__(self, item):
        enc_start = item * self.stride
        enc_tail = enc_start + self.enc_length
        dec_tail = enc_tail + self.dec_length

        return self.graph_features[enc_start: enc_tail, :, :], self.dis[enc_start: enc_tail, :], self.labels[enc_tail: dec_tail, :]

    def __len__(self):
        return self.length


class DataIter(object):
    def __init__(self, n_train, enc_length, dec_length, batch_size, device=torch.device('cpu'), show=True):
        file_path = os.path.join("GIN_PAPER", str(n_train))
        if __name__ == 'main':
            pass
        else:
            root_file = os.path.dirname(__file__)
            file_path = os.path.join(root_file, file_path)
        self.graph_features = torch.load(os.path.join(file_path, "graph_features.pt")).to(device)
        self.labels = torch.load(os.path.join(file_path, "labels.pt")).to(device)[400:1200, :]
        self.dis = self.labels.clone()[:, 4:]
        self.adj = torch.load(os.path.join(file_path, "adj_matrix.pt")).to(device)

        self.n_train = n_train

        self.enc_length, self.dec_length, self.batch_size = enc_length, dec_length, batch_size

        if show:
            show_labels(self.labels)

    def data_iter(self, proportion=0.8, shuffle=True, batch_size=None, data_mode="simple"):
        if batch_size is None:
            batch_size = self.batch_size
        dataset = CustomerDataSet(self.graph_features, self.dis, self.labels, self.enc_length, self.dec_length, proportion, data_mode=data_mode)

        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def predict(self, net, show_point=False, show=True, save=True):
        net.eval()
        predict = []
        true = []
        with torch.no_grad():
            for graph_features, dis, labels in self.data_iter(proportion=1, shuffle=False):
                predict_ = net(graph_features, dis, self.adj)
                predict.append(predict_.detach().cpu())
                true.append(labels.detach().cpu())

        # (L, D)
        predict = torch.cat(predict, dim=0).squeeze()
        true = torch.cat(true, dim=0).squeeze()
        mae_loss = torch.mean(torch.abs(predict - true), dim=0).sum()
        print(f"All MAE Loss: {mae_loss}")

        # if save:
        #     predict_data_name = f'{net}-{self.n_train}.pt'
        #     true_data_name = f'true-{self.n_train}.pt'
        #     predict_data_path = os.path.join("predict_data", predict_data_name)
        #     true_data_path = os.path.join("predict_data", true_data_name)
        #     torch.save(predict, predict_data_path)
        #     torch.save(true, true_data_path)
        if show:
            show_predict(true, predict)


class DataIterMixer(object):
    # 混合训练模式
    def __init__(self, n_bridge_list, enc_length, dec_length, batch_size, device=torch.device('cpu'), show=True):
        assert isinstance(n_bridge_list, list), print("n span list should be a list")
        self.data_iter_list = [DataIter(n_span, enc_length, dec_length, batch_size, device=device, show=show) for n_span in n_bridge_list]
        self.idx = -1

    def choose_data_iter(self) -> DataIter:
        self.idx += 1
        return self.data_iter_list[self.idx % self.data_iter_list.__len__()]


if __name__ == '__main__':
    count = 0
    data_iter = DataIter(3, 36, 1, 64)
    for graph_features, dis, label in data_iter.data_iter(data_mode="simple", proportion=0.8):
        print("Graph_features shape: ", graph_features.shape)
        print("Dis shape: ", dis.shape)
        print("Label shape: ", label.shape)
        count += 1
    print("Total Batch: ", count)


