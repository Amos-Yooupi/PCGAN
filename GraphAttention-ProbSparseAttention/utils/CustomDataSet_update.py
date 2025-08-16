import os.path
from show_data import show_tensor_data, show_predict, show_sample_predict
import torch
from torch.utils.data import DataLoader, Dataset
from load_data import *


class CustomerDataSet(Dataset):
    def __init__(self, train_data, bridge_data, earth_data, enc_length, dec_length, proportion=1., data_mode='simple'):
        # 读取列车和地震加速度的数据
        self.train_data = train_data
        self.earth_data = earth_data
        self.bridge_data = bridge_data

        # enc_length代表输入窗口的尺寸，dec_length用于取出label
        self.enc_length = enc_length
        self.dec_length = dec_length

        # data length
        self.proportion = proportion

        # 训练模式 -> simple的stride为dec_length complex的stride为1
        self.data_mode = data_mode
        if self.data_mode == "simple":
            self.stride = self.dec_length
            self.length = (int(self.train_data.shape[-1] * self.proportion) - self.enc_length) // self.dec_length
        elif self.data_mode == "complex":
            self.stride = 1
            self.length = int(self.train_data.shape[-1] * self.proportion) - self.enc_length - self.dec_length + 1
        else:
            raise Exception("data mode only can be simple or complex")

    def __getitem__(self, item):
        enc_start = item * self.stride
        enc_tail = enc_start + self.enc_length
        dec_tail = enc_tail + self.dec_length

        # train和bridge和earthquake作为编码输入，train数据作为预测的值
        return self.train_data[:, :, enc_start: enc_tail].permute(2, 0, 1), self.bridge_data[:, :, enc_start: enc_tail].permute(2, 0, 1), self.earth_data[:, :, enc_start: enc_tail].permute(2, 0, 1), self.train_data[:, :, enc_tail: dec_tail].permute(2, 0, 1)

    def __len__(self):
        return self.length


class DataIter(object):
    def __init__(self, n_span, enc_length, dec_length, batch_size, freq=100, device=torch.device('cpu'), show=True, k=4):
        train_path_list = init_file_path(n_span)
        test_path_list = init_file_path_update(n_span)
        # 训练集
        self.train_data = read_file(train_path_list[0], freq=freq).to(device).to(torch.float)
        self.bridge_data = read_file(train_path_list[1], freq=freq).to(device).to(torch.float)
        self.earth_data = read_file(train_path_list[-1], freq=freq).to(device).to(torch.float)
        # 测试集
        if read_file(test_path_list[0], freq=freq) is not None:
            self.train_data_test = read_file(test_path_list[0], freq=freq).to(device).to(torch.float)
            self.bridge_data_test = read_file(test_path_list[1], freq=freq).to(device).to(torch.float)
            self.earth_data_test = read_file(test_path_list[-1], freq=freq).to(device).to(torch.float)
        else:
            self.train_data_test = None
            self.bridge_data_test = None
            self.earth_data_test = None

        self.freq = freq

        self.n_span = n_span

        self.enc_length, self.dec_length, self.batch_size = enc_length, dec_length, batch_size
        # 进行归一化  --> 归一化的参数是训练集的
        self.train_mean, self.train_std, self.bridge_mean, self.bridge_std, self.earth_mean, self.earth_std = self.get_norm_parameters()
        self.train_data, self.bridge_data, self.earth_data = self.norm_data(self.train_data, self.bridge_data, self.earth_data)
        if self.train_data_test is not None:
            self.train_data_test, self.bridge_data_test, self.earth_data_test = self.norm_data(self.train_data_test, self.bridge_data_test, self.earth_data_test)

        self.device = device

        self.adj_mat = self.adj()

        self.n_span = n_span

        self.earth_freq = self.calculate_k_freq(k)

        # # 桥梁数据自己学习或者加载
        self.bridge_data = torch.zeros_like(self.bridge_data)

        # 读取数据显示数据
        if show:
            show_tensor_data(self.train_data[0])

    def get_norm_parameters(self):
        train_mean = torch.mean(self.train_data, dim=-1, keepdim=True)
        train_std = torch.std(self.train_data, dim=-1, keepdim=True)

        earth_mean = torch.mean(self.earth_data, dim=-1, keepdim=True)
        earth_std = torch.std(self.earth_data, dim=-1, keepdim=True)

        bridge_mean = torch.mean(self.bridge_data, dim=-1, keepdim=True)
        bridge_std = torch.std(self.bridge_data, dim=-1, keepdim=True)

        return train_mean, train_std, bridge_mean, bridge_std, earth_mean, earth_std

    def norm_data(self, train_data, bridge_data, earth_data):
        # 归一化
        train_data = (train_data - self.train_mean) / self.train_std
        bridge_data = (bridge_data - self.bridge_mean) / self.bridge_std
        earth_data = (earth_data - self.earth_mean) / self.earth_std
        return train_data, bridge_data, earth_data

    def re_norm(self, data):
        return data * self.train_std.to(data.device) + self.train_mean.to(data.device)

    def data_iter(self, proportion=0.8, shuffle=True, batch_size=None, data_mode="simple", data='train'):
        if batch_size is None:
            batch_size = self.batch_size
        if data == 'train':
            dataset = CustomerDataSet(self.train_data, self.bridge_data, self.earth_data, self.enc_length, self.dec_length, proportion, data_mode=data_mode)
        elif data == 'test':
            # 没有数据
            if self.train_data_test is None:
                print("Test没有数据")
                return [[["None"] * 4]]
            dataset = CustomerDataSet(self.train_data_test, self.bridge_data_test, self.earth_data_test, self.enc_length, self.dec_length, proportion, data_mode=data_mode)
        elif data == "all":
            return [*self.data_iter(proportion, shuffle, batch_size, data_mode, data="train"), *self.data_iter(proportion, shuffle, batch_size, data_mode, data="test")]
        else:
            print("data mode only can be train or test")
            exit()

        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def adj(self):
        n_train = self.train_data.shape[0]
        n_bridge = self.n_span * 3
        n_pier = n_bridge // 3 + 1
        train_idx = list(range(0, n_train))
        bridge_idx = list(range(n_train, n_bridge+n_train))
        pier_idx = list(range(n_bridge+n_train, n_bridge+n_train+n_pier))
        adj = []
        # 列车自己连接
        for idx in train_idx[:-1]:
            adj.append([idx, idx+1])
        # 列车和桥梁相互连接
        for r in train_idx:
            for c in bridge_idx:
                adj.append([r, c])
        # 桥梁自己连接
        for idx in bridge_idx[:-1]:
            adj.append([idx, idx+1])
        # 桥梁和桥墩连接
        # 除了两端的进行连接
        for n in range(1, n_pier-1):
            adj.extend([[pier_idx[n], b_idx] for b_idx in bridge_idx[(n-1)*3:(n+1)*3]])
        # 两端特殊情况进行连接
        n_total = pier_idx[-1] + 1
        adj.extend([[pier_idx[0], b_idx] for b_idx in bridge_idx[:3]])
        adj.extend([[pier_idx[-1], b_idx] for b_idx in bridge_idx[-3:]])
        values = torch.ones(len(adj))
        adj = torch.tensor(adj).T
        adj = torch.sparse_coo_tensor(adj, values, (n_total, n_total)).to_dense()
        adj = adj + adj.T.multiply(adj.T>adj) + torch.eye(n_total)
        return adj.to(self.device)

    def calculate_k_freq(self, k):
        _, _, L = self.earth_data.shape
        signal = self.earth_data.detach().clone().contiguous().view(-1, L)
        magnitude = torch.abs(torch.fft.fft(signal)).mean(dim=0)[:L//2]
        values, freq = torch.topk(magnitude, k)
        return freq

    def predict(self, net, show_point=False, show=True, save=True, data='train'):
        net.eval()
        predict = []
        true = []
        if data == 'test' and self.train_data_test is None:
            print("Test 没有数据")
            return
        with torch.no_grad():
            for train_data, bridge_data, earth_data, labels in self.data_iter(proportion=1, shuffle=False, data=data, batch_size=1):
                predict_ = net(train_data, bridge_data, earth_data, self.adj_mat)
                predict.append(predict_.detach().cpu())
                true.append(labels.detach().cpu())

        # (N, D, L)
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, predict.shape[-2], predict.shape[-1]).permute(1, 2, 0)
        true = torch.cat(true, dim=0)
        true = true.reshape(-1, true.shape[-2], true.shape[-1]).permute(1, 2, 0)

        mae_loss = torch.mean(torch.abs(predict - true), dim=[1, 2]).mean()
        mse_loss = torch.pow(predict - true, 2).mean(dim=[1, 2]).mean()

        mape = torch.abs((predict - true) / (torch.abs(true) + 0.05)).mean(dim=[1, 2]).mean()

        rmse = torch.sqrt(mse_loss)
        print('----' * 10)

        predict = self.re_norm(predict)
        true = self.re_norm(true)

        if save:
            predict_data_name = f'{net}-{self.n_span}.pt'
            true_data_name = f'true-{self.n_span}.pt'
            predict_data_path = os.path.join("predict_data", predict_data_name)
            true_data_path = os.path.join("predict_data", true_data_name)
            torch.save(predict, predict_data_path)
            torch.save(true, true_data_path)
        if show:
            show_predict(true, predict, time_interval=5e-5*self.freq, show_point=show_point, n_span=self.n_span, model_name=net.name())
        return mae_loss, mse_loss, mape, rmse

    def predict_sample(self, net, show_point=False, show=True, save=True, n_sample=10):
        net.eval()
        predict_sample = []
        n_sample = n_sample

        with torch.no_grad():
            for train_data, bridge_data, earth_data, labels in self.data_iter(proportion=1, shuffle=True, batch_size=1)[0]:
                predict = []
                true = labels.clone().squeeze(dim=0).detach().cpu()
                for _ in range(n_sample):
                    predict_ = net(train_data, bridge_data, earth_data, self.adj_mat)
                    predict.append(predict_.detach().cpu())
                break

        # (N, D, L)
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, predict.shape[-2], predict.shape[-1]).permute(1, 2, 0)
        # predict_sample.append(predict)
        # predict_sample = torch.cat(predict_sample, dim=-1)
        # true = torch.cat(true, dim=0)
        true = true.permute(1, 2, 0)
        predict = self.re_norm(predict).contiguous().view(predict.shape[0], predict.shape[1], n_sample, -1)
        true = self.re_norm(true)
        mae_loss = torch.mean(torch.abs(predict - true.unsqueeze(dim=-2).repeat(1, 1, n_sample, 1)), dim=[-1, -2]).sum()
        print(f"All Sampled MAE Loss: {mae_loss}")

        if show:
            show_sample_predict(true, predict, time_interval=5e-5*self.freq, show_point=show_point, n_span=self.n_span)


class DataIterMixer(object):
    # 混合训练模式
    def __init__(self, n_span_list, enc_length, dec_length, batch_size, freq=100, device=torch.device('cpu'), show=True):
        assert isinstance(n_span_list, list), print("n span list should be a list")
        self.data_iter_list = [DataIter(n_span, enc_length, dec_length, batch_size, freq=freq, device=device, show=show) for n_span in n_span_list]
        self.idx = -1

    def choose_data_iter(self, idx=None) -> DataIter:
        self.idx += 1
        if idx is not None:
            return self.data_iter_list[idx]
        return self.data_iter_list[self.idx % self.data_iter_list.__len__()]


if __name__ == '__main__':
    count = 0
    data_iter = DataIter(3, 36, 72, 64, freq=1, show=False)
    data_iter.adj()
    for dataloader in data_iter.data_iter(data_mode="simple", proportion=0.8, data="test"):
        for train, bridge, earth, label in dataloader:
            if train == "None":
                break
            print("Train shape: ", train.shape, end=' ')
            print("Bridge shape: ", bridge.shape, end=' ')
            print("Earth shape: ", earth.shape, end=' ')
            print("label shape: ", label.shape)
        count += 1
    print("Total Batch: ", count)


