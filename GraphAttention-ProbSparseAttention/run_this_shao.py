import os.path
import torch
import os
from utils.CustomDataSet_Shao import DataIter, DataIterMixer
from utils.Recoder import RecoderSimple
from utils.show_data import show_compare
import argparse
from models.model_utils.Config import Config
from models.model_shao import GAT_Shao, GAT_LSTM_Shao, TCN_Shao


# 创建参数解析器
parser = argparse.ArgumentParser(description='GATttention')
# 添加命令行参数
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
# 解析参数
args = parser.parse_args()
# 加载配置文件
config = Config(args.config)

# 网络参数设置
"dropout 影响很大，数据太少了，dropout太大容易欠拟合"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_train_list = [3, 4, 5, 6, 7, 9, 10, 11]


recoder = RecoderSimple()
# 网络设置
# net = GAT_Shao(config).to(device)
# net = GAT_LSTM_Shao(config).to(device)
net = TCN_Shao(config).to(device)

# 损失函数和优化器
criterion = torch.nn.L1Loss(reduction="none")
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
# 加载数据集
data_iter = DataIterMixer(n_train_list, config.enc_length, config.dec_length, config.batch_size, device, show=False)
# 模型名称
model_name = f'{net.__repr__()}-shao_{config.enc_length}.pt'
model_path = os.path.join("model_parameter", model_name)
# 是否训练和加载模型
training = True
load = False


if __name__ == '__main__':
    if load and os.path.exists(model_path):
        try:
            net.load_state_dict(torch.load(model_path))
            print(f"{model_name}加载成功！")
        except:
            print(f"{model_name}发生改变！---error----重新训练")
    else:
        print("训练模型！")
    if training:
        for i in range(config.num_epochs):
            count = 1
            solo_data_iter = data_iter.choose_data_iter()
            for graph_features, dis, labels in solo_data_iter.data_iter(proportion=1, data_mode="simple"):
                predict = net(graph_features, dis, solo_data_iter.adj)
                loss = criterion(predict, labels).mean(dim=0).sum()
                recoder.append(loss)
                print(f"epoch: {i}---iterations: {count}---{recoder}")
                count += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not i % config.save_freq:
                torch.save(net.state_dict(), model_path)
                print(f"模型--{model_name}--保存成功")
            if recoder.stop():
                break

        torch.save(net.state_dict(), model_path)
        print(f"模型--{model_name}--保存成功")
        recoder.show()

    for i in range(len(n_train_list)):
        solo_data_iter = data_iter.choose_data_iter()
        solo_data_iter.predict(net, show_point=False, show=True)

    # for span in n_train_list:
    #     show_compare(span, show=False)
