import os.path
import torch
import os
from models.GatNet import GNN, GAT_ProbSparseAttention1, GAT_ProbSparseAttention2, GAT, GAT_LSTM, GATtenion
from models.TCN import TCN, GAT_TCN
from models.LSTM import LSTM
from models.STGNN import STGCN
from models.Graph_wave import GraphWaveNet
from utils.CustomDataSet_update import DataIter, DataIterMixer
from utils.Recoder import RecoderSimple, RecoderComplex
from models.model_utils.xavier_parameter import xavier_parameters
from utils.show_data import show_compare
import argparse
from models.model_utils.Config import Config
from models.Dlinear import Dlinear
from models.PatchTST import PatchTST_Customer, PatchTST_GAT
from models.PCGAN import PCGAN, PCGAN_no_exaction, PCGAN_no_exaction_attention_point
from utils.KLLoss import kl_divergence_loss


# 创建参数解析器
parser = argparse.ArgumentParser(description='GATttention')
# 添加命令行参数
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
# 解析参数
args = parser.parse_args()
# 加载配置文件
config = Config(args.config)

# 网络参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
test_n_span_list = 11
n_span_list = [5, 6, 7, 9]

#
n_span_list = [3, 4, 10, 11]
test_n_span_list = 6


recoder = RecoderComplex()
recoder_val = RecoderSimple()
# 网络设置
# net = GATtenion(config).to(device)
# net = GAT_LSTM(config).to(device)
# net = GNN(config).to(device)
# net = GAT(config).to(device)
# net = LSTM(config).to(device)
# net = TCN(config).to(device)
# net = GAT_TCN(config).to(device)
# net = Dlinear(config).to(device)
# net = PatchTST_Customer(config).to(device)
# net = PatchTST_GAT(config).to(device)
net = PCGAN(config).to(device)
# net = PCGAN_no_exaction(config).to(device)
# net = PCGAN_no_exaction_attention_point(config).to(device)
# net = STGCN(config).to(device)
net = GraphWaveNet(config).to(device)

# 损失函数和优化器
criterion = torch.nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
# 加载数据集
data_iter = DataIterMixer(n_span_list, config.enc_length, config.dec_length, config.batch_size, 200, device, show=False)
test_data_iter = DataIter(test_n_span_list, config.enc_length, config.dec_length, config.batch_size, 200, device, show=False)
# 模型名称 zx
best_model_name = f'{net.__repr__()}-multi_best.pt'
final_model_name = f'{net.__repr__()}-multi-final.pt'
load_model_name = f'{net.__repr__()}-multi-final.pt'
best_model_path = os.path.join("model_parameter", best_model_name)
final_model_path = os.path.join("model_parameter", final_model_name)
load_model_path = os.path.join("model_parameter", load_model_name)
# 是否训练和加载模型
training = True
load = False

# training = False
# load = True

if __name__ == '__main__':
    if load and os.path.exists(load_model_path):
        try:
            net.load_state_dict(torch.load(load_model_path))
            print(f"{net.name()}加载成功！")
        except:
            print(f"模型发生改变！---error----重新训练")
    else:
        print("训练模型！")
    if training:
        for i in range(config.num_epochs):
            count = 1
            solo_data_iter = data_iter.choose_data_iter()

            for train_data, bridge_data, earth_data, labels in solo_data_iter.data_iter(data_mode="simple", proportion=1, data="train"):
                if train_data == "None":
                    break
                predict = net(train_data, bridge_data, earth_data, solo_data_iter.adj_mat)
                loss = criterion(predict, labels).mean(dim=[0, 1]).mean()
                try:
                    kl_loss = kl_divergence_loss(net.latent_layer.mean, net.latent_layer.logvar)
                except:
                    kl_loss = torch.tensor([0])
                recoder.append(loss, kl_loss)
                print(f"epoch: {i}---iterations: {count}---{recoder}")
                # loss += kl_loss / 1e2
                count += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if recoder.stop():
                break

            if i % 20 == 0:
                for train_data, bridge_data, earth_data, labels in test_data_iter.data_iter(data_mode="simple", proportion=1, data="train", batch_size=64):
                    if train_data == "None":
                        break
                    predict = net(train_data, bridge_data, earth_data, test_data_iter.adj_mat)
                    loss = criterion(predict, labels).mean(dim=[0, 1]).mean()
                    try:
                        kl_loss = kl_divergence_loss(net.latent_layer.mean, net.latent_layer.logvar)
                    except:
                        kl_loss = torch.tensor([0])
                    val_loss = loss
                    recoder_val.append(val_loss)
                    if val_loss <= min(recoder_val.mse_list):
                        torch.save(net.state_dict(), best_model_path)
                        print(f"模型--{best_model_name}--保存成功")
            if i % 200 == 0:
                torch.save(net.state_dict(), final_model_path)
                print(f"模型--{final_model_name}--保存成功")
        recoder.show()
        recoder_val.show()

    best_model = GAT(config).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    for i in range(len(n_span_list[:])):
        solo_data_iter = data_iter.choose_data_iter()
        loss_final = solo_data_iter.predict(net, show_point=False, show=False, data='train')
        loss_best = solo_data_iter.predict(best_model, show_point=False, show=False, data='train')
        new_loss_list = []
        for i in range(len(loss_best)):
            new_loss_list.append(min(loss_final[i], loss_best[i]))
        mae_loss, mse_loss, mape, rmse = new_loss_list
        print(f"All MAE Loss: {round(mae_loss.item(), 5)}")
        print(f"All MSE Loss: {round(mse_loss.item(), 5)}")
        print(f"All MAPE Loss: {round(mape.item(), 5)}")
        print(f"All RMSE Loss: {round(rmse.item(), 5)}")


        solo_data_iter.predict(net, show_point=True, show=True, data='train')

        # solo_data_iter.predict_sample(net, show_point=False, show=True, n_sample=200)

    # for span in n_span_list:
    #     show_compare(span, show=True)
