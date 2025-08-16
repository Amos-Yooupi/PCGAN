import torch
import matplotlib.pyplot as plt


# 用来记录每次的loss
class RecoderComplex(object):
    def __init__(self):
        self.mse_list = []
        self.kl_list = []

    def append(self, mse_loss, kl_loss):
        self.mse_list.append(mse_loss.detach().cpu().item())
        self.kl_list.append(kl_loss.detach().cpu().item())

    def show(self):
        fig, axes = plt.subplots(2, 1)
        axes = axes.flatten()

        axes[0].set_xlabel("Iterations")
        axes[0].set_ylabel("MSE Loss Value")
        axes[0].set_title("Loss for training")
        axes[0].plot(self.mse_list)

        axes[1].set_xlabel("Iterations")
        axes[1].set_ylabel("KL Loss Value")
        axes[1].set_title("Loss for training")
        axes[1].plot(self.kl_list)

        plt.savefig("loss.jpg")
        plt.show()

    def __repr__(self):
        return f"train loss: {round(self.mse_list[-1], 4)}, kl loss: {round(self.kl_list[-1], 4)}"

    def stop(self, min=1e-5, limit=1e-6):
        if len(self.mse_list) < 2:
            return False
        if 0 < self.mse_list[-2] - self.mse_list[-1] < limit and self.mse_list[-1] < min:
            print("自动停止")
            return True
        return False



class RecoderSimple(object):
    def __init__(self):
        self.mse_list = []

    def append(self, mse_loss):
        self.mse_list.append(mse_loss.detach().cpu().item())

    def show(self):
        fig, axes = plt.subplots(1, 1)

        axes.set_xlabel("Iterations")
        axes.set_ylabel("MSE Loss Value")
        axes.set_title("Loss")
        axes.plot(self.mse_list)

        plt.savefig("loss.jpg")
        plt.show()

    def __repr__(self):
        return f"train loss: {round(self.mse_list[-1], 6)}"

    def stop(self, min=1e-5, limit=1e-6):
        if len(self.mse_list) < 2:
            return False
        if 0 < self.mse_list[-2] - self.mse_list[-1] < limit and self.mse_list[-1] < min:
            print("自动停止")
            return True
        return False
