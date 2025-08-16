import torch


def kl_divergence_loss(mu, logvar):
    """
    计算 VAE 的 KL 散度损失

    参数:
    - mu: 编码器输出的均值，形状为 (batch_size, latent_dim)
    - logvar: 编码器输出的对数方差，形状为 (batch_size, latent_dim)

    返回值:
    - kl_loss: KL 散度损失
    """
    # KL 散度的公式 - 1/2 * (1 + log(var) - mu^2 - var)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=[0, 1]).sum()
    return kl_loss


if __name__ == '__main__':
    # 示例输入
    batch_size, latent_dim = 16, 10
    mu = torch.randn(batch_size, latent_dim)  # 均值
    logvar = torch.randn(batch_size, latent_dim)  # 对数方差

    # 计算 KL 损失
    kl_loss = kl_divergence_loss(mu, logvar)

    print(kl_loss)
