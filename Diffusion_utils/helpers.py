import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------------------------------#
# 對 timestep 進行位置編碼 (positional embedding)，讓模型知道現在是 Diffusion 的第幾步
# 並非一個神經網路，只是因為很常被當作模組呼叫所以繼承 nn.Module 來寫
# 用在 model.py，會將時間輸入並輸出 embedding 後的結果
#------------------------------------------------------------------------------------------------------#
'''About Sinusoidal Position Embedding
* 動機 : 因為模型單靠數字沒辦法深刻理解時間的涵義，因此需要將時間資訊轉換成讓模型能夠理解的張量來表示。

* 直觀理解 : 把時間步 t，投影到多個不同頻率的 sin/cos 波中，每個頻率充當一個不同刻度的時間尺。就像我們會用時、分、秒來描述時間一樣，一個頻率就相當於一種刻度。
            不同 t 在每個頻率下會有不同的相位位置，為了唯一表示該 t 在該頻率的相位，我們用 sin 和 cos 兩個波來表示出該相位的相位位置。(頻率中的一個相位能透過其在 sin 波和 cos 波的相位位置來唯一表示)
            所有頻率的相位位置組合起來，就形成一個能唯一代表 t 的向量。此為時間點 t embedding 後的結果。
            時間點相近的向量會相似，遠的會差很多。由此讓模型知道兩者的差距。

* 下方作法 : 傳入 dim，此及一個時間點 t 要被 embedding 成的張量的維度。也就是其在各種頻率中的相位位置。
            由上我們已經知道表達一個相位需要 sin & cos 值，因此我們可以推得我們總各會有 half_dim = dim / 2 個不同的頻率，每個頻率會有兩個波來表達該頻率的相位位置。
            (3) 會切出各頻率間隔
            (4) 根據 (3) 所切的間隔創建出各種不同的頻率
            (5) 用各種不同的 t 去乘上各種不同的頻率，會得到各種不同的 t 在各種不同頻率上的相位。(還要套上 sin/cos 才會變成各種時間點在 sin/cos 波上的相位位置)
            (6) 套上 sin & cos，讓前一步得出的相位變成此兩波上的相位位置。用這兩個相位位置能得出在該頻率中一個唯一的值。這就是我們 embedding 後的結果。
                我們用 half_dim 個頻率，就會有 half_dim 個 embedding 結果 (每一個結果都由一個 sin 波的相位和 cos 波的相位所組成)。

'''

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 把一個時間資訊 embbeding 成一個 dim 維的張量
    
    # x : batch 中各資料的時間步 t。shape (bacth_size)
    def forward(self, x):
        device = x.device
        # half_dim : 會使用到的頻率種類，每一個頻率種類都會產生出對應該頻率的一個 sin 波 & cos 波 (因為要同時使用 sin & cos 才能表達頻率中唯一的相位)
        half_dim = self.dim // 2
        # 把 log(10000) 切成 half_dim 塊 (除以 half_dim-1 沒錯，這是為了讓第一個元素跟最後一個元素對應到 log 空間的兩端)
        # 切出兩個不同頻率間的間隔
        # 這種切法再搭配下一行能將頻率種類均勻地落在 1 ~ 1/10000 之間
        emb = math.log(10000) / (half_dim - 1)
        # 得出各頻率
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # shape (half_dim)
        # 用時間步 t 去乘上各種不同的頻率，由此創造出時間步 t 在不同頻率之下的相位 (還要套上 sin or cos 才會變成波)
        # x[:, None] 相當於 x.squeeze(dim= -1)，使得 X 的 shape 從 (batch_size) 變成 (batch_size, 1)
        # emb 的 shape 從 (half_dim) 變成 (1, half_dim)
        emb = x[:, None] * emb[None, :]  # shape (batch_size, half_dim)
        # concat 後得到 shape (batch_size, dim)
        # 每一個時間點 t 皆需要其在 sin 波的相位位置和 cos 波的相位位置，這樣才可以唯一表示其在該頻率的相位位置
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb  # shape (batch_size, dim)。

#------------------------------------------------------------------------------------------------------#
# 為了一次處理一個 batch 的資料才需要此函式
# 將一個 batch 中各 x_t 的時間步 t 和其對應的係數 a 取出放在一個 tensor 之中
# a -> 跟 t 有關的係數，例如 alphas_cumprod = [alpha_bar_0, alpha_bar_1, ..., alpha_bar_n-1]。shape (timesteps)
# t -> 時間點，每一維度的值皆介於 [0, timesteps-1]。shape (batch_size)。t is used as an index, so set dtype of t to torch.long
#------------------------------------------------------------------------------------------------------#
# * -> 讓值從資料結構中解脫變成獨立的值。ex: *(1, 2, 3) -> 1, 2, 3
def extract(a, t, x_shape):
    b, *_ = t.shape  # b -> batch_size
    out = a.gather(-1, t)  # dim= -1 (即最後一個維度，在這邊 a.shape = (timesteps)) # 這邊就是取出時間點 t 對應的係數 # output_shape (timesteps)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # shape (timesteps, 1, 1, ..., 1) 後面的 1 有 len(x_shape - 1) 
    # 若 x_shape (batch_size, action_dim)，len(x_shape) = 1，那 out.shape 會變成 (batch_size, 1)

#------------------------------------------------------------------------------------------------------#
# 3 types of \beta scheduling
# 回傳 timestpes 個 beta_t，並將這些值放入一個 tensor
#------------------------------------------------------------------------------------------------------#
# beta 利用 cosine 產生平滑的值
def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

# beta 從 1e-4 線性增加到 2e-2
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

# 此 project 中所使用，使隨機變數的變異數保持為 1 的那種
# 保證訊號在整個 Diffusion 過程中變異數的變化平滑可控，適用於 SDE-based diffusion
def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

#------------------------------------------------------------------------------------------------------#
# Noise Generator
#------------------------------------------------------------------------------------------------------#
class GaussianNoise:
    """Generates Gaussian noise."""

    def __init__(self, mu= 0.0, sigma= 0.1):
        """
        :param mu: Mean of the Gaussian distribution.
        :param sigma: Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    # 從均勻分布中抽樣出一個值
    def generate(self, shape):
        """
        Generate Gaussian noise based on a shape .

        :param shape: Shape of the noise to generate, typically the action's shape.
        :return: Numpy array with Gaussian noise.
        """
        noise = np.random.normal(self.mu, self.sigma, shape)
        return noise

