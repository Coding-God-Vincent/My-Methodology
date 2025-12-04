import torch
import numpy as np
# 明明 to_torch 在 tianshou.data.utils.converter.py 中，(utils 也是一個 package) 
# 但可以這樣直接找到 to_torch 是因為在 data 的 __init__.py 中有先找出來。
# 不然照理說確實是要一步一步的引用出來
from tianshou.data import Batch, ReplayBuffer, to_torch
from pathlib import Path


# log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs")
# exp_path = log_path / "D2AC" / "exp1"
# # 於 exp_path 中建立一個
# exp_path.mkdir(parents= True, exist_ok= True)


# a = torch.tensor([[1, -2, 3]])
# b = torch.abs(a).squeeze(dim= 0).numpy()
# print(b)
# a = torch.tensor([-5, -1, 9])
# print(torch.tanh(a))

# a = np.array([5])
# print(a.squeeze())

# h = {'actor_loss' : 1,
#      'critic_loss' : 1}
# print(h['actor_loss'])
# a = torch.tensor([[3], [1]], dtype= torch.float)
# print(torch.randn_like(a))
# a = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype= torch.float32)  # shape (2, 3)
# b = torch.tensor([[4, 5, 6], [4, 5, 6]], dtype= torch.float32)  # shape (2, 3)
# loss = torch.nn.functional.mse_loss(a, b)
# print(loss.shape, loss)

# a = torch.tensor([[1], [2], [3]], dtype= torch.float32)  # shape (3, 1)
# print(a.shape)
# print(a.mean().shape, a.mean())

