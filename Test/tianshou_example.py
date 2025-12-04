import numpy as np
from tianshou.data import Batch, ReplayBuffer, to_torch
import torch

device = 'cuda' if torch.cuda.is_available else 'cpu'

'''
* 存入 ReplayBuffer 中的東西一定要是 np.array 且每一筆資料一定都要有 obs, act, rew, terminated, truncated, obs_next
* 不論你存入的 shape 是 (1, state_dim) 還是單純的 (state_dim)，從 ReplayBuffer 中取樣出來之後都會是 (batch_size, state_dim)
'''

#-------------------------------------------------------------------------------------------------------------------------------------------#
# 創建一個 ReplayBuffer with size= 5
buf = ReplayBuffer(size= 5)

# 創建一筆資料，一定要有以下 6 個欄位 (gym 的標準資料類型)
# 也可以把其他參數加入到 info 之中
data = Batch(
    obs=np.array([1, 2]),  # shape (2)
    act=np.array([0]),  # shape (1)
    rew=1,  # shape (1)
    terminated=False,
    truncated=False,
    obs_next=np.array([3, 3])
)

# 將資料加入 buffer 中
# 若想加入多筆資料加入 replaybuffer，必須要用 for 一筆一筆加，目前沒有一次加入全部的函式
# 資料若沒有要加入 replaybuffer，那你要放啥、key 叫啥都沒差，但只要你是要放進 replaybuffer 的，就是要有下面那六個 keys
buf.add(data)

data = Batch(
    obs=np.array([3, 4]),  # shape (2)
    act=np.array([1]),  # shape (1)
    rew=1,  # shape (1)
    terminated=False,
    truncated=False,
    obs_next=np.array([4, 4])
)

buf.add(data)

# 可以用 key 找到 value
print(data['obs'])  # [3, 4]

print(f"目前 Buffer 長度: {len(buf)}")

# sample() 會回傳抽到的 batch_data (存在一個 Batch 中) & indices (該 batch_data 在 Batch 中的 index)
batch_data, indices = buf.sample(batch_size= 2)
# batch.obs.shape (batch_size, state_dim) np.array
print(f"batch.obs= {batch_data.obs}, batch.obs.shape= {batch_data.obs.shape}, batch.obs.dtype= {batch_data.obs.dtype}")
# batch.act.shape (batch_size, action_dim)  np.array
print(f"batch.act= {batch_data.act}, batch.obs.shape= {batch_data.act.shape}, batch.act.dtype= {batch_data.act.dtype}")
# batch.rew.shape (batch_size, r_dim)  np.array (float64)
print(f"batch.rew= {batch_data.rew}, batch.rew.shape= {batch_data.rew.shape}, batch.obs.dtype= {batch_data.rew.dtype}")
print(f"batch.terminated= {batch_data.terminated}, batch.terminated.shape= {batch_data.terminated.shape}, batch.terminated.dtype= {batch_data.terminated.dtype}")
print(f"batch.truncated= {batch_data.truncated}, batch.truncated.shape= {batch_data.truncated.shape}, batch.truncated.dtype= {batch_data.truncated.dtype}")
# batch.obs_next.shape (batch_size, state_dim) np.array
print(f"batch.obs_next= {batch_data.obs_next}, batch.obs_next.shape= {batch_data.obs_next.shape}, batch.obs_next.dtype= {batch_data.obs_next.dtype}")

print("\n")

#-------------------------------------------------------------------------------------------------------------------------------------------#
# --- 轉換 ---
# 用 to_torch 能夠直接將一個 Batch 中所有的資料一次轉為 tensor，並可以同時設定 dtype & device
# 1. 基本轉換 (轉成 Tensor)
batch_tensor = batch_data.to_torch(device= device, dtype= torch.float32)

print(f"batch_data.datatype = {type(batch_data.obs)}")  # <class 'numpy.ndarray'>
print(f"batch_tensor.datatype = {type(batch_tensor.obs)}")  # <class 'torch.Tensor'>
print(f"batch_tensor.device = {batch_tensor.obs.device}")  # cuda

#-------------------------------------------------------------------------------------------------------------------------------------------#
# 沒有要放入 replaybuffer 的 batch，想放啥就放啥
data2 = Batch(key1= 1, key2= 3)
print(data2['key1'])  # return 1

