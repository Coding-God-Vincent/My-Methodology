'''
* 本論文的時間定義：
    frame = learning window = 1s = 2000 subframe (每一個 frame 做一次上層 by GAN-DDQN)
    time_subframe = 0.5ms (每一個 time_subframe 做一次下層 by RR)

* 這邊的對模型輸入輸出的註解都是以 batch 的角度在寫，但其實這種 batch 的操作模型會自動做，所以可以看到在程式的寫法中都沒有考慮 batch，只考慮單一狀態的輸入。

* 每 1s 會做一次上層的分配 (SDN Controller 分給 NS)，每 0.5ns 會做一次下層分配 (NS 分給 UE)。

* 論文的系統時間的 Granularity = 0.5ms (timeslot)，一個 Learning window = 2000 個 timslot，即 2000 * 0.5ms = 1s。
  一個 learning window 會做一次上層分配，2000 次下層分配。每個 learning window 會更新一次參數。
'''

'''跟 GANDDQN_v1 的差別
* 加上 moving_average()，讓結果輸出更好看。
* 把 WGAN-GP 改為 WGAN-LP (Lipschitz Penalty)。
'''

#%%
import torch, time, os, pickle, glob, math, json
import numpy as np
import csv
from timeit import default_timer as timer 
from datetime import timedelta
import itertools
import pandas as pd
from tqdm.auto import tqdm

# simulation environment
from Env.env_fixedUE import cellularEnv  # root path must be "~/NCKU/Paper/My Methodology"

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import torchvision.transforms as T

from GAN_utils.ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory
# from utils.wrappers import *
from GAN_utils.utils import initialize_weights

#=============================================================================================================================================#
#%%  # 就像 Ipynb 一樣的功能，把程式碼切成一個一個的 Cell
# 建立 Generator 網路
# 將狀態、Quantiles 向量傳入後輸出所有 Q 值的機率分布 (|A| * num_samples 個 particles，對應 |A| 個 Q 值機率分布)
class Generator(nn.Module):
    def __init__(self, state_size, num_actions, num_samples, embedding_dim):
        super(Generator, self).__init__()  # same as super().__init__()
        self.state_size = state_size  # len(ser_cat)
        self.num_actions = num_actions  # |Action_Space|
        self.num_samples = num_samples  # 每個 Q 值機率分布的 particles 數量
        self.embedding_dim = embedding_dim  # quantile vector 的向量維度 (N)

        # 將模型直接建在 GPU
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda')
        
        # 設定 Embedding Layer (兩個的輸出維度皆為 N，一定要一樣長，因為要做 Hadamard Product)
        self.embed_layer_1 = nn.Linear(self.state_size, self.embedding_dim)  
        # self.embed_layer_drop_1 = nn.Dropout(p = 0.5)  # 防止 overfitting，每次 forward propagation 的時候隨機將輸入中的美個神經元以機率 p 設為 0
        self.embed_layer_2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.embed_layer_drop_2 = nn.Dropout(p = 0.5)
        
        # 設定 Particle Generation Component 的 Layer
        # Hadamard Product (N, N) -> N (embedding_dim)，故 fc1.in_features = N
        self.fc1 = nn.Linear(self.embedding_dim, 256)
        self.drop1 = nn.Dropout(p = 0.5)  # forward 那邊也沒用，可能效果不好
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(128, self.num_actions)  # out_shape = |Action_Space|

        initialize_weights(self)  # 初始化各層參數 (寫在 utiils.py)

    # 上面 __init__() 中的參數是在創建模型是要輸入的，這邊的 forward() 中的參數是在進行 forward Propagation 時要輸入的，常寫成 model("state")。
    def forward(self, x, tau):   # tau : quantiles vector = torch.randn(self.batch_size * self.num_samples)
        
        # 分別對 State & quantiles 經過 embedding layer 
        state_tile = x.repeat(1, self.num_samples)  # 將 state 複製 self.num_samples 次，其 shape = [self.batch_size, (self.state_size * self.num_samples)]
        state_reshape = state_tile.view(-1, self.state_size)  # shape : [(self.batch_size * self.num_samples), self.state_size]
        state = F.relu(self.embed_layer_1(state_reshape))  # shape : [(self.batch_size * self.num_samples), self.embedding_dim]
        # state = self.embed_layer_drop_1(state)

        # 設定 quantile vector，做了這麼多而非直接使用原始 tau 是為了讓模型捕捉到不同的 tau 要輸出不同的 Q 值
        # 將傳入的 tau (隨機產生的) 進行 reshape
        tau = tau.view(-1, 1)  # 效果同 tau = tau.reshape(-1, 1)，即把 tau 的 shape 從 (self.batch_size * self.num_samples) 變成 (原本的 shape, 1)
        # 回傳一個 (1, embedding_dim) 的 np.ndarray。
        # np.expand_dims("array", "axis") : 在指定 axis 新增一個維度。ex : np.expand_dims([1, 2, 3], 0).shape -> [1, 3]
        pi_mtx = torch.from_numpy(np.expand_dims(np.pi * np.arange(0, self.embedding_dim), axis=0)).to(torch.float).to(self.device)
        # 做 cosine basis encoding，類似 Transformer 中的 position encoding，引入不同頻率的週期性資訊。
        # 由此能讓模型把 quantile 映射到更高維、非線性的空間
        cos_tau = torch.cos(torch.matmul(tau, pi_mtx)) #  shape : [(self.batch_size * self.num_samples), self.embedding_dim]
        # 將處理過的 quantile 經過 embedding layer (relu) 抽取特徵。
        pi = F.relu(self.embed_layer_2(cos_tau))  # shape : [(self.batch_size * self.num_samples), self.embedding_dim]
        # pi = self.embed_layer_drop_2(pi)

        # Hadamard Product (shape 相同的 np 矩陣相乘會做對應項相乘 (Hadamard))
        x = state * pi  # shape : [(self.batch_size * self.num_samples), self.embedding_dim]
        
        # Particle Generation Component
        x = F.relu(self.fc1(x))  # shape : [(self.batch_size * self.num_samples), 256]
        # x = self.drop1(x)  
        x = F.relu(self.fc2(x))  # shape : [(self.batch_size * self.num_samples), 128]
        # x = self.drop2(x)
        x = self.fc3(x)  # shape : [(self.batch_size * self.num_samples), self.num_actions]

        # x.view() : shape from [(self.batch_size * self.num_samples), self.num_actions] -> [self.batch_size, self.num_samples, self.num_actions]
        # transpose : shape from [self.batch_size, self.num_samples, self.num_actions] -> [self.batch_size, self.num_actions, self.num_samples]
        net = torch.transpose(x.view(-1, self.num_samples, self.num_actions), 1, 2)  # [self.batch_size, self.num_actions, self.num_samples]
        return net

#=============================================================================================================================================# 
# 建立 Disciminator 網路  
# input_shape : [self.batch_size, self.num_samples]
# output_shape : [self.batch_size, self.num_outputs]
class Discriminator(nn.Module):
    def __init__(self, num_samples, num_outputs):  # num_output = 1
        super(Discriminator, self).__init__()
        self.num_inputs = num_samples
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(self.num_inputs, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.num_outputs)
        initialize_weights(self)

    def forward(self, x, z):
        # add little noise
        # z = 0. * torch.randn(self.batch_size, self.num_samples).to(self.device) = 0
        # 所以其實沒有加上 noise
        # x = x + z
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

#=============================================================================================================================================#
# 沒用到
# 到 start_timestep 之後開始做線性衰弱 (Linear Schedule)，用於控制 ɛ-greedy
# schedule_timesteps : 從 init_p 到 final_p 所需的時間步數
class LinearSchedule(object):  # 沒繼承還寫上 "Object" -> 這是舊式寫法，效果同 Class LinearSchedule():
    def __init__(self, schedule_timesteps, start_timesteps, final_p, initial_p = 1.0):
        '''
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        '''
        self.schedule_timesteps = schedule_timesteps
        self.start_timesteps = start_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):  # t 為當前 timestep
        if t < self.start_timesteps:
            return self.initial_p
        else:  # 開始做線性衰弱
            # fraction 隨時間步增大
            fraction = min(float(t) / (self.schedule_timesteps + self.start_timesteps), 1.0)
            return self.initial_p + fraction * (self.final_p - self.initial_p)

#=============================================================================================================================================#
class WGAN_GP_Agent(object):
    def __init__(self, static_policy, num_input, num_actions):
        super(WGAN_GP_Agent, self).__init__()
        
        # device
        self.device = torch.device('cuda')

        # parameters
        self.gamma = 0.75
        self.lr_G = 1e-4
        self.lr_D = 1e-4
        self.target_net_update_freq = 10  # target_G 參數逼近 G (instead of sync.) every 10 steps
        self.experience_replay_size = 2000  # 2000
        self.batch_size = 32  # 32
        self.update_freq = 200  # update WGAN_GP every 200 learning windows
        # 一次 update 會更新 len(memroy) / 32 輪。每輪更新 n_critic 次 D，n_gen 次 G
        self.learn_start = 0  # 從第幾步開始允許模型開始更新參數
        self.tau = 0.1  # default is 0.005 (不是 G 使用的 tau，是 Target_G 逼近 G 時使用的)
        self.static_policy = False  # True -> evaluation，False -> train
        self.num_feats = num_input  # state_size
        self.num_actions = num_actions  # |Action_space|
        self.z_dim = 32  # len(quantile vector)
        self.num_samples = 32  # samples in 1 Q-value distribution
        self.lambda_ = 10  # Gradient Penalty 的 lambda
        self.n_critic = 5  # 訓練一次 G 會之前會先訓練 5 次 D
        self.n_gen = 1

        # model
        # 創建網路 (G_model, G_target_model, D_model)
        self.declare_networks()
        self.G_target_model.load_state_dict(self.G_model.state_dict())
        # set optimizer
        # betas((0.5, 0.999)) -> (一階動量保持多少比例的梯度, 二階動量保持多少比例的梯度) 參考 DCGAN 的建議
        self.G_optimizer = optim.Adam(self.G_model.parameters(), lr = self.lr_G, betas = (0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D_model.parameters(), lr = self.lr_D, betas = (0.5, 0.999))
        # device
        self.G_model = self.G_model.to(self.device)
        self.G_target_model = self.G_target_model.to(self.device)
        self.D_model = self.D_model.to(self.device)
        # train or test
        if self.static_policy:
            self.G_model.eval()
            self.D_model.eval()
        else:
            self.G_model.train()
            self.D_model.train()

        # 創建 replay buffer
        self.declare_memory()

        # 紀錄訓練過程中的 loss 變化
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        
        # 創建 1 & -1 的 tensor
        self.one = torch.tensor([1], device = self.device, dtype = torch.float)
        self.mone = self.one * -1
        
        self.update_count = 0  # 用來記錄目前更新的次數
        self.nsteps = 1  # 使用 single step
        self.nstep_buffer = []

        # (這邊沒用到) 要寫的話也是 nn.BatchNorm1d(num_features).to(self.device)
        # 使用場景：
        # 一個資料中有 num_features 個 features。有 batch_size 筆資料
        # 會一個一個的把每一筆資料中的第 k 號 features (共有 batch_size 個 k 號 features) 一起做正規化，k = [1, num_features]
        self.batch_normalization = nn.BatchNorm1d(self.batch_size).to(self.device)

    #=====================================================================================================================================
    # 創建網路
    def declare_networks(self):
        # Output the probability of each sample
        self.G_model = Generator(self.num_feats, self.num_actions, self.num_samples, self.z_dim) # output: batch_size x (num_actions*num_samples)
        self.G_target_model = Generator(self.num_feats, self.num_actions, self.num_samples, self.z_dim)
        self.D_model = Discriminator(self.num_samples, 1) # input: batch_size x num_samples output: batch_size

    # 創建 replay buffer (list)
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

    # push experience into replay buffer
    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))
        

    # 存模型的參數
    def save_w(self):
            if not os.path.exists('./saved_agents/GANDDQN'):
                os.makedirs('./saved_agents/GANDDQN')  # "在當前路徑/saved_agents/GANDDQN" 創建一個資料夾 
            # .dump 是作者取的副檔名，可以隨便取，只要讀取的時候用 torch.load 就行
            torch.save(self.G_model.state_dict(), './saved_agents/GANDDQN/G_model_10M_0.01.dump')  # 把模型參數存在該資料夾
            torch.save(self.D_model.state_dict(), './saved_agents/GANDDQN/D_model_10M_0.01.dump')

    # 存 replay buffer 中的東西
    # pickle.dump() : 將 self.memory 序列化後將其轉為二進制存入 exp_replay_agent.dump 
    # 'wb' : write binary，以 binary 的形式寫入
    def save_replay(self):
        pickle.dump(self.memory, open('./saved_agents/exp_replay_agent.dump', 'wb'))

    # 載回之前儲存的 replay buffer
    def load_replay(self):
        fname = './saved_agents/exp_replay_agent.dump'
        if os.path.isfile(fname):
            self.memory = pickle.load(open(fname, 'rb'))

    # 載回之前的模型參數
    def load_w(self):
        fname_G_model = './saved_agents/G_model_0.dump'
        fname_D_model = './saved_agents/D_model_0.dump'

        if os.path.isfile(fname_G_model):
            self.G_model.load_state_dict(torch.load(fname_G_model))
            self.G_target_model.load_state_dict(self.G_model.state_dict())
        
        if os.path.isfile(fname_D_model):
            self.D_model.load_state_dict(torch.load(fname_D_model))

    # 顯示 G、D 訓練過程的 loss 變化
    def plot_loss(self):
        plt.figure(2)  # 切換到 2 號 figure (若不存在會自行建立)
        plt.clf()  # 清除 2 號 figure 裡面的東西
        plt.title('Training loss')  
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(self.train_hist['G_loss'], 'r')  # r -> red
        plt.plot(self.train_hist['D_loss'], 'b')
        plt.legend(['G_loss', 'D_loss'])  # 加上圖例 (對應到線的顏色)
        # plt.pause(0.001)  # 讓 figure 短暫暫停 0.001s，讓圖表即時更新和顯示

    # 從 replay buffer 中取出 minibatch 的資料並依照其性質分成數個 batch 的 tensor
    def prep_minibatch(self, prev_t, t):
        # 取出 replay buffer 中第 prev_t 到第 t 筆的資料
        # transitions = (s, a, r, s_)
        transitions = self.memory.determine_sample(prev_t, t)
        # *transition = (s1, a1, r1, s_1), (s2, a2, r2, s_2), ..., (sn, an, rn, s_n)
        # zip(*transition) = (s1, s2, ..., sn), (a1, a2, ..., an), (r1, r2, ..., rn), (s_1, s_2, ..., s_n)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        batch_state = torch.tensor(batch_state).to(torch.float).to(self.device)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).view(-1, 1)
        batch_next_state = torch.tensor(batch_next_state).to(torch.float).to(self.device)
        return batch_state, batch_action, batch_reward, batch_next_state

    # 每 self.target_net_update_freq steps 就更新一次 G & target_G 的參數
    # 兩個網路的參數不會馬上完全一樣，而是 target_G 會 "慢慢追上" G 的參數 (EMA, Exponential Moving Average)
    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            for target_param, param in zip(self.G_target_model.parameters(), self.G_model.parameters()):
                # EMA 更新公式 (self.tau = 0.1，追的速度，越小越穩定，越大越快追上)
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    # target_Q distribution = immediate_r + gamma * max(target_Q(next_state)
    # 這邊是在做 max 那一段，回傳的 shape : (self.batch_size, 1, self.num_samples)
    def get_max_next_state_action(self, next_states, noise):
        samples = self.G_target_model(next_states, noise)  # samples.shape = (self.batch_size, self.num_actions, self.num_samples)
        # .mean(2) -> 對第二維度的值取 mean，相當於取各動作的平均 Q 值，回傳 shape 為 (self.batch_size, self.num_actions)
        # .max(1)[1] -> 前面的 (1) 是對第一維度的值取 max，回傳 max 的 value, index。後面要的是 [1]，所以是 index。回傳 shape 為 (self.batch_size)
        # next_state.size(0) 是取出第零維度的 size。即 size = (1, 2, 3) 的 1。
        # view() -> 將 max 做完後的 shape (self.batch_size) 轉為 (self.batch_size, 1, 1)
        # expand() -> 將 shape (self.batch_size, 1, 1) 轉為 (self.batch_size, 1, self.num_samples)
        # expand() 那邊轉完 self.num_samples 是複製 self.num_samples 次
        return samples.mean(2).max(1)[1].view(next_states.size(0), 1, 1).expand(-1, -1, self.num_samples)

    # 計算 GP (Gradient Penalty)
    # real_data & fake_data.shape = (self.batch_size, self.num_samples)
    def calc_gradient_penalty(self, real_data, fake_data, noise):
        # 找 ε，為 real_data & fake_data 的混合比例
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size()).to(self.device)
        # interpolates = x_hat，即檢查點
        interpolates = alpha * real_data.data + (1 - alpha) * fake_data.data
        # 因為後面要使用 grad() 去算其梯度
        interpolates.requires_grad = True

        # 算 GP
        disc_interpolates = self.D_model(interpolates, noise)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates, 
                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        # mean() 是因為總共有 num_samples 個 Q 值，要取其 mean() 作為 gradient 代表
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_
        return gradient_penalty
    
    # 計算 LP (Lipschitz Penalty)
    def calc_lipschitz_penalty(self, real_data, fake_data, noise):
        # 找 ε，為 real_data & fake_data 的混合比例
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size()).to(self.device)
        # interpolates = x_hat，即檢查點
        interpolates = alpha * real_data.data + (1 - alpha) * fake_data.data
        # 因為後面要使用 grad() 去算其梯度
        interpolates.requires_grad = True

        # 算 LP
        disc_interpolates = self.D_model(interpolates, noise)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates, 
                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        lipschitz_penalty = (max(0, ((gradients.norm(2, dim=1) - 1) ** 2).mean())) * self.lambda_
        return lipschitz_penalty

    # 自己控制 G & D 的學習率，隨時間遞減。不單只靠 Adam
    # 0~2999 -> self.lr_G
    # 3000~5999 -> 0.1 * self.lr_G
    # 6000~8999 -> 0.01 * self.lr_G
    def adjust_G_lr(self, epoch):
        lr = self.lr_G * (0.1 ** (epoch // 3000))
        # 更改 Optimizer 中的 lr 參數，改成自己的
        for param_group in self.G_optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_D_lr(self, epoch):
        lr = self.lr_D * (0.1 ** (epoch // 3000))
        for param_group in self.D_optimizer.param_groups:
            param_group['lr'] = lr

    # 更新 G、D 參數
    # frame : 當前是第幾個 Learning windows
    def update(self, frame = 0):
        
        # 若現在是測試階段就不做 update
        if self.static_policy:
            return None

        # 若現在還沒到開始學習的時間步就不做 update
        if frame < self.learn_start:
            return None

        # 若還沒到更新的步數就不做 update
        if frame % self.update_freq != 0:
            return None

        # 若 replay buffer 還沒滿就不做 update
        if self.memory.__len__() != self.experience_replay_size:
            return None
        
        # 開始 update ============================================
        print("\nTraining...\n")
        
        # 決定 G & D 的 lr
        self.adjust_G_lr(frame)
        self.adjust_D_lr(frame)

        # 因為樣本不重複使用，因此切好每次要訓練的樣本範圍，將範圍 index 存入 slicing_idx
        self.memory.shuffle_memory()
        len_memory = self.memory.__len__()
        memory_idx = range(len_memory)
        # memory_idx[::self.batch_size] -> 於 memory_idx 中每隔 self.batch_size 取一次 index
        slicing_idx = [i for i in memory_idx[::self.batch_size]]
        # 此論文中，slicing_idx = [0, 32, 64, ..., len_memory]
        slicing_idx.append(len_memory)

        # update G、D 的模型參數
        # Loss_D = E[D(G^a(s, tau))] - E[D(y^a)] + p(lambda_)
        # Loss_G = -E[D(G^a(s, tau))]
        self.G_model.eval()  # G 在這邊還沒要更新，所以設為 eval() 模試
        for t in range(len_memory // self.batch_size):  # 0 ~ 61
            # D 更新 n_critic 次才會更新 1 次 G，避免 mode collapse
            for _ in range(self.n_critic):  # _ 意思是不重要，只要次數有到就好
                # 取出一個 batch 的資料並依照屬性拆成數個 tensor by prep_minibatch()
                batch_vars = self.prep_minibatch(slicing_idx[t], slicing_idx[t+1])
                batch_state, batch_action, batch_reward, batch_next_state = batch_vars
                # batch_action 的 shape 從 (batch_size, 1) 轉成 (batch_size, 1, 1) by unsqueeze 
                # 再轉成 (self.batch_size, 1, self.num_samples) by expand (值等同於對應的 action)
                # ex: 
                # a = torch.tensor([[1], [2], [3]])  # shape = (3, 1)
                # a = a.unsqueeze(dim= -1)  # shape = (3, 1, 1)
                # a = a.expand(-1, -1, 5)  # shape = (3, 1, 5)
                # output : 
                # tensor([ [ [1, 1, 1, 1, 1] ],
                #          [ [2, 2, 2, 2, 2] ],
                #          [ [3, 3, 3, 3, 3] ]
                #        ])
                # 這樣轉 shape 是為了當作下面 gather 的 index。
                batch_action = batch_action.unsqueeze(dim = -1).expand(-1, -1, self.num_samples)  # output_shape = (self.batch_size, 1, self.num_samples)

                # 取得 G 生成的 Q 值機率分布。(fake)，shape : (self.batch_size, self.num_samples)
                G_noise = (torch.rand(self.batch_size, self.num_samples)).to(self.device)
                current_q_values_samples = self.G_model(batch_state, G_noise) # output_shape : (self.batch_size, self.num_actions, self.num_samples)
                # gather : 取出實作於環境的 Q 值的所有 particles (shape : self.batch_size, 1, self.num_samples)
                # squeeze : shape : (self.batch_size, self.num_samples)
                # 最後得出各筆資料實作的動作的 Q 值的所有 particles
                current_q_values_samples = current_q_values_samples.gather(1, batch_action).squeeze(1)

                # Target Q 值機率分布。(real)，shape : (self.batch_size, self.num_samples)
                # Target_Q = immediate_r + gamma * max{Target_Q(s_, max_a(Target_Q(s, tau)), tau)}
                with torch.no_grad():
                    expected_q_values_samples = torch.zeros((self.batch_size, self.num_samples), device=self.device, dtype=torch.float) 
                    # 最後取出各筆資料最大期望值的動作的 Q 值的所有 particles
                    max_next_action = self.get_max_next_state_action(batch_next_state, G_noise)
                    # 這邊的 G_model 應改為 G_target_model 才對 *****
                    expected_q_values_samples = self.G_target_model(batch_next_state, G_noise).gather(1, max_next_action).squeeze(1)
                    expected_q_values_samples = batch_reward + self.gamma * expected_q_values_samples

                # Loss_D = E[D(G^a(s, tau))] - E[D(y^a)] + p(lambda_)
                # D 不用傳 noise，下面那句的 D_noise = 0
                D_noise = 0. * torch.randn(self.batch_size, self.num_samples).to(self.device)
                # E[D(y)]
                D_real = self.D_model(expected_q_values_samples, D_noise)
                D_real_loss = torch.mean(D_real)  
                # E[D(G(s, tau))]
                D_fake = self.D_model(current_q_values_samples, D_noise)
                D_fake_loss = torch.mean(D_fake) # 

                # 計算 GP by calc_gradient_penalty()
                # gradient_penalty = self.calc_gradient_penalty(expected_q_values_samples, current_q_values_samples, D_noise)
                # 從 GP 改為 LP (by calc_lipschitz_penalty())
                lipschitz_penalty = self.calc_lipschitz_penalty(expected_q_values_samples, current_q_values_samples, D_noise)

                # 計算 Loss_D
                D_loss = D_fake_loss - D_real_loss + lipschitz_penalty
                # update D 的參數
                self.D_model.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

            # Loss_G = -E[D(G^a(s, tau))]
            self.G_model.train()
            # 取得 G 生成的 Q 值機率分布。(fake)，shape : (self.batch_size, self.num_samples)
            current_q_values_samples = self.G_model(batch_state, G_noise)  # 用的是 D 第五次更新所用的那個 batch
            current_q_values_samples = current_q_values_samples.gather(1, batch_action).squeeze(1)
            # G^a(s, tau)
            D_fake = self.D_model(current_q_values_samples, D_noise)
            # E[G^a(s, tau)]
            G_loss = -torch.mean(D_fake)
            # update G 的參數
            self.G_model.zero_grad()
            G_loss.backward()
            # gradient clipping，將梯度限制在 (-1, 1)，防止梯度爆炸，訓練不穩定
            for param in self.G_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.G_optimizer.step()

            # 紀錄 G 和 D 這次的 loss，D 記錄的 loss 是其第五次更新時的 loss
            self.train_hist['G_loss'].append(G_loss.item())
            self.train_hist['D_loss'].append(D_loss.item())

            # target_G 逼近 G 的參數 every target_net_update_freq
            self.update_target_model()

        # print('current q value', current_q_values_samples.mean(1))
        # print('expected q value', expected_q_values_samples.mean(1))


#=============================================================================================================================================#
# 回傳一個 WGAN-GP 的 Action Space : list
def action_space(total, num):  # total = 10 (total_band = 10MHz)，num = 3 (3 types of NS)
    tmp = list(itertools.product(range(total + 1), repeat=num))  # itertools.product() : 產出所有 0~10 中取 3 個數字的所有組合。將 (x, y, z) 存入 tmp
    result = []
    # 從一堆 (x, y, z) 中挑出 x + y + z = 10 的組合存入 result
    for value in tmp:  # 
        if sum(value) == total:
            result.append(list(value))
    result = np.array(result)
    
    [i, j] = np.where(result == 0)  # 找出 result 中那些 [x, y, z] 其中任一為 0 的組合。
    result = np.delete(result, i, axis=0)  # 刪除那些任一為 0 的組合
    # example : 
    # result = [ [3, 3, 4], [2, 4, 4], [5, 5, 0], [6, 4, 0] ]
    # result = np.array(result)
    # np.where(result == 0)
    # return : (array([2, 3]), array([2, 2]))  # 回傳 0 的位址。i = [2, 3] (2 個零所在的列數)，j = [2, 2] (2 個零所在的行數)
    # print(result.shape)

    return result  # x + y + z = 10，且任一不為 0

#=============================================================================================================================================#
# 進入神經網路前的狀態前處理，將 state 中的值做正規化，使整個 state 各元素的均值為 0、標準差為 1
def state_update(state, ser_cat):  # state : 當前 Learning window 各網路切片要傳送的封包個數 [d0, d1, d2]
    discrete_state = np.zeros(state.shape)
    # 若 state 內皆為 0，則不用進行前處理，輸入的狀態為一個零矩陣
    # **但這邊只有有任一元素不為 0 時都不應該回傳 0，故這邊應該改為 state.sum() == 0 (已修正)
    # if state.all() == 0:  
    #     return discrete_state
    if state.sum() == 0: return discrete_state
    # 若 state 內有任一不為 0，則需做正規化
    for ser_name in ser_cat:  # 一次考慮一種網路切片
        ser_index = ser_cat.index(ser_name)
        discrete_state[ser_index] = state[ser_index]
    discrete_state = (discrete_state - discrete_state.mean()) / discrete_state.std()  # 對整個 state 做 z-score 正規化
    return discrete_state

#=============================================================================================================================================#
# Reward function，有做 reward clipping

# def calc_reward(qoe, se, low, high):
#     utility = np.matmul(qoe_weight, qoe.reshape((3, 1))) + se_weight * se
#     if utility < low:
#         reward = -1
#     elif utility > high:
#         reward = 1
#     else:
#         reward = 0
#     return utility, reward

# qoe -> 三種網路切片在整個 learning window 的 SSR
# se  -> 整個 learning window 中的平均每個 timeslot 的 SE
# threshold -> 當前 Learning window 對模型的利用率要求，會隨時間單調上升
def calc_reward(qoe, se, threshold):
    # 依照權重算出 utility
    utility = np.matmul(qoe_weight, qoe.reshape((3, 1))) + se_weight * se 
    
    # # 這演算法的 threshold 是會隨時間而單調上升的，故要限制 threshold
    # threshold = 3.5 + 1.5 * frame / (total_timesteps / 1.5)
    # if threshold > 5.5:
    #     threshold = 5.5

    # # reward clipping
    # if utility < threshold:
    #     reward = 0
    # else:
    #     reward = 1

    # 照論文參數設定 (URLLC 小封包)
    threshold1 = 6.5
    threshold2 = 4.5
    if utility >= threshold1: reward = 1
    elif utility < threshold1 and utility > threshold2: reward = 0
    else: reward = -1
    return utility, reward

#=============================================================================================================================================#
# WGAN-GP 的 Generator (G_model) 選擇動作 by ɛ-greedy
# model -> 網路、s -> state (已經過前面的 state_update 狀態前處理，(3))、z -> quantile embedding (tau 組成的 vector)、eps -> epsilon
def get_action(model, s, z, eps, device):
    if np.random.random() >= eps:
        # X.shape (1, 3)
        X = torch.tensor(s).unsqueeze(0).to(torch.float).to(device)
        a = model.G_model(X, z).squeeze(0).mean(1).max(0)[1]
        # print(a)  
        return a.item()  # from tensor to aboriginal python data type
    else:
        return np.random.randint(0, model.num_actions)

#=============================================================================================================================================#
def plot_rewards(rewards):
    plt.figure(1)
    plt.clf()
    rewards = np.array(rewards)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards.cumsum())
    plt.pause(0.001)

#=============================================================================================================================================#
# moving average
def moving_average(data, window_size):
    data = np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

#=============================================================================================================================================#
#%%
# 訓練過程

# 設定 GPU
# torch.cuda.manual_seed(100)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')

# 設定 epsilon
# Create the schedule for exploration starting from 1.
# exploration_final_eps = 0.02
# exploration_fraction = 0.3
# exploration_start = 0.
# exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
#                                 start_timesteps = int(total_timesteps * exploration_start),
#                                 initial_p=1.0,
#                                 final_p=exploration_final_eps)
#epsilon variables
epsilon_start    = 1.0
epsilon_final    = 0.01
epsilon_decay    = 3000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# plt.ion()  # 開啟圖片互動模式，可以在訓練時即時更新圖片

# 總共會有 10000 個 learning windows
# 一個 learning windows = 2000 timeslot (0.5ms) = 1s
total_timesteps = 10
# parameters of celluar environment
ser_cat_vec = ['volte', 'embb_general', 'urllc']
band_whole_no = 10 * 10**6  # 10MHz
band_per = 1 * 10**6  # bandwidth allocation resolution : 1MHz
qoe_weight = [1, 1, 1]
se_weight = 0.01
dl_mimo = 64  # MIMO 天線數
learning_window = 2000  # 一個 episode
env = cellularEnv(ser_cat = ser_cat_vec, learning_windows = learning_window, dl_mimo = dl_mimo)
env.countReset()  # 初始化各計數器 (每個 learning window 都會重置一次)
env.activity()  # 開始第一個 timeslot，指派各 UE readtime，並依照 readtime 決定是否要新增封包

# 設定 action_space
action_space = action_space(10, 3) * band_per
num_actions = len(action_space)
# print(num_actions)  # 36

# 設定 model
# static_policy = False -> 代表現在是 train。
# num_input = state_size = 3。一個 state 是由三個數字組成 (三種網路切片每秒要傳送的封包個數)，num_actions = 36
model = WGAN_GP_Agent(static_policy = False, num_input = 3, num_actions = num_actions)
G_noise = (torch.rand(1, model.num_samples)).to(device)
observation = state_update(env.tx_pkt_no, env.ser_cat)  # 進行狀態前處理
# print(f"obeservation shape : {observation.shape}")

log = {}
observations = []
actions = []  # 紀錄每個 learning window 的 action
rewards = []  # 紀錄每個 learning window 的 reward
utilities = [0.]  # 紀錄每個 learning window 的 utility
SE = []  # 紀錄每個 learning window 的 SE (一個 Learning window 平均一個 timeslot 的 SE)
QoE = []  # 紀錄每個 learning window 的 QoE (SSR) (一個 Learning window 中滿足要求傳送的封包數 / 一個 learning window 要傳送的封包數)

# 跑 10000 個 learning windows
for frame in tqdm(range(1, total_timesteps + 1)):
    # 產生該 learning windows 要用的 epsilon
    epsilon = epsilon_by_frame(frame)
    
    # 分上層 (每 1s 做一次)，做 1 次
    # Select and perform an action
    observations.append(observation.tolist())  # 整個資料結構都變成 list
    action = get_action(model, observation, G_noise, epsilon, device)  # 根據 ε-greedy 選出 action
    actions.append(action)
    env.band_ser_cat = action_space[action]  # 將各網路接片的資源分配結果存入 env
    prev_observation = observation

    # 上層分完，分下層 (每 0.5ms 做一次)，做 200 次
    # i 就是一個 timeslots (0.5ms)
    for i in itertools.count():  # itertools.count() : 從 0 開始到無限
        env.scheduling()  # 每一個 timeslots 做一次下層分配
        env.provisioning()  # 根據 UE 分到的 RB 個數算出當前 timeslot 的 SE、SSR
        if i == (learning_window - 1):  # 分 1999 次 (下 1s 開始前的一個 timeslot)
            break
        else:
            env.bufferClear()  # 維護使用者對應的 Queue (還沒分配出去的封包繼續留在 buffer 中)
            env.activity()  # 指派 UE readtime，並依照 readtime 決定是否產生封包
    
    # 做完一整個 learning window，算這個 learning window 的 reward
    qoe, se = env.get_reward()  # 該 Learning window 中滿足要求傳送出去的封包 / 總封包數, 該 learning window 中平均一個 timeslot 的 SE
    # utility, reward = calc_reward(qoe, se, 3, 5.7)
    threshold = 3.5 + 1.5 * frame / (total_timesteps / 1.5)  # threshold -> 當前 learning window 模型預計要達到的標準 (隨時間單調上升)，達到才有 reward = 1
    utility, reward = calc_reward(qoe, se, threshold)  # 根據 threshold 算出當前 learning window 得到的 reward
    
    # 紀錄相關結果
    QoE.append(qoe.tolist())
    SE.append(se[0])
    rewards.append(reward)
    utilities.append(utility)

    # 準備做下一次的上層，取出 state
    # 該 state 為上一個 learning window 中，各網路切片欲傳輸的封包總數
    observation = state_update(env.tx_pkt_no, env.ser_cat)
    
    # 將 experience 存入 replay buffer
    model.append_to_replay(prev_observation, action, reward, observation)

    # 更新模型
    model.update(frame)
    
    # 結束一個 learning window，重設相關計數器
    env.countReset()
    # 設定各 UE readtime，並根據該 readtime 決定是否新增封包
    env.activity()
    
    
    print(f'\nGANDDQN=====episode: {frame}, epsilon: {epsilon:.3f}, utility: {utility}, reward: {reward:.5f}')
    print(f'qoe: volte = {qoe[0]}, video = {qoe[1]}, urllc = {qoe[2]}')
    print('bandwidth-allocation solution', action_space[action])

    # 每個 learning window 結束後都更新一次 reward 圖
    # plot_rewards(rewards)
    
    # 每 200 個 learning window 紀錄一次
    # if frame % 200 == 0:
        # print('frame index [%d], epsilon [%.4f]' % (frame, epsilon))
        # model.save_w()
        # log['state'] = observations
        # log['action'] = actions
        # log['SE'] = SE
        # log['QoE'] = QoE
        # log['reward'] = rewards

        # f = open('./log/GANDDQN/log_10M_1M_LURLLC.txt', 'w')
        # f.write(json.dumps(log))
        # f.close()
    
print('Complete')

#%%
# QoE_volte, embb, urllc 為長度 10000 的 list
qoe_volte = [v for (v, e, u) in QoE]
qoe_embb = [e for (v, e, u) in QoE]
qoe_urllc = [u for (v, e, u) in QoE]

# utilities_ 長度 10000 的 list
utilities = utilities[1: ]  # 去除第一筆
utilities_ = [u.item() for u in utilities]

# SE 長度 10000 的 list



#%%
# show figures of each metrics

ma_qoe_volte = moving_average(qoe_volte, window_size = 200)
ma_qoe_embb = moving_average(qoe_embb, window_size = 200)
ma_qoe_urllc = moving_average(qoe_urllc, window_size = 200)
ma_SE = moving_average(SE, window_size = 200)
ma_utility = moving_average(utilities_, window_size = 200)

# loss figure (figure(2))
model.plot_loss()
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/GAN_DDQN/loss.png")

# qoe figure (figure(3))
plt.figure(3)
plt.clf()
plt.title('QoE')
plt.xlabel('Episode')
plt.ylabel('SLA Satisfication Rate')
plt.plot(ma_qoe_volte)
plt.plot(ma_qoe_embb)
plt.plot(ma_qoe_urllc)
plt.legend(["VoLTE", "Video", "URLLC"])
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/GAN_DDQN/QoE.png")

# se figure (figure(4))
plt.figure(4)
plt.clf()
plt.title('SE')
plt.xlabel('Episode')
plt.ylabel('bits/Hz')
plt.plot(ma_SE)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/GAN_DDQN/SE.png")

# utility figure (figure(5))
plt.figure(5)
plt.clf()
plt.title('Utility')
plt.xlabel("Episode")
plt.ylabel("utility")
plt.plot(ma_utility)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/GAN_DDQN/Utility.png")


# %%
