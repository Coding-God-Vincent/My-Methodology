#%%
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv
from Diffusion_utils.diffusion import Diffusion
from Diffusion_utils.D2AC_opt import D2AC_OPT
from Diffusion_utils.model import GDM, DoubleCritic
from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer
from gymnasium.spaces import Discrete, Box  # In order to use BasePolicy

'''改善的地方
* clamp(-1, 1) 的作法可能過於強硬，可以改為使用 tanh
* (未改善) 可以改善下層的做法，把剩的、當不了一塊 RB 的資源整合起來平均分給各個網路切片
*
'''


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# state : np.array, shape (state_dim)
# ser_cat : list, len = 3
# return preprocessed state : np.array, shape (state_dim)
def state_preprocessing(state):
    preproc_state = np.zeros(state.shape)
    # if state = [0, 0, 0], then return np.array([0, 0, 0])
    if state.sum() == 0: return preproc_state
    # if any element in state is not 0, then normalize the state (z-score normalization)
    else: 
        preproc_state = state.copy()
        return (preproc_state - preproc_state.mean()) / preproc_state.std()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 回傳各網路切片所分到的頻寬量 (Hz)
# state : preprocessed state, np.array, shape (state_dim)
# total_bandwidth : int, 10* 10**6 (Hz)
# return logit (torch.tensor with shape (batch_size(1), action_dim)) , real_action (np.array with shape (action_dim))
def get_actions(state, total_band, model, device):
    state = torch.from_numpy(state).reshape(1, state_dim).to(dtype= torch.float32, device= device)  # shape (batch_size(1), state_dim)
    with torch.no_grad():
        action_logit = model(state= state)  # shape (batch_size(1), action_dim)
    action_logit = action_logit.cpu()
    action = torch.abs(action_logit).squeeze(dim= 0).numpy()  # np.array with shape (action_dim)
    normalized_action = action / np.sum(action)
    real_action = total_band * normalized_action
    return action_logit, real_action

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# cal reward based on utility = \alpha * SE + (\betas * SSRs).sum() after a learning window
# qoe : SSRs of 3 NS of a complete learning window, np.array, shape (3)
# qoe_weights : list, len = 3
# se : average SE of a timeslot of a complete learning window, np.array, shape (1)
# se_weight : no
# reward_clipping : clip the reward or not
# return utility, reward, int
def cal_reward(qoe, se, qoe_weights, se_weight, reward_clipping= False):
    utility = np.matmul(qoe_weights, qoe.reshape((3, 1))) + se_weight * se
    if reward_clipping: 
        threshold1 = 6.5
        threshold2 = 4.5
        if utility >= threshold1: reward = 1
        elif utility < threshold1 and utility > threshold2: reward = 0
        else: reward = -1
    else: reward = utility
    return utility, reward

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# np.convolve(data, kernel= np.ones(window_size) / window_size, mode= 'valid')，用 kernel 掃過整個 data (stride = 1)
# kernel : if window_size = 3, then kernel = [1/3, 1/3, 1/3]. 可以想成是每一個資料所佔的比例
# mode= 'valid'，不做 padding，只對完整的 window 做 moving average
def moving_average(data, window_size):
    data = np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode= 'valid')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# set the device & some parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ser_cat = ['volte', 'embb_general', 'urllc']

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# training parameters
state_dim = len(ser_cat)
action_dim = len(ser_cat)
max_action = 1
beta_schedule = 'vp'  # 'vp', 'cosin', 'linear'
denoise_step = 5
actor_lr = 0.001
critic_lr = 0.001
weight_decay = 0
prioritized_replay = False  # use Prioritized ReplayBuffer or not
buffer_size = 2000  # set in GANDDQN
batch_size = 32  # set in GANDDQN
prior_alpha = 0.4  # used in prioritized replay buffer
prior_beta = 0.4  # used in prioritized replay buffer

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# generate the models
gdm = GDM(state_dim= state_dim, action_dim= action_dim)
actor = Diffusion(
    state_dim= state_dim,
    action_dim= action_dim,
    model= gdm,
    max_action= max_action,
    beta_schedule= beta_schedule,
    denoise_steps= denoise_step,
    clip_denoised= True,
    device= device
).to(device= device)
actor_optim = torch.optim.AdamW(
    # Diffusion inherits nn.Module, so actor.parameters() will be redirect to the parameters of all nn.Modules included in actor
    params= actor.parameters(),  
    lr= actor_lr,
    weight_decay= weight_decay
)

critic = DoubleCritic(state_dim= state_dim, action_dim= action_dim).to(device= device)
critic_optim = torch.optim.AdamW(
    params= critic.parameters(),
    lr= critic_lr,
    weight_decay= weight_decay
)

# generate the ReplayBuffer
if prioritized_replay: 
    buffer = PrioritizedReplayBuffer(
        size= buffer_size,
        # used to control the strength of the prioritization (alpha = 0 : uniform, alpha = 1 : complete prioritized)
        alpha= prior_alpha,
        # used to control the strength of revision of the sampling bias
        beta= prior_beta
    )
else: buffer = ReplayBuffer(size= buffer_size)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# generate an instance of D2AC_OPT to handle the update of the model
fake_action_space = Discrete(3)
fake_action_space = Box(low= -1, high= 1, shape= (3,))
d2ac_opt = D2AC_OPT(
    state_dim= state_dim,
    action_dim= action_dim,
    actor= actor,
    actor_optim= actor_optim,
    critic= critic,
    critic_optim= critic_optim,
    device= device,
    n_steps= 3,
    # 以下參數會放在 **kwargs，放一些用不到但 BasePolicy 規定要放的參數
    action_space= fake_action_space
)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# generate the env
total_band = 10 * 10**6  # 1MHz
# J = \alpha * SE + \betas * SSRs
qoe_weights = [1, 1, 1]  # \betas
se_weight = 0.01  # \alpha
total_timesteps = 2  # 10000 learning_windows (episodes)
learning_windows = 2000  # 1 learning window (episode) = 2000 timeslots
dl_mimo = 64
env = cellularEnv(ser_cat= ser_cat, learning_windows= 2000, dl_mimo= dl_mimo)
env.countReset()  # reset 所有計數器
env.activity()  # 所有 UE 開始根據其網路切片產生封包
observation = env.get_state()  # init. state, np.array with shape (3)

# recording lists
QoEs = []
SEs = []
Utilities = []
Rewards = []
Observations = []
Actor_losses = []
Critic_losses = []

for frame in tqdm(range(1, total_timesteps+1)):

    # state is the loading (no. of packets) of each NS of the previous learning window
    # print(f"\nstate : {env.get_state()}")
    state = state_preprocessing(state= observation)  
    # action_logit : Actor 輸出 torch.tensor with shape (batch_size(1), action_dim), values are within the range(-1, 1)
    # real_action : 將 logit 轉為真實動作，即各網路切片的分配到的頻寬 (Hz)。np.array with shape (3)
    action_logit, real_action = get_actions(state= state, total_band= total_band, model= actor, device= device)
    # print(f"action = {real_action}")
    # assign to the env.
    env.band_ser_cat = real_action
    # print(env.band_ser_cat)  # ex: [3442405.76028824 3145710.52789688 3411883.71181488]
    
    # 2000 slots in 1 learning window
    for i in range(learning_windows):
        env.scheduling()  # do lower-level allocation every timeslots
        env.provisioning()  # evaluate the SE & SSR of the current timeslot
        env.bufferClear()  # update the Queue of each UE
        env.activity()  # assign readtime & generate packet according to the readtime

    
    # calculate the reward of the current learning window
    qoe, se = env.get_reward()
    # use qoe & se to calculate utility as a reward
    # utility = \alpha * SE + (\betas * SSRs).sum()
    utility, reward = cal_reward(qoe= qoe, se= se, qoe_weights= qoe_weights, se_weight= se_weight, reward_clipping= False)

    # Record the values of the current learning window
    QoEs.append(qoe.tolist())
    SEs.append(se.tolist()[0])
    Rewards.append(reward.item())
    Utilities.append(utility.item())
    print(f"\nqoe = {qoe}, se = {se}, reward = {reward}, utility = {utility}")
    # print(f"QoEs = {QoEs}, SEs = {SEs}, Rewards = {Rewards}, Utilities = {Utilities}")

    # store the experience to the ReplayBuffer
    buffer.add(Batch(
        obs= observation,  # np.array with shape (3)
        act = action_logit.numpy().squeeze(),  # np.array with shape (3)
        rew = reward.squeeze(),  # int
        terminated= False,
        truncated= False,
        obs_next= env.get_state()  # np.array with shape (3)
    ))

    # update the model
    loss = d2ac_opt.update(sample_size= batch_size, buffer= buffer)
    Actor_losses.append(loss['actor_loss'])
    Critic_losses.append(loss['critic_loss'])
    print(f"{loss}\n")
    print(Actor_losses, Critic_losses)

    # gain next state (loading of each NS in the previous learning window)
    observation = env.get_state()


    # reset all counters after each learning window
    env.countReset()

    # start next learning window (start to generate new packet)
    env.activity()

print("Complete")

#%%
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Generate Outcome Figures

# QoE_volte, embb, urllc 為長度 10000 的 list
qoe_volte = [v for (v, e, u) in QoEs]
qoe_embb = [e for (v, e, u) in QoEs]
qoe_urllc = [u for (v, e, u) in QoEs]

# utilities_ 長度 10000 的 list
Utilities = Utilities[1: ]  # 去除第一筆
Utilities_ = [u for u in Utilities]

# use moving average to smooth the curve
ma_qoe_volte = moving_average(qoe_volte, window_size = 200)
ma_qoe_embb = moving_average(qoe_embb, window_size = 200)
ma_qoe_urllc = moving_average(qoe_urllc, window_size = 200)
ma_SE = moving_average(SEs, window_size = 200)
ma_utility = moving_average(Utilities_, window_size = 200)
ma_actor_loss = moving_average(Actor_losses, window_size= 200)
ma_critic_loss = moving_average(Critic_losses, window_size= 200)

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
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/D2AC/QoE.png")

# se figure (figure(4))
plt.figure(4)
plt.clf()
plt.title('SE')
plt.xlabel('Episode')
plt.ylabel('bits/Hz')
plt.plot(ma_SE)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/D2AC/SE.png")

# utility figure (figure(5))
plt.figure(5)
plt.clf()
plt.title('Utility')
plt.xlabel("Episode")
plt.ylabel("utility")
plt.plot(ma_utility)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/D2AC/Utility.png")

# loss figure (figure(6))
plt.figure(6)
plt.clf()
plt.title('Loss')
plt.xlabel("Episode")
plt.ylabel("loss")
plt.plot(Actor_losses, label= 'actor_loss')
plt.plot(Critic_losses, label= 'critic_loss')
plt.legend()
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/D2AC/Losses.png")
# %%
