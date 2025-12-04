import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from Env.env_fixedUE import cellularEnv  # GANDDQN 環境 (不考慮使用者移動、考慮 100 人)
from Env.env_movingUE import EnvMove  # LSTM 環境 (考慮使用者移動、考慮 1200 人)
from Diffusion_utils.diffusion import Diffusion
from Diffusion_utils.D2AC_opt import D2AC_OPT
from Diffusion_utils.model import GDM, DoubleCritic
from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer
from gymnasium.spaces import Discrete, Box  # In order to use BasePolicy
from pathlib import Path
from pprint import pprint
from seed import set_seed

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 環境參數
set_seed(seed= 123)
fixed_UE = True  # True if using GANDDQN env, False if LSTM_A2C env
if fixed_UE: print("\n================================================== GANDDQN_env ==================================================\n")
else: print("\n================================================== LSTM-A2C_env ==================================================\n")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


'''改善的地方
* clamp(-1, 1) 的作法可能過於強硬，可以改為使用 tanh
    * 但這樣 1 & -1 的地方梯度會接近 0，所以改回 clamp (-1, 1) 讓邊界也保有該有的梯度
* def get_action() 那個地方把他從使用先 abs 再 normalize (-5 & 5 有一樣的分配策略，模型可能會混亂)，改為直接套 softmax。
* (未改善) 可以改善下層的做法，把剩的、當不了一塊 RB 的資源整合起來平均分給各個網路切片
* QoE 應該要以傳送出的 bits 為主，而非封包數，因為單靠封包數沒辦法體現出個網路切片的 loading。故改用各網路切片所需傳送的 bits 數作為狀態。
* 改掉狀態欲處理的做法，改用 max_scaling，保留各網路切片的大小關係。
'''

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 設定圖片 / log 路徑
algo_name = 'D2AC'
exp_name = 'exp3'
log_file = 'Logs_movingUE_env' if fixed_UE == False else 'Logs_fixedUE_env'
log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") /log_file / algo_name / exp_name / 'tensorboard'
# generate log writer
writer = SummaryWriter(log_dir= log_path)

# 要看 tensorboard 結果，輸入在 terminal 中他會給你一個網址
# tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/"algo_name"/"exp_name"/tensorboard"
# tensorboard --logdir "/home/super_trumpet/NCKU/Paper/My Methodology/Logs/D2AC/exp1/tensorboard"
# 程式跑下去之後就可以用另一個 terminal 開啟 tensorboard，接著你任何時候想看進度就去點一下 tensorboard 頁面的重置就好了

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# state : np.array, shape (state_dim)
# ser_cat : list, len = 3
# return preprocessed state : np.array, shape (state_dim)
# def state_preprocessing(state):
#     preproc_state = np.zeros(state.shape)
#     # if state = [0, 0, 0], then return np.array([0, 0, 0])
#     if state.sum() == 0: return preproc_state
#     # if any element in state is not 0, then normalize the state (z-score normalization)
#     else: 
#         preproc_state = state.copy()
#         return (preproc_state - preproc_state.mean()) / preproc_state.std()

# 改用 max-scaling (讓輸出介於 [0~1])
# 因為用 z-score 沒辦法體現當前流量的負載是忙碌還是很輕鬆，乍看之下根本環境沒差，但其實有
# ex:
# 負載輕鬆 : slice A 需要 1 單位，slice B 需要 9 單位資源 -> Z-score 視角：A 很小 (負值)，B 很大 (正值)。模型決定給 Slice A 10% (1 MHz)，Slice B 90% (9 MHz) -> ok
# 負載很大 : slice A 需要 100 單位，slice B 需要 900 單位 -> Z-score 只看分佈，這兩個數字經過正規化後，會跟場景 1 幾乎一模一樣！ 模型看到 A 很小 (負值)，B 很大 (正值)。
#           模型決策：模型回想起場景 1 的成功經驗，再次決定給 Slice A 10% (1 MHz)，Slice B 90% (9 MHz)。
#           Slice A 這次負載很重，它至少需要 2 MHz 才能活命 (SLA 門檻)，結果你只給它 1 MHz。
#           結局：Slice A 直接死亡 (SSR=0)。
# 但觀察後發現本實驗環境中沒有這種情況發生，每個 learning window 的 loading 都差不多。不會有突然暴衝的情況發生。
def state_preprocessing(state):
    Max_ = 10000000
    preproc_state = np.zeros(state.shape)
    if state.sum() == 0:  return preproc_state
    else: 
        preproc_state = state.copy()
        preproc_state = preproc_state / Max_
    return preproc_state
        

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 回傳各網路切片所分到的頻寬量 (Hz)
# state : preprocessed state, np.array, shape (state_dim)
# total_bandwidth : int, 10* 10**6 (Hz)
# return logit (np.array with shape (action_dim) , real_action (np.array with shape (action_dim))
def get_actions(state, total_band, model, device):
    state = torch.from_numpy(state).reshape(1, state_dim).to(dtype= torch.float32, device= device)  # shape (batch_size(1), state_dim)
    with torch.no_grad():
        action_logit = model(state= state)  # shape (batch_size(1), action_dim)
    
    # 作法一 : 取絕對值後正規化 -> 不然 -5 & 5 會得到相同的分配，模型會錯亂
    # action = torch.abs(action_logit).squeeze(dim= 0).numpy()  # np.array with shape (action_dim)
    # normalized_action = action / np.sum(action)
    # 作法二 : 將模型的輸出使用 softmax，至少 -5 & 5 會有合理的分配
    proportion = torch.nn.functional.softmax(action_logit, dim= 1).cpu().numpy().squeeze()
    real_action = total_band * proportion
    action_logit = action_logit.cpu()
    return action_logit.cpu().numpy().squeeze(), real_action

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# cal reward based on utility = \alpha * SE + (\betas * SSRs).sum() after a learning window
# qoe : SSRs of 3 NS of a complete learning window, np.array, shape (3)
# qoe_weights : list, len = 3
# se : average SE of a timeslot of a complete learning window, np.array, shape (1)
# se_weight : no
# reward_clipping : clip the reward or not
# return utility, reward, float (np.array with shape ())
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
denoise_step = 6
actor_lr = 0.001
critic_lr = 0.001
weight_decay = 0
prioritized_replay = False  # use Prioritized ReplayBuffer or not
buffer_size = 50000  # set in GANDDQN
batch_size = 32  # set in GANDDQN
prior_alpha = 0.4  # used in prioritized replay buffer
prior_beta = 0.4  # used in prioritized replay buffer


# log params
note = '使用 reconstruction loss'
hparams_dict = {
    'denoise step' : denoise_step,
    'actor_lr' : actor_lr,
    'critic_lr' : critic_lr,
    'weight_decay' : weight_decay,
    'buffer_size' : buffer_size,
    'batch_size' : batch_size,
    'note' : note
}

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
# ser_cat = ['volte', 'embb_general', 'urllc']
total_band = 10 * 10**6  # 1MHz
# J = \alpha * SE + \betas * SSRs
qoe_weights = [1, 1, 1]  # \betas
se_weight = 0.01  # \alpha
total_timesteps = 10000  #  10000 in GAN_DDQN & LSTM_A2C learning_windows (episodes)
learning_windows = 2000  # 1 learning window (episode) = 2000 timeslots
dl_mimo = 64
UE_no = 100 if fixed_UE else 1200
if fixed_UE: env = cellularEnv(ser_cat= ser_cat, learning_windows= learning_windows, dl_mimo= dl_mimo, UE_max_no= UE_no)
else: env = EnvMove(UE_max_no= UE_no, ser_prob= np.array([1, 2, 3], dtype= np.float32), learning_windows= learning_windows, dl_mimo= dl_mimo)
env.countReset()  # reset 所有計數器
if not fixed_UE: env.user_move()  # user move in LSTM-A2C env
env.activity()  # 所有 UE 開始根據其網路切片產生封包
# observation_packets : total packets of each NSs, np.array with shape (3)
# observation_bits : total bits of each NSs, np.array with shape (3)
observation_packets, observation_bits = env.get_state()  

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# recording lists
QoEs = []
SEs = []
Utilities = []
Rewards = []
Observations = []
Actor_losses = []
Critic_losses = []

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# training
for frame in tqdm(range(1, total_timesteps+1)):
    print(f"\n\n******Episode {frame} :")
    # state is the loading (no. of packets) of each NS of the previous learning window
    state = state_preprocessing(state= observation_bits)  
    # print(f"observation_packets = {observation_packets}, observation_bits = {observation_bits}")
    # print(f"state = {state}")

    # action_logit : Actor 輸出 torch.tensor with shape (batch_size(1), action_dim), values are within the range(-1, 1)
    # real_action : 將 logit 轉為真實動作，即各網路切片的分配到的頻寬 (Hz)。np.array with shape (3)
    action_logit, real_action = get_actions(state= state, total_band= total_band, model= actor, device= device)
    # print(f"action_logit = {action_logit}, real action = {real_action}")
    # print(f"action = {real_action}")
    # assign to the env.
    env.band_ser_cat = real_action
    # print(env.band_ser_cat)  # ex: [3442405.76028824 3145710.52789688 3411883.71181488]
    
    # 2000 slots in 1 learning window
    for i in range(learning_windows):
        env.scheduling()  # do lower-level allocation every timeslots
        env.provisioning()  # evaluate the SE & SSR of the current timeslot
        env.activity()  # assign readtime & generate packet according to the readtime

    # calculate the reward of the current learning window
    qoe, se = env.get_reward()
    # use qoe & se to calculate utility as a reward
    # utility = \alpha * SE + (\betas * SSRs).sum()
    utility, reward = cal_reward(qoe= qoe, se= se, qoe_weights= qoe_weights, se_weight= se_weight, reward_clipping= False)


    # print(f"qoe = {qoe}, se = {se}, utility = {utility}, reward = {reward}")

    # Record the values of the current learning window
    QoEs.append(qoe.tolist())  # qoe.tolist() -> [qoe1, qoe2, qoe3]
    SEs.append(se.tolist()[0])  # se.tolist() -> [se]
    Rewards.append(reward.item())  
    Utilities.append(utility.item())

    # store the experience to the ReplayBuffer
    data = Batch(
        obs= state,  # np.array with shape (3)
        act = action_logit,  # np.array with shape (3)
        rew = reward.squeeze(),  # int
        terminated= False,
        truncated= False,
        obs_next= state_preprocessing(env.get_state()[1])  # np.array with shape (3)
    )
    buffer.add(data)
    
    # update the model after warming up
    if len(buffer) > batch_size * 30:
        loss = d2ac_opt.update(sample_size= batch_size, buffer= buffer)
        pprint(f"loss = {loss}")
        writer.add_scalar(tag= 'loss/actor_loss', scalar_value= loss['actor_loss'].item(), global_step= frame)
        writer.add_scalar(tag= 'loss/policy_loss', scalar_value= loss['policy_loss'].item(), global_step= frame)
        writer.add_scalar(tag= 'loss/recon_loss', scalar_value= loss['recon_loss'].item(), global_step= frame)
        writer.add_scalar(tag= 'loss/critic_loss', scalar_value= loss['critic_loss'].item(), global_step= frame)
    # Actor_losses.append(loss['actor_loss'].item())
    # Critic_losses.append(loss['critic_loss'].item())


    # print the outcome of the current learning window
    print(f"qoe = {qoe}, se = {float(se[0]):.3f}, reward = {float(reward[0]):.3f}, utility = {float(utility[0]):.3f}")
    # print(f"QoEs = {QoEs}, SEs = {SEs}, Rewards = {Rewards}, Utilities = {Utilities}")
    
    writer.add_scalar(tag= 'observationBits/volte', scalar_value= observation_bits[0], global_step= frame)
    writer.add_scalar(tag= 'observationBits/embb_general', scalar_value= observation_bits[1], global_step= frame)
    writer.add_scalar(tag= 'observationBits/urllc', scalar_value= observation_bits[2], global_step= frame)
    writer.add_scalar(tag= 'qoe/volte', scalar_value= qoe[0], global_step= frame)
    writer.add_scalar(tag= 'qoe/embb_general', scalar_value= qoe[1], global_step= frame)
    writer.add_scalar(tag= 'qoe/urllc', scalar_value= qoe[2], global_step= frame)
    writer.add_scalar(tag= 'se', scalar_value= se[0], global_step= frame)
    writer.add_scalar(tag= 'reward', scalar_value= reward[0], global_step= frame)
    writer.add_scalar(tag= 'utility', scalar_value= utility[0], global_step= frame)
    
    # gain next state (loading of each NS in the previous learning window)
    observation_packets, observation_bits = env.get_state()

    # reset all counters after each learning window
    env.countReset()

    # if using the env of LSTM-A2C then move the users
    if not fixed_UE: env.user_move()

    # start next learning window (start to generate new packet)
    env.activity()

metric_dict = {}
writer.add_hparams(hparam_dict= hparams_dict, metric_dict= metric_dict)

print("Complete")

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
# ma_actor_loss = moving_average(Actor_losses, window_size= 200)
# ma_critic_loss = moving_average(Critic_losses, window_size= 200)

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
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/D2AC/exp3/QoE.png")

# se figure (figure(4))
plt.figure(4)
plt.clf()
plt.title('SE')
plt.xlabel('Episode')
plt.ylabel('bits/Hz')
plt.plot(ma_SE)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/D2AC/exp3/SE.png")

# utility figure (figure(5))
plt.figure(5)
plt.clf()
plt.title('Utility')
plt.xlabel("Episode")
plt.ylabel("utility")
plt.plot(ma_utility)
plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/D2AC/exp3/Utility.png")

# loss figure (figure(6))
# plt.figure(6)
# plt.clf()
# plt.title('Loss')
# plt.xlabel("Episode")
# plt.ylabel("loss")
# plt.plot(Actor_losses, label= 'actor_loss')
# plt.plot(Critic_losses, label= 'critic_loss')
# plt.legend()
# plt.savefig("/home/super_trumpet/NCKU/Paper/My Methodology/Outcome/D2AC/Losses.png")

print("Graph Saved")
