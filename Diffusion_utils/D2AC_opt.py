from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy  # deepcopy can copy "everything" and generate a new one
from .helpers import GaussianNoise
import numpy as np

'''About Inherit BasePolicy
只要繼承 tianshou 的 BasePolicy 就一定要有以下幾個函式 : 
1. forward() : 傳入一個 batch，算出各 state 輸出的 Action。return tensor with shape (batch_size, action_dim)
2. process_fn() : 傳入一個 batch，算出該 batch 中各資料的 Target Q (用於更新 Twin Q network)。return Batch (多了一個屬性 : Batch.returns)
3. learn() : 用一個 batch 更新所有的網路
4. update() : 傳入 replay buffer，從中抽取出一個 batch 之後，利用該 batch 更新所有的網路 (用到 process_fn() 產出新 batch 後傳入 learn())
'''

class D2AC_OPT(BasePolicy):
    def __init__(
        self,
        state_dim : int, 
        action_dim : int,  
        actor : nn.Module,  # instance of Diffusion() (contain model : GDM)
        actor_optim : torch.optim.Optimizer,
        critic : nn.Module,  # model : Double Critic
        critic_optim : torch.optim.Optimizer,
        device : torch.device,
        tau : float = 0.005,  # soft_update param.
        gamma : float = 1.0,
        reward_normalization : bool = False,
        n_steps : int = 1,  # use n_step return as Target
        lr_decay : bool = False,
        lr_max_step : int = 1000,  # steps to decay lr
        # noise ~ std of gaussian dist. noise will be added to action to enhance exploration
        exploration_noise : float = 0.1,
        **kwargs : any
    ):
        super().__init__(**kwargs)
        self.actor = actor
        self.target_actor = deepcopy(actor) 
        self.target_actor.eval()  # set target network to evaluation mode
        self.actor_optim = actor_optim
        self.critic = critic
        self.target_critic = deepcopy(critic)
        self.target_critic.eval()  # set target network to evaluation mode
        self.critic_optim = critic_optim
        
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.reward_normalization = reward_normalization
        self.n_steps = n_steps
        self.lr_decay = lr_decay
        self.lr_max_step = lr_max_step
        self.noise_generator = GaussianNoise(sigma= exploration_noise)
        
        # if we want to decay the lr, use CosineAnnealingLR
        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optim, T_max= lr_max_step, eta_min= 0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optim, T_max= lr_max_step, eta_min= 0.)
            
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 batch 的資料，根據 obs/obs_ 傳入 actor/target_actor 後得到該 batch 各資料的 act，並把結果放到一個新的 batch 後回傳該新 batch
    # state : str  # indicate to use obs or obs_next
    # model : str  # indicate to use actor or target_actor
    # return : 計算的結果，用一個 Batch 裝起來
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    '''any Instance of D2AC_OPT will redirect to this function (like forward in nn.Module)'''
    def forward(
        self,
        batch : Batch,
        state : str = 'obs',  # can be 'obs' or 'obs_next'
        model : str = 'actor'  # can be 'actor' or 'target_actor'
    ):
        # extract obs/obs_next
        # shape (batch_size, state_dim)
        state = to_torch(batch[state], dtype= torch.float32, device= self.device)
        # choose the model
        model_ = self.actor if model == 'actor' else self.target_actor
        # feed the state to the selected model and gain action logits
        # shape (batch_size, action_dim)
        # clampped but not convert to the actual action (proportion of the resource) yet
        logits = model_(state= state)

        # There's 10% chance of adding noise to the action
        if np.random.rand() < 0.1:
            noise = to_torch(self.noise_generator.generate(logits.shape), dtype= torch.float32, device= self.device)  # shape (batch_size, action_dim)
            acts = logits + noise
            # acts need to clamp in [-max_action, max_action] ((-1, 1) here)
            acts = torch.clamp(acts, -1, 1)  # preserve gradient
            # we use tanh to provide smoother gradient
            # acts = torch.tanh(acts)
        else: acts = logits

        # 若是要從機率分布中抽樣，就把該機率分布存入 dist
        dist = None

        # 回傳一個新的 Batch，因為這個 Batch 沒有要放到 ReplayBuffer 所以裡面的 key 跟 value 都可以隨便設隨便放
        return Batch(logits= logits, act= acts, state= state, dist= dist)


    '''Assume time t
    * Actor loss (不管 n_step 都是做一步更新一次) : 
        Actor_loss = -E[ min{Q^k(s(t), a(t))} ], k= 1, 2 

    * Critic loss (受 n_step 影響，每 n_step 更新一次) : 
        Critic^k_loss = E[ MSE( Q^k(s(t), a(t)), ( r(t) + \\gamma*r(t+1) + \\gamma^2*r(t+2) + ... + \\gamma^(n-1)*r(t+n-1) + \\gamma^(n)*min{ Q^k_target(s(t+n), a_target(t+n) } ) ) ) ], k= 1, 2
    '''

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入當前 Batch 的資料，算出各資料的 Critic^k_loss 中最後的 min{ Q_target^k(s(t+n), a_target(t+n) }，即 TD Target 的 Q 部分
    # 不用考慮 n_step，這邊只是在設定如何透過 target_actor 得出 min{Q_target}
    # 這邊傳入的 batch = buffer[indices]，batch.obs_next 已經是 s(t+n) (process_fn 所呼叫的 compute_nstep_return 會算前面的)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def td_target_q(
        self,
        buffer : ReplayBuffer, 
        indices : np.ndarray  # shape (batch_size)
    ):
        batch = buffer[indices]  # 取出當前 batch 的資料，type = Batch()，此時 batch.obs_next 即是 s(t+n)
        # 算出 a_target(t+n), shape (batch_size, action_dim)
        act_target = self.forward(batch= batch, state= 'obs_next', model= 'target_actor').act
        # 為了傳入 targetQ，把 obs_next 取出來。shape (batch_size, state_dim)
        s_t_n = to_torch(batch.obs_next, device= self.device, dtype= torch.float32)
        return self.target_critic.q_min(state= s_t_n, action= act_target)  # shape (batch_size)
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 Batch，算出其中各資料的 TD Target。ex: r(t) + \gamma*r(t+1) + \gamma^2*r(t+2) + ... + \gamma^(n-1)*r(t+n-1) + \gamma^(n)*min{ Q^k(s(t+n), a_target(t+n) }
    # 最後將算出的各資料的 TD_target 放入傳入的那個 Batch.returns。回傳傳入的那個 Batch
    # 這個 batch 是一個新的 Batch，不會再回到 ReplayBuffer。所以 return 不用是 np.array 形式，他是 torch.tensor，且是有梯度的。
    # 此函式所回傳的 batch 是複製原本的 batch 的所有資料後再加一個欄位 (returns)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    def process_fn(
        self, 
        batch : Batch,  # batch data
        buffer : ReplayBuffer,  # replay buffer, used to calculate the n-step return (rewards are stored in order in the replay buffer)
        indices : np.ndarray  # 傳入的 batch 於 replay buffer 中的 indices。shape (batch_size)
    ):
        return self.compute_nstep_return(
            batch= batch,  # 傳入的一個 batch 的資料
            buffer= buffer,  # replay buffer (用在算 n_step reward & self.TD_Target_Q)
            indices= indices,  # 傳入的 batch 在 replaybuffer 中的 indices (用在算 n_step reward & self.TD_Target_Q)
            target_q_fn= self.td_target_q,
            gamma= self.gamma,
            n_step= self.n_steps,
            rew_norm= self.reward_normalization
        )
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 輸入一個 batch 的資料 (已經有做過 process_fn 的 Batch，裡面已經有 Batch.returns) 來更新一次 Twin Q-Network (不包括 Target Twin Q-network)
    # 最後會回傳一個 batch 中各資料得出的兩個 Critic loss 的總和 (torch.tensor with shape (1))
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''recall
    * Critic loss (受 n_step 影響，每 n_step 更新一次) : 
        Critic^k_loss = E[ MSE( Q^k(s(t), a(t)), (TD_target: r(t) + \\gamma*r(t+1) + \\gamma^2*r(t+2) + ... + \\gamma^(n-1)*r(t+n-1) + \\gamma^(n)*min{ Q^k_target(s(t+n), a_target(t+n) } ) ) ) ], k= 1, 2
        **兩個 Critic 會有相同的 td_target_q (which's stored in batch.return and processed in def process_fn)**
    '''
    '''Critic 會在本函式中更新'''
    def update_critic(
        self, 
        batch : Batch
    ):
        state = to_torch(batch.obs, dtype= torch.float32, device= self.device)
        action = to_torch(batch.act, dtype= torch.float32, device= self.device)
        current_q1, current_q2 = self.critic(state= state, action= action)
        td_target = batch.returns.detach()  # avoid transporting the gradient to the target networks
        critic_loss = nn.functional.mse_loss(input= current_q1, target= td_target) + nn.functional.mse_loss(input= current_q2, target= td_target)
        # update DoubleCritic Network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 batch 的資料，並用此 batch 去算這個 batch 得出的 Actor_loss 並回傳。torch.tensor with shape(1)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#  
    '''
    * Actor loss (不管 n_step 都是做一步更新一次) : 
        Actor_loss = -E[ min{Q^k(s(t), a(t))} ], k= 1, 2 
    '''
    '''Actor 不會在本函式中更新，會在後面的 def learn 中才更新'''
    def update_policy(
        self,
        batch : Batch,
        update : bool = False  # 是否在此函式中進行 actor 的參數更新
    ):
        state = to_torch(batch.obs, dtype= torch.float32, device= self.device)
        action = self.forward(batch= batch, state= 'obs', model= 'actor').act
        action = to_torch(action, dtype= torch.float32, device= self.device)
        # mean() 只接受 torch.float32，而這邊 q_min 是從神經網路出來，自然是 torch.float32
        actor_loss = -self.critic.q_min(state= state, action= action).mean()  
        
        if update:
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        return actor_loss

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # soft update target actor and target Twin Q network
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def update_target_networks(self):
        self.soft_update(tgt= self.target_actor, src= self.actor, tau= self.tau)
        self.soft_update(tgt= self.target_critic, src= self.critic, tau= self.tau)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 batch 的資料並更新所有網路的更新，最後回傳該 batch 的 Actor loss & Critic loss (in dict)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    def learn(
        self,
        batch : Batch
    ):
        # update DoubleCritic through batch
        critic_loss = self.update_critic(batch= batch)
        # update Actor through batch
        actor_loss = self.update_policy(batch= batch, update= False)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update all target networks
        self.update_target_networks()

        # return actor loss & critic loss via dict
        return {
            'critic_loss' : critic_loss.item(), 
            'actor_loss' : actor_loss.item()
        }
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 ReplayBuffer，從該 buffer 中抽取一個 batch 的資料來更新一遍所有的 networks
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    '''essential in BasePolicy'''
    def update(
        self, 
        sample_size: int,  # batch_size
        buffer: ReplayBuffer
    ):
        # sample a batch of data from buffer
        batch, indices = buffer.sample(batch_size= sample_size)
        # compute TD_Target of each data in the batch
        batch = self.process_fn(batch= batch, buffer= buffer, indices= indices)
        # update 
        result = self.learn(batch= batch)
        # once update the network update the lr
        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
        # return actor loss & critic loss via dict
        return result