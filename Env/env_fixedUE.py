'''
0. 本論文的時間定義：
        frame = learning window = 1s = 2000 subframe (每一個 frame 做一次上層 by GAN-DDQN)
        time_subframe = 0.5ms (每一個 time_subframe 做一次下層 by RR) (我自己把 time_subframe 叫 timeslot)
1. For downlink simulations in one-single base station environment.
2. This simulation is based on 4G LTE & 3GPP TS 36.814 Standard.
3. unit in this environment: 
    tx, rx power : dBW
    gain, loss   : dB (dB : relative unit, 10 x log_10 (A / B))
    in SNR       : W (dBW must turn into W ---> value_in_W = 10** (value_in_dBW / 10))
'''

import numpy as np
import time

class cellularEnv(object):
    # __init__() 括號中的值都有 assign，這些 assign 的值是預設值，若在實例化的時候有傳入值進來會以實例化傳的值為主
    def __init__(self,
                 
        # Base Station Position & Area
        BS_pos = np.array([0,0]),
        BS_radius = 40,
        
        # BS transmit power # tx = transmit
        # unit is dBW (16dbW = 46dBm) # 為常見的 LTE 傳輸功率
        BS_tx_power = 16, 

        # 最大 UE 個數
        UE_max_no = 100, 
        # 每個 UE 最多可以一次處理五個 packets
        Queue_max = 5,

        # 噪聲功率譜密度 (unit = dbw/Hz) 用於算 SNR
        # noise Power Spectral Density
        # 每 Hz 有多少功率的噪聲
        noise_PSD = -204, # 約-174 dbm/Hz

        # 使用的通道模型為 3GPP TR 36.814 (常用於 LTE 模擬)
        # 包含 Path Loss 和 Shadow Fading
        chan_mod = '36814',
        
        # 用在說明使用的頻段的中心頻率，即使用哪一段的 spectrum
        carrier_freq = 2 * 10 ** 9, # 2 GHz
        # 總共可用頻寬為 10MHz
        band_whole = 10 * 10 ** 6, # unit = Hz
        # 由上可以合理推出本模擬使用的頻段是 [2GHz - 5MHz, 2GHz + 5MHz]

        # LTE 的一個 subframe = 1ms，一個 slot = 0.5ms。(1 subframe = 2 slots)
        # 這邊是指一個 timeslot = 0.5ms，只是他這邊命名沒在管 LTE
        # unit : s
        time_subframe = 0.5 * 10 ** (-3), # by LTE, 0.5 ms

        # 用 RR 來做網路切片到使用者之間的分配
        schedu_method = 'round_robin',
        
        # 網路切片種類 (VoLTE、eMBB、URLLC)
        ser_cat = ['volte','embb_general','urllc'],
        # 隨機分配 UE 的服務需求 (VoLTE : eMBB : URLLC = 6 : 6 : 1)
        ser_prob = np.array([6,6,1], dtype=np.float32),

        # MIMO 天線數
        dl_mimo = 32,
        # UE 端 (接收端) 的增益，跟距離無關，是硬體的能力 (結構)。可以把接收到的訊號的功率放大 20dB
        # rx = receive
        rx_gain = 20,  # dB

        # RL 的學習視窗，可以想成是一個 episode。在這個 episode 中會更新模型數次
        # 可以得出一次訓練會要 60000 * 0.5ms = 30s
        # 在論文實驗設定中將 learning windows 設為 2000，由此，每 2000 * 0.5ms = 1s 會更新一次
        learning_windows = 60000,
    ):

        self.BS_tx_power = BS_tx_power
        self.BS_radius = BS_radius
        
        self.carrier_freq = carrier_freq
        self.band_whole = band_whole
        self.noise_PSD = noise_PSD
        self.dl_mimo = dl_mimo
        self.UE_rx_gain = rx_gain

        self.chan_mod = chan_mod
        
        self.time_subframe = round(time_subframe, 4)
        # 為模擬環境中的全域時間指標 (unit : s)
        self.sys_clock = 0

        self.schedu_method = schedu_method 

        self.UE_max_no = UE_max_no
        # 存要傳給各 UE 的封包，有五格，每格都存要傳送的封包大小 (unit = bits) ex : UE_buffer[0].shape = (100)，UE_buffer[0][1] 是存要傳給 UE1 的封包大小
        # 創建一個 shape = (Queue_max (5), UE_max_no (100)) 的零矩陣
        # 有 5 個 np.ndarray，每個 array 中有 100 格，所以某一個 array 代表隊應所有 UE 的 Queue 的某一格
        self.UE_buffer = np.zeros([Queue_max, UE_max_no])  
        # 備份 UE_buffer，存 buffer 中每個封包的原始大小，用在 store_reward() 那邊，算花超過一個 timeslot 傳送的封包的傳輸速率 (原始封包大小 / 總 latency = 該封包的資料傳輸速率)
        self.UE_buffer_backup = np.zeros([Queue_max, UE_max_no])
        # 存所有 UE 的各封包的 Latency 情況 (unit = sec.)
        self.UE_latency = np.zeros([Queue_max, UE_max_no])
        # 存所有 UE 下一個封包可以產生的時間，當值為 0 的時候才可以產生新的封包
        self.UE_readtime = np.zeros(UE_max_no)
        # 存每個 UE 在當前 timeslot 分到多少頻寬 (units : RB)
        self.UE_band = np.zeros(UE_max_no)
        # 隨機生成所有 UE 的位置
        # [-self.BS_radius, self.BS_radius] 中均勻隨機產生出一個 shape 為 [self.UE_max_no, 2] 的二維陣列
        UE_pos = np.random.uniform(-self.BS_radius, self.BS_radius, [self.UE_max_no, 2])
        # 計算每個 UE 到基地台的距離 sqrt( (x1 - x2)^2 + (y1 - y2)^2 )
        # shape = (UE_max_no, 1)，即每一個 UE 跟 BS 的距離
        dis = np.sqrt(np.sum((BS_pos - UE_pos) **2 , axis = 1)) / 1000 # unit changes to km，為了要算 path loss
        
        # 根據 3GPP TR 36.814 的 path loss model (dis.unit = km)
        # 回傳一個 shape 是 (UE_max_no, 1) 的 np.array，每一格都代表該 UE 的 path loss
        self.path_loss = 145.4 + 37.5 * np.log10(dis).reshape(-1, 1)

        # RL 學習視窗，以 s 為單位，這邊可以推得就是 60000 * 0.0005 = 30s。代表一每 30s 更新一次參數
        # 論文實驗環境中是設為 2000，代表每 2000 * 0.5ms = 1s 會更新一次參數
        self.learning_windows = round(learning_windows * self.time_subframe, 4)

        self.ser_cat = ser_cat
        if len(self.ser_cat) > 1:
            self.band_ser_cat = np.zeros(len(ser_cat))  # 創建存各種網路切片每個 learning window 分到的頻寬資源的 np.ndarray
            if len(ser_prob) == len(self.ser_cat):  # 這邊就是把 [6 : 6 : 1] 正規化成機率
                self.ser_prob = ser_prob / np.sum(ser_prob)  # normalization
            else:
                self.ser_prob = np.ones(len(ser_cat)) / len(ser_cat)
        else:  # 只有一種網路切片，全部 UE 都用他，他也擁有全部的資源
            self.ser_prob = np.array([1])
            self.band_ser_cat = self.band_whole
        
        # 依照前面的機率分布產生各 UE 要使用的網路切片種類，shape : (UE_max_no)
        self.UE_cat = np.random.choice(self.ser_cat, self.UE_max_no, p=self.ser_prob)

        # tx_pkt_no 為每個類型網路切片成功傳輸的封包數的計數器，唯一個長度為 3 的 np.ndarray
        self.tx_pkt_no = np.zeros(len(self.ser_cat))

    #=======================================================================================================================================#
    # 通道模型 (只考慮大尺度衰弱 (考慮 path loss 和 shadow fading)) : 會得出每一個 UE 的通道狀況 (chan_loss, shape = (UE_max_no, 1))。 unit : dB
    def channel_model(self): 
        if self.chan_mod == '36814':
            shadowing_var = 8  # rayleigh fading shadowing variance 8dB。代表會有正負 8dB 的功率波動。
            # path_loss.shape = (UE_max_no, 1)，為每一個 UE 會有的 path_loss
            # 後面的 random.normal(...) 會產生出一個 (UE_max_no) 的 np.ndarray，內容為各 UE 的 shadow fading 值
            # 最後 reshape() 會將 shape 從 (UE_max_no) 轉成 (UE_max_no, 1)
            self.chan_loss = self.path_loss + np.random.normal(0, shadowing_var, self.UE_max_no).reshape(-1,1)  

    #=======================================================================================================================================#
    # 排程模型 : 網路切片分 RB 給其 UE，下面所說的兩種分法是分配在均分給 Active Users 後剩餘的 RB
    # 兩種分法 : round_robin (針對每個 slice 分開做排程，有先把頻寬先分給各切片) & round_robin_nons (所有使用者混在一起做排程，啟發式，直接把所有頻寬資源切成 RB 後直接分給全部的使用者)
    def scheduling(self):
        # initializing，UE_band : 各 UE 分到的 RB 個數 (shape : (UE_max_no))
        self.UE_band = np.zeros(self.UE_max_no) 
        
        # 第一種 : round_robin 
        if self.schedu_method == 'round_robin':
            ser_cat = len(self.ser_cat)
            band_ser_cat = self.band_ser_cat
            
            # 每個 Learning Window 的第一個 timeslot (在這邊就是一個 time_subframe) 就重置一次 ser_schedu_ind[0], [1], [2] = 0
            # *10000 是避免有小數
            if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):
                self.ser_schedu_ind =  [0] * ser_cat  # ser_schedu_ind (shape: (3)) 是紀錄每個 slice 下一個要分配的 UE 是哪個
            
            # 每次排一種網路切片
            for i in range(ser_cat): 
                UE_index = np.where((self.UE_buffer[0,:]!=0) & (self.UE_cat == self.ser_cat[i]))[0]  # 把 active 的 UE 編號取出放入 UE_index
                UE_Active_No = len(UE_index)  # UE_Active_No 為 active 的 UE 個數
                
                if UE_Active_No != 0:
                    RB_No = band_ser_cat[i] // (180 * 10**3)  # 把可用頻寬轉為 RB 個數 (1RB = 180KHz，LTE 定義) (// : 整數除法)
                    RB_round = RB_No // UE_Active_No  # 均分 RB 個數給各 Active Users
                    self.UE_band[UE_index] += 180 * 10**3 * RB_round  # 每個 UE 分得到的頻寬 (從 RB 轉回頻寬) 存入 UE_band
                    
                    # 除不盡的 RB 個數用 RR 的方式分給 Active Users
                    RB_rem_no = int(RB_No - RB_round * UE_Active_No)  # RB_rem_no : 除不盡剩餘的 RB 個數 
                    left_no = np.where(UE_index > self.ser_schedu_ind[i])[0].size  # 算剩下多少人 (從 ser_schedu_ind[i] 開始往後算到底)
                    if left_no >= RB_rem_no:  # 剩下的 RB 比剩下的人少 (不會需要從頭輪)，分給剩下的人一人一塊 RB。
                        # np.greater_equal() : 選出 index 大於等於 ser_schedu_ind[i] 的
                        # np.less() : 選出 index 小於 (ser_schedu_ind[i] + 剩餘 RB 個數)
                        # np.logical_and : 前面兩個條件的 and，可以得到哪些人會分的到 RB
                        # ex: UE_index = [1, 2, 3, 4, 5, 6, 7, 8], ser_schedu_ind[i] = 2, RB_rem_no = 3, 那 3, 4, 5 號 UE 都能分到一塊 RB
                        UE_act_index = UE_index[np.where(np.logical_and(np.greater_equal(UE_index, self.ser_schedu_ind[i]), np.less(UE_index, RB_rem_no + self.ser_schedu_ind[i])))]
                        if UE_act_index.size != 0:
                            self.UE_band[UE_act_index] += 180 * 10**3  # 選出來的 UE 都各分一塊 RB # np.ndarray 才可以這樣在 index 中放另一個 np.ndarray，list 不行
                            self.ser_schedu_ind[i] = UE_act_index[-1] + 1 # 更新 ser_schedu_ind[i] 
                    else:  # 剩下的 RB 比剩餘的人多 (會需要從頭輪)
                        UE_act_index_par1 = UE_index[np.where(UE_index > self.ser_schedu_ind[i])]
                        UE_act_index_par2 = UE_index[0 : RB_rem_no-left_no]
                        self.UE_band[np.hstack((UE_act_index_par1,UE_act_index_par2))] += 180 * 10**3
                        self.ser_schedu_ind[i] = UE_act_index_par2[-1] + 1

        # 第二種 : round_robin_nons，啟發式，直接把全部頻寬切成 RB 之後用 RR 依序分給所有的 Active UE
        elif self.schedu_method == 'round_robin_nons':
            band_whole = self.band_whole
            if self.sys_clock == self.time_subframe:  # 整個模擬剛開始，初始化 ser_schedu_ind[0], [1], [2] = 0
                self.ser_schedu_ind =  0
                
            UE_index = np.where((self.UE_buffer[0, :] != 0))[0]  # UE_index : Active Users' index
            UE_Active_No = len(UE_index)  # UE_Active_No : No. of Active Users
            if UE_Active_No != 0:
                RB_No = band_whole // (180 * 10**3)  # Divide Bandwidth into RBs
                RB_round = RB_No // UE_Active_No  # Evenly distribute the RBs among all active users
                
                self.UE_band[UE_index] += 180 * 10**3 * RB_round  # 紀錄每個 UE 分到的頻寬資源
                
                # 除不盡，剩餘的 RBs。這邊同前面的 RR
                RB_rem_no = RB_No % UE_Active_No  
                left_no = np.where(UE_index > self.ser_schedu_ind)[0].size
                if left_no >= RB_rem_no:  # 剩餘的人夠分完剩餘的 RB，不用回頭
                    UE_act_index = UE_index[np.where(np.logical_and(np.greater_equal(UE_index,self.ser_schedu_ind),np.less(UE_index, RB_rem_no + self.ser_schedu_ind)))]
                    if UE_act_index.size != 0:
                        self.UE_band[UE_act_index] += 180 * 10**3
                        self.ser_schedu_ind = UE_act_index[-1] + 1 
                else:  # 剩餘的人不足以分完剩餘的 RB，要回頭
                    UE_act_index_par1 = UE_index[np.where(UE_index > self.ser_schedu_ind)]
                    UE_act_index_par2 = UE_index[0 : RB_rem_no - left_no]
                    self.UE_band[np.hstack((UE_act_index_par1, UE_act_index_par2))] += 180 * 10**3
                    self.ser_schedu_ind = UE_act_index_par2[-1] + 1
            
            # 每個 Learning Window 都重置一次各網路切片所分到的頻寬資源
            if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):
                self.band_ser_cat = np.zeros(len(self.ser_cat))
            # 計算這整個 Learning Window，各網路切片平均一個 timeslot 分到多少的頻寬資源
            for i in range(len(self.ser_cat)):
                if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):  # 本 Learning Window 第一個 timeslot
                    self.band_ser_cat[i] = np.sum(self.UE_band[self.UE_cat == self.ser_cat[i]])
                else:  # 非第一個 timeslot
                    self.band_ser_cat[i] += np.sum(self.UE_band[self.UE_cat == self.ser_cat[i]])
                    if (self.sys_clock * 10000) % (self.learning_windows * 10000) == 0:  # 本 Learning Window 最後一個 timeslot
                        lw = (self.learning_windows * 10000) / (self.time_subframe * 10000)  # (Learning Window 有幾個 timeslot)
                        self.band_ser_cat[i] = self.band_ser_cat[i] / lw  # 算出每個網路切片平均一個 timeslot 分到的頻寬資源量。

    #=======================================================================================================================================#
    # 根據 Shannon 計算每個 UE 能達到的資料傳輸速率
    # 透過算出的資料傳輸速率進行封包傳輸，並算出當前 timeslot 的 reward
    # 結算當前 timeslot 結束後的 buffer 情況，傳完的封包會將其對應的 latency 刪除，還沒傳完的會續留
    def provisioning(self):
        UE_index = np.where(self.UE_band != 0)  # 找出有被分配到頻寬的 UE
        self.channel_model()  # 算出所有 UE 的通道狀況 (考慮大尺度衰弱後的通道，unit = dB) -> chan_loss (shape : (UE_max_no, 1))
        rx_power = 10 ** ((self.BS_tx_power - self.chan_loss + self.UE_rx_gain) / 10)  # 接收端收到的訊號的功率強度 (unit : W)，shape : (UE_max_no, 1)
        rx_power = rx_power.reshape(1, -1)[0]  # 把 shape 從 (UE_max_no, 1) 轉為 (1, UE_max_no) 再由 [0] 取出

        # 算出各 UE 的資料傳輸速率 (unit : bits/s) by Shannon，shape : (UE_max_no, 1)
        rate = np.zeros(self.UE_max_no)
        # rate (bits / s) = bandwidth (unit : 次數/s) * log (1 + SNR) (unit : bits / 次數)
        # 最後乘上天線數量是一種理想情況的近似方法，為簡化建模。
        # 下述 Shannon 單位轉換見 Hackmd
        rate[UE_index] = self.UE_band[UE_index] * np.log10(1 + rx_power[UE_index] / ( 10 **(self.noise_PSD / 10) * self.UE_band[UE_index] )) * self.dl_mimo
        
        # 找出有哪些 UE 是封包的傳送目的地
        # ex : 
        # 有三個 User，每個 User 有一個空間大小為 2 的 Queue
        # Buffer = np.array([[2, 3, 4], [1, 6, 7]])  # shape = (2, 3)
        # k = np.sum(Buffer, axis = 0)  # k = array([3, 9, 11]) # 要傳送 3 bits 到 UE1
        # l = np.where(k != 0)  # output : array([0, 1, 2])
        # 代表 3 個 UE 對應的 Queue 都有東西

        # UE_buffer.shape = (Queue_max (5), UE_max_no (100))
        buffer = np.sum(self.UE_buffer, axis = 0)  # 這個 timeslot 總共要傳送的 bits 總量
        # UE_index_b (shape : (n, n), n = buffer != 0 的 UE 個數，第一個 n 為該 UE 所在 UE_buffer 中的列數，即要傳封包的目標 UE 的 index) : 
        UE_index_b = np.where(buffer != 0)  

        # 更新各 UE 中各封包的延遲，一次考慮一個 UE
        # 就算是第一個 timeslot 就傳完還是會有一個 timeslot 的 latency
        # UE_latency[:, ue_id] : ue_id 的 UE 對應的 Queue 的那五格中的封包的 Latency
        # UE_buffer[:, ue_id] : ue_id 的 UE 對應的 Queue 的那五格中的封包大小
        for ue_id in UE_index_b[0]:
            self.UE_latency[:, ue_id] = latencyUpdate(self.UE_latency[:, ue_id], self.UE_buffer[:, ue_id], self.time_subframe)
        
        # 更新各 UE 中的 buffer，一次考慮一個 UE
        for ue_id in UE_index[0]: 
            self.UE_buffer[:, ue_id] = bufferUpdate(self.UE_buffer[:, ue_id], rate[ue_id], self.time_subframe)  
        
        # 算出當前 reward 並與前面的 (同 Learning Window) 的累加
        self.store_reward(rate)
        
        # 在一個 timeslot 結束後，更新一下 buffer 中的封包狀況以及他們的 latency。
        # 封包大小 = 0，latency != 0 -> latency 清 0
        self.bufferClear()
        
    #=======================================================================================================================================#
    # 每個 UE 依照其隸屬的網路切片種類，以間隔時間產生不同大小的封包並放進其對應的 Queue 中
    # 若 UE 對應的 Queue 是滿的就不產生，空的的話就產生一個
    def activity(self): #https://www.ngmn.org/fileadmin/user_upload/NGMN_Radio_Access_Performance_Evaluation_Methodology.pdf
        # VoLTE uses the VoIP model
        # embb_general uses the video streaming model
        # urllc uses the FTP2 model

        # 設定使用者的 readtime，即隔多久會再產生一筆資料進入 buffer 的時間間隔，各 UE 的 readtime 會存入 self.UE_readtime
        if self.sys_clock == 0:  # 模擬剛開始，初始化
            for ser_name in self.ser_cat:  # 一次考慮一種網路切片
                ue_index = np.where(self.UE_cat == ser_name)  # ue_index : 隸屬於該網路切片的使用者的 index
                ue_index_Size = ue_index[0].size
                
                if ser_name == 'volte':
                    # the silence lasts 160 ms in maximum
                    # readtime 用 (0, 160ms) 隨機分布，產生 size 為 (1, ue_index_Size) 的 np.ndarray
                    self.UE_readtime[ue_index] = np.random.uniform(0, 160 * 10 ** (-3), [1, ue_index_Size]) 

                elif ser_name == 'embb_general':
                    tmp_readtime = np.random.pareto(1.2, [1, ue_index_Size]) * 6 * 10 ** -3
                    tmp_readtime[tmp_readtime > 12.5 * 10 ** -3] = 12.5 * 10 ** -3  # readtime 上限為 12.5 ms，超過者設為 12.5 ms
                    self.UE_readtime[ue_index]  = tmp_readtime

                elif ser_name == 'urllc':
                    # read time is determines much smaller; the spec shows the average time is 180s, but here it is defined as 180 ms
                    # 模擬突發封包
                    self.UE_readtime[ue_index]  = np.random.exponential(180* 10 ** -3, [1, ue_index_Size]) 

        # 針對每個 UE 看是否要產生新的封包，封包大小的單位為 bits
        for ue_id in range(self.UE_max_no):  # 每次考慮一個 UE

            if self.UE_readtime[ue_id] <= 0: # 到了 readtime，該產生封包
                # buffer 如果是空的就產生封包，如果不是空的就不產生
                if self.UE_buffer[:, ue_id].size - np.count_nonzero(self.UE_buffer[:, ue_id]) != 0: # The buffer is not full (5 - 目前 UE 對應的 Queue 中的封包個數)
                    buf_ind = np.where(self.UE_buffer[:, ue_id] == 0)[0][0]  # buffer index 從 0 開始找到第一個為空的地方，將封包產生在這
                    
                    # 依照 UE 隸屬的網路切片規則來產生封包
                    if self.UE_cat[ue_id] == 'volte':
                        self.UE_buffer[buf_ind, ue_id] = 40 * 8  # 產生一個大小為 40Byte 的封包放到剛剛的找到的空位
                        self.UE_readtime[ue_id] = np.random.uniform(0,160 * 10 ** (-3), 1).squeeze()  # 每次產生完封包之後都會再隨機產生 readtime

                    elif self.UE_cat[ue_id] == 'embb_general':
                        # squeeze() is used to unpack [no] to no
                        tmp_buffer_size = np.random.pareto(1.2, 1).squeeze() * 800  # 產生 800 bits 為 base 的 Pareto 分布的封包大小
                        if tmp_buffer_size > 2000:  # 封包大小至多 2000 bits，超過就改成 2000 bits
                            tmp_buffer_size = 2000
                        # tmp_buffer_size = np.random.choice([1*8*10**6, 2*8*10**6, 3*8*10**6, 4*8*10**6, 5*8*10**6])
                        self.UE_buffer[buf_ind, ue_id] = tmp_buffer_size
                        # 再產生一次 readtime
                        self.UE_readtime[ue_id] = np.random.pareto(1.2, 1).squeeze() * 6 * 10 ** -3  
                        if self.UE_readtime[ue_id] > 12.5 * 10 ** -3:
                            self.UE_readtime[ue_id] = 12.5 * 10 ** -3 

                    elif self.UE_cat[ue_id] == 'urllc':
                        #tmp_buffer_size = np.random.lognormal(14.45,0.35,[1,1])
                        # if tmp_buffer_size > 5 * 10 **6:
                        #      tmp_buffer_size > 5 * 10 **6
                        # 小封包
                        tmp_buffer_size = np.random.choice([6.4*8*10**3, 12.8*8*10**3, 19.2*8*10**3, 25.6*8*10**3, 32*8*10**3])  # {6.4, 12.8, 19.2, 25.6, 32} KB 
                        # 大封包
                        # tmp_buffer_size = np.random.choice([0.3*8*10**6, 0.4*8*10**6, 0.5*8*10**6, 0.6*8*10**6, 0.7*8*10**6])  # buffer_size 介於 0.3 ~ 0.7M bits
                        self.UE_buffer[buf_ind,ue_id] = tmp_buffer_size
                        # 再產生一個 readtime
                        self.UE_readtime[ue_id]  = np.random.exponential(180* 10 ** -3, 1).squeeze()

                    self.tx_pkt_no[self.ser_cat.index(self.UE_cat[ue_id])] += 1  # 將記錄 learning window 中總封包總數的計數器 (tx_pkt_no) + 1
                    self.UE_buffer_backup[buf_ind, ue_id] = self.UE_buffer[buf_ind, ue_id]  # 產生完新封包後馬上備份 UE_buffer 到 UE_buffer_backup
                    
            else:  # 還沒到 readtime，扣掉 timeslot
                self.UE_readtime[ue_id] -= self.time_subframe
        
        # 每個 timeslot 後更新一次系統時間
        self.sys_clock += self.time_subframe
        self.sys_clock = round(self.sys_clock,4)

    #=======================================================================================================================================#   
    # 取得環境的狀態，即個網路切片分別要傳的封包個數 (d0, d1, d2)
    def get_state(self):
        #state = np.zeros(len(self.ser_cat))
        #for ser_name in self.ser_cat:
        #    ue_index = np.where(self.UE_cat == ser_name)
        #    state[self.ser_cat.index(ser_name)] = np.where(self.UE_buffer[0,ue_index[0]] != 0)[0].size
        state = self.tx_pkt_no
        return state
    
    #=======================================================================================================================================#
    # 一個 Learning Window 中每一個 timeslot 都會執行
    # 計算當前 timeslot 的系統總 SE & EE & SSR，並與之前的 (同一個 Learning Window) 累加
    # 到當前 Learning Window 的最後一個 timeslot 時會有整個 Learning Window 的總 SE & EE & SSR
    def store_reward(self, rate):
        
        # 計算各網路切片的 SE & EE & 總系統的 SE & EE
        se = np.zeros(len(self.ser_cat))  # 存各種網路切片的 Spectrum Efficiency (unit : bits/s /Hz) (每 Hz 貢獻多少資料傳輸速率)
        ee = np.zeros(len(self.ser_cat))  # 存各種網路切片的 Energy Efficiency (unit : /W) (每 W 貢獻多少光譜效率)
        sys_rate_frame = 0  # 計算這個 timeslot 所有 UE 的資料傳輸速率的總和 (系統總資料傳輸速率)
        for ser_name in self.ser_cat:  # 一次考慮一種網路切片
            ser_index = self.ser_cat.index(ser_name)  # 該網路切片的 index
            ue_index_ = np.where(self.UE_cat == ser_name)  # 隸屬於該網路切片服務的 UE 的 index 集合
            allo_band = np.sum(self.UE_band[ue_index_])  # 該網路切片所分到的頻寬資源總和
            sum_rate = np.sum(rate[ue_index_])  # 隸屬於該網路切片服務的 UE 的資料傳輸速率總和
            if allo_band != 0:
                sys_rate_frame += sum_rate  # 計算系統總資料傳輸速率
                se[ser_index] = sum_rate / allo_band  # 計算該網路切片的 SE
                ee[ser_index] = se[ser_index] / 10**(self.BS_tx_power / 10)  # 計算該網路切片的 EE
        
        # 計算當前 timeslot 整個系統 (所有種類的網路切片總和) 的 SE & EE
        # 此外，累加之前的 (同一個 Learning window 的)
        self.sys_se_per_frame += sys_rate_frame / self.band_whole

        # 原本應該是要處理計算 latency 時的延遲
        handling_latency = 2 * 10 ** (-3)
        handling_latency = 0

        # 每個 UE 成功傳送封包的次數 (成功的標準取決於該 UE 使用的網路切片)
        for ue_id in range(self.UE_max_no):  # 一次考慮一個 UE
            for i in range(self.UE_latency[:, ue_id].size):  # 考慮對應到 ue_id 的 UE 的 Queue 的所有封包 (i 是封包的 index)
                if (self.UE_buffer[i, ue_id] == 0) & (self.UE_latency[i, ue_id] != 0):  # 考慮已經傳完的封包的 latency
                    
                    # 屬於 VoLTE 網路切片，SLA : rate >= 51kbps、latency < 10ms
                    if self.UE_cat[ue_id] == 'volte': 
                        cat_index = self.ser_cat.index('volte')  # 該網路切片的 index 
                        if (self.UE_latency[i, ue_id] == self.time_subframe):  # 封包只用一個 timeslot 就傳完
                            if (rate[ue_id] >= 51 * 10 ** 3) & (self.UE_latency[i, ue_id] < 10 * 10 **(-3) - handling_latency): 
                                self.succ_tx_pkt_no[cat_index] += 1  # succ_tx_pkt_n[i] : 網路切片 i 成功傳出的封包總數
                        else:  # 封包用不只一個 timeslot 才傳完
                            # buffer_backup 為一開始 UE_buffer.copy，記錄著封包的初始大小
                            # 由封包原始大小 / latency (即傳輸的時間) 得到該封包的資料傳輸速率
                            if (self.UE_buffer_backup[i,ue_id] / self.UE_latency[i, ue_id] >= 51 * 10 ** 3) & (self.UE_latency[i, ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                    
                    # 屬於 eMBB_general 網路切片，SLA : rate >= 100Mbps、latency < 10ms
                    elif self.UE_cat[ue_id] == 'embb_general':
                        cat_index = self.ser_cat.index('embb_general')    
                        if (self.UE_latency[i, ue_id] == self.time_subframe):  # 封包只用一個 timeslot 就傳完
                            if (rate[ue_id] >= 100 * 10 ** 6) & (self.UE_latency[i, ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                        else: # 封包用不只一個 timeslot 才傳完
                            # rate 的部分 : 該封包的大小
                            if (self.UE_buffer_backup[i, ue_id] / self.UE_latency[i, ue_id] >= 100 * 10 ** 6) & (self.UE_latency[i, ue_id] < 10 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                    
                    # 屬於 uRLLC 網路切片，SLA : rate >= 10Mbps、latency < 1ms
                    elif self.UE_cat[ue_id] == 'urllc': 
                        cat_index = self.ser_cat.index('urllc')   
                        if (self.UE_latency[i, ue_id] == self.time_subframe):  # 封包只用一個 timeslot 就傳完
                            if (rate[ue_id] >= 10 * 10 ** 6) & (self.UE_latency[i, ue_id] < 1 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                        else:  # 封包用不只一個 timeslot 才傳完
                            if (self.UE_buffer_backup[i, ue_id] / self.UE_latency[i, ue_id] >= 10 * 10 ** 6) & (self.UE_latency[i, ue_id] < 1 * 10 **(-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1

    #=======================================================================================================================================#
    # 一個 Learning window 只會執行準備要更新模型時的那一次
    # 回傳整個 Learning Window 的 Reward
    def get_reward(self):

        # 1. 當前 Learning Window 中，平均一個 timeslot 的 SE
        # 整個 Learning Window 各 timeslots 的總 SE 總和 / 一個 Learning Windows 的 timeslots 總數
        se = self.sys_se_per_frame / (self.learning_windows / self.time_subframe)
        # ee_total = se_total/10**(self.BS_tx_power/10)   
        
        # 2. 整個 Learning Window 滿足 SLA 傳送成功的封包總數 / 整個 Learning Window 的封包總數
        qoe = self.succ_tx_pkt_no / self.tx_pkt_no 
        
        return qoe, se

    #=======================================================================================================================================#
    # 於每個 timeslot 結束後依照 UE_buffer 清理 UE_buffer_backup & UE_latency
    # 沒傳完的封包會繼續存在 buffer 中。這邊主要是要清掉已經傳完的封包的 latency & 其 backup
    def bufferClear(self):  # UE_latency.shape = (Queue_max, UE_max_no)
        latency = np.sum(self.UE_latency, axis = 0)  # latency.shape(UE_max_no) : 每個 UE 對應的 Queue 內的封包的 Latency 總和
        UE_index = np.where(latency != 0)  # 只要有任一封包的 Latency 不為 0 則該 UE 的 index 會被記錄於 UE_index
        bufSize = self.UE_latency[:, 0].size  # bufSize = Queue_max (5)
        for ue_id in UE_index[0]:  # 一次考慮一個 Latency 不為 0 的 UE
            
            # 備份，避免改到原始物件。
            buffer_ = self.UE_buffer[:, ue_id].copy()  # UE_buffer 的備份，即對應於 ue_id 的 UE 的 Queue (shape = Queue_max (5))
            buffer_bk = self.UE_buffer_backup[:, ue_id].copy()  # buffer_backup 的備份
            latency_ = self.UE_latency[:, ue_id].copy()  # UE_latency 的備份，即對應於 ue_id 的 UE 的 Queue 中封包的 Latency (shape = Queue_max (5))
            
            # 處理封包已經傳完，但 latency 還沒清掉的情況
            ind_1 = np.where(np.logical_and(buffer_ == 0, latency_ != 0))   # ind_1 : 封包已經為 0 但 latency != 0 的封包所在 index
            indSize_1 = ind_1[0].size
            if indSize_1 != 0:
                self.UE_latency[ind_1, ue_id] = np.zeros(indSize_1) # 把那些位置的 latency 清為 0
                self.UE_buffer_backup[ind_1, ue_id] = np.zeros(indSize_1)  # 那些位置對應的 backup 的位置清為 0

            # 封包還沒傳完，latency 也 != 0 的情況，東西留到下一個 timeslot
            # 把還沒傳完的封包的大小、latency、backup 全都往前移，讓空的在 Queue 的最後面
            ind = np.where(np.logical_and(buffer_ != 0, latency_ != 0))  # ind : 封包大小和 latency 皆不為 0 的封包所在 index
            ind = ind[0]  # "只要那 5 格中的其中幾格" 這個資訊就好
            indSize = ind.size
            if indSize != 0:
                # 全部清 0
                self.UE_buffer[:, ue_id] = np.zeros(bufSize)
                self.UE_latency[:, ue_id] = np.zeros(bufSize)
                self.UE_buffer_backup[:, ue_id] = np.zeros(bufSize)
                
                # 從 0 開始放東西，根據前面得知的有東西的 index (ind) 來把還沒傳完的封包往前移
                self.UE_buffer[:indSize, ue_id] = buffer_[ind]  # [:indSize] 就是從 0 ~ indSize
                self.UE_latency[:indSize, ue_id] = latency_[ind]
                self.UE_buffer_backup[:indSize, ue_id] = buffer_bk[ind]

    #=======================================================================================================================================#   
    # 在每個 Learning window 結束後重置計數器      
    def countReset(self):
        self.tx_pkt_no = np.zeros(len(self.ser_cat))
        '''for ser_name in self.ser_cat:
            ser_index = self.ser_cat.index(ser_name)
            ue_index_ = np.where(self.UE_cat == ser_name)
            self.tx_pkt_no[ser_index] = np.where(self.UE_buffer[:,ue_index_]!=0)[0].size'''
        self.succ_tx_pkt_no = np.zeros(len(self.ser_cat))
        self.sys_se_per_frame = np.zeros(1)  
        self.UE_buffer = np.zeros(self.UE_buffer.shape)
        self.UE_buffer_backup = np.zeros(self.UE_buffer.shape)
        self.UE_latency = np.zeros(self.UE_buffer.shape)
          
#=======================================================================================================================================#
# 模擬封包傳輸給 ue_id 的 UE 的過程 : 所有封包共用 rate，從 index0 的開始傳
# buffer -> UE_buffer[:, ue_id] : ue_id 的 UE 對應的 Queue 的那五格中的封包大小
# rate -> ue_id 的 UE 的資料傳輸速率
# def bufferUpdate(buffer, rate, time_subframe):  
#     bSize = buffer.size  # Queue 中的封包個數
#     for i in range(bSize):
#         if buffer[i] >= rate * time_subframe:
#             buffer[i] -= rate * time_subframe  # 剩餘 bits
#             rate = 0
#             break
#         else:  # ??
#             rate_ = buffer[i]
#             buffer[i] = 0
#             rate -= rate_
#     return buffer

# 我認為正確的版本
def bufferUpdate(buffer, rate, time_subframe):
    bSize = buffer.size
    remaining_transmit_bits = rate * time_subframe  # unit : bits
    for i in range(bSize):
        if buffer[i] >= remaining_transmit_bits:
            buffer[i] -= remaining_transmit_bits
            remaining_transmit_bits = 0
            break
        else:
            remaining_transmit_bits -= buffer[i]
            buffer[i] = 0
    return buffer
            
#=======================================================================================================================================#
# 更新對應於 ue_id 的 UE 的 Queue 中的五格封包的 Latency，即各封包的 latency 加上一個 timeslot (0.5ms)
# latency -> UE_latency[:, ue_id] : ue_id 的 UE 對應的 Queue 的那五格中的封包的 Latency
# buffer -> UE_buffer[:, ue_id] : ue_id 的 UE 對應的 Queue 的那五格中的封包大小
def latencyUpdate(latency, buffer, time_subframe):
    lSize = latency.size  # Queue 中的封包個數
    for i in range(lSize):
        if buffer[i] != 0:
            latency[i] += time_subframe
    return latency
