'''
* 除了 User Mobility 以外，不清楚的地方可見 GAN-DDQN/Code/cellular_env.py

* For downlink simulations in one-single base station environment

* unit in this environment: 
    tx, rx power : dBW
    gain, loss   : dB (dB : relative unit, 10 x log_10 (A / B))
    in SNR       : W (dBW must turn into W ---> value_in_W = 10** (value_in_dBW / 10))

* 本論文的時間定義：
    frame = learning window = 1s = 2000 subframe (每一個 frame 做一次上層 by GAN-DDQN)
    time_subframe = 0.5ms (每一個 time_subframe 做一次下層 by RR) (我自己把 time_subframe 叫 timeslot)

* 模擬環境中的時間單位 : s

* 模擬環境中的範圍 :
    * BS Coverage Area : 以 BS (0, 0) 為中心，半徑 R (40m) 的圓
    * 整個範圍 : 以 BS 為中心，半徑 3R 的圓
    * 可能一開始不在 Coverage Area 之內，但下一刻可能會進來，故有使用一個 UE_cell 來維護目前在 Coverage Area 內的 ue_id

'''

import numpy as np
import time

np.random.seed(1)

class EnvMove(object):
    def __init__(self,
                 BS_pos = np.array([0, 0]),
                 BS_radius = 40,  # Area Size = circle with radius 40
                 # BS_tx_power = 0, # unit is dBW
                 BS_tx_power = 16,  # unit is dBW, 46dBm
                 UE_max_no = 1000,    
                 Queue_max = 5,
                 noise_PSD = -204,  # -174 dbm/Hz
                 chan_mod = '36814',  # channel model : 3GPP TR 36.814 (including Path Loss & Shadow Fading)
                 carrier_freq = 2 * 10 ** 9,  # 2 GHz  # the center frequency of the utilized band
                 time_subframe = 0.5 * 10 ** (-3),  # by LTE, 0.5 ms # unit is s
                 ser_cat = ['volte', 'embb_general', 'urllc'],
                 band_whole = 10 * 10 ** 6,  # 10MHz
                 schedu_method = 'round_robin',
                 ser_prob = np.array([6, 6, 1], dtype = np.float32),  # the proportion of users utilizing each service cat.
                 dl_mimo = 32,  # MIMO 天線數
                 rx_gain = 20,  # dB
                 learning_windows = 60000,
    ):
        self.BS_pos = BS_pos
        self.BS_tx_power = BS_tx_power
        self.BS_radius = BS_radius
        self.band_whole = band_whole
        self.chan_mod = chan_mod
        self.carrier_freq = carrier_freq
        self.time_subframe = round(time_subframe, 4)
        self.noise_PSD = noise_PSD
        self.sys_clock = 0  # system clock (the only time line)
        self.schedu_method = schedu_method
        self.dl_mimo = dl_mimo
        self.UE_rx_gain = rx_gain
        self.UE_max_no = UE_max_no
        self.UE_buffer = np.zeros([Queue_max, UE_max_no])  # stores the packet size (unit : bit), format: [ [queue[0 ~ 4] for ue_id1], [queue[0 ~4] for ue_id2] ~ [queue[0 ~ 4] for ue_id UE_max_no] ]
        self.UE_buffer_backup = np.zeros([Queue_max, UE_max_no])  # data structure same as UE_buffer
        self.UE_latency = np.zeros([Queue_max, UE_max_no])  # stores the latency time of each packet, same format as self.UE_buffer
        self.UE_readtime = np.zeros(UE_max_no)  # time domain of two continuous packets
        self.UE_band = np.zeros(UE_max_no)  # bandwidth each UE get in one timeslot
        self.learning_windows = round(learning_windows * self.time_subframe, 4)
        self.ser_cat = ser_cat  # no. of service categories
        if len(self.ser_cat) > 1:  # more than 1 service category
            # self.band_ser_cat : bandwidth each service cat. get in one frame
            self.band_ser_cat = np.zeros(len(ser_cat))
            if len(ser_prob) == len(self.ser_cat):
                self.ser_prob = ser_prob / np.sum(ser_prob)
            else:
                self.ser_prob = np.ones(len(ser_cat)) / len(ser_cat)
        else:  # only 1 service category
            self.ser_prob = np.array([1])
            self.band_ser_cat = self.band_whole

        # Assign each user to a network slice type according to p (self.ser_prob)
        self.UE_cat = np.random.choice(self.ser_cat, self.UE_max_no, p = self.ser_prob)  # TBD

        # params of UE mobility model
        self.UE_pos = np.random.uniform(-3 * self.BS_radius, 3 * self.BS_radius, [self.UE_max_no, 2])  # pos = ([-3R ~ 3R], [-3R ~ 3R])
        # UE_cell : boolean array, indicating whether the UE is located in the BS coverage area
        # In every time unit we only consider the user which its cell = 1
        # if UE is moving into the BS coverage area, we can reconsider the resource allocation of this UE by maintaining this array
        # criterion : x^2 + y^2 <= radius^2 -> 1, otherwise 0
        self.UE_cell = np.zeros(self.UE_max_no)
        self.UE_cell[np.where(np.sum(self.UE_pos ** 2, axis = 1) <= self.BS_radius ** 2)] = 1
        # UE_speed depend on which service_cat they use # unit : m/s
        self.UE_speed = np.zeros(UE_max_no)
        self.UE_speed[np.where(self.UE_cat == 'volte')] = 1
        self.UE_speed[np.where(self.UE_cat == 'embb_general')] = 4
        self.UE_speed[np.where(self.UE_cat == 'urllc')] = 8
        # UE_direction obey the uniform probability in the range of [-180° ~ 180°] (basically most of the models use this)
        # why not [0° ~ 359°] ? + indicates turn left, - indicates turn right,  more convenient calculating the moving angle
        self.UE_direction = np.random.uniform(-180, 180, self.UE_max_no)
        
        self.tx_pkt_no = np.zeros(len(self.ser_cat))  # packets pending transmission in the current learning window
        self.tx_bit_no = np.zeros(len(self.ser_cat))  # bits pending transmision in the current learning window

    #=======================================================================================================================================#
    # Calculating the channel loss # unit : dB 
    # output_shape : (self.UE_max_no, 1)
    def channel_model(self):
        if self.chan_mod == '36814':
            shadowing_var = 8  # rayleigh fading shadowing variance 8dB
            # dis between the BS and UE
            # np.sqrt( (x^2 - 0 + y^2 - 0) )
            dis = np.sqrt(np.sum((self.BS_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            self.path_loss = 145.4 + 37.5 * np.log10(dis).reshape(-1, 1)  # by 3GPP TR 36.814
            self.chan_loss = self.path_loss + np.random.normal(0, shadowing_var, self.UE_max_no).reshape(-1, 1)

    #=======================================================================================================================================#
    # UE mobility model
    # self.UE_cell will update every windows
    # 1. maintain the user mobility, including the out of bounds problem
    # 2. calculate the ratio of users near the BS (distance <= R) to the total users in each network slice
    def user_move(self):
        l = self.UE_speed * self.learning_windows  # calculate the distance of each UE moves in this window (unit : m)
        # calculate & update the new position of every UEs
        delta_x = l * np.cos(self.UE_direction * np.pi / 180)
        delta_y = l * np.sin(self.UE_direction * np.pi / 180)
        self.UE_pos[:, 0] = self.UE_pos[:, 0] + delta_x
        self.UE_pos[:, 1] = self.UE_pos[:, 1] + delta_y
        
        # Address the out of bounds problem -> bounce back after hitting the boundries 
        # bounces back by the same amount it went out
        
        # Situation1 : Out of the left boundary (x < -3R)
        UE_index = np.where(self.UE_pos[:, 0] < -3 * self.BS_radius)  # find the ue_ids of UEs that there x are < -3R
        # set the bounce back x position
        # ex. UE_x = -3.2R, the reflect UE_x is -6R-(-3.2R) = -2.8R
        self.UE_pos[UE_index, 0] = -6 * self.BS_radius - self.UE_pos[UE_index, 0]  
        self.UE_direction[UE_index] = 180 - self.UE_direction[UE_index]  # set the direction (horizontal reflection)
        # make sure the direction is in the range of [-180° ~ 180°]
        UE_index = np.where(self.UE_direction >= 180)
        self.UE_direction[UE_index] -= 360
        
        # Situation2 : Out of the right boundary (x >= 3R)
        UE_index = np.where(self.UE_pos[:, 0] >= 3 * self.BS_radius)  # find the ue_ids of UEs that there x are >= 3R
        self.UE_pos[UE_index, 0] = 6 * self.BS_radius - self.UE_pos[UE_index, 0]  # set the bounce back x position
        self.UE_direction[UE_index] = 180 - self.UE_direction[UE_index]  # set the direction (horizontal reflection)
        # make sure the direction is in the range of [-180° ~ 180°]
        UE_index = np.where(self.UE_direction >= 180)
        self.UE_direction[UE_index] -= 360
        
        # Situation3 : Out of the bottom boundary (y < -3R)
        UE_index = np.where(self.UE_pos[:, 1] < -3 * self.BS_radius)  # find the ue_ids of UEs that there y are < -3R
        self.UE_pos[UE_index, 1] = -6 * self.BS_radius - self.UE_pos[UE_index, 1]  # set the bounce back y position
        self.UE_direction[UE_index] = -self.UE_direction[UE_index]  # set the direction (horizontal reflection)

        # Situation4 : Out of the top boundary (y > 3R)
        UE_index = np.where(self.UE_pos[:, 1] >= 3 * self.BS_radius)  # find the ue_ids of UEs that there y are >= 3R
        self.UE_pos[UE_index, 1] = 6 * self.BS_radius - self.UE_pos[UE_index, 1]  # set the bounce back y position
        self.UE_direction[UE_index] = -self.UE_direction[UE_index]  # set the direction (horizontal reflection)

        # update self.UE_cell
        self.UE_cell = np.zeros(self.UE_max_no)  # reset
        self.UE_cell[np.where(np.sum(self.UE_pos ** 2, axis=1) <= self.BS_radius ** 2)] = 1  # check all the UE again
        
        # calculate the ratio of users near the BS (distance <= R) to the total users in each network slice
        # ex. volte users whose distance is <= R / all of the volte users
        # recall : UE_pos.shape = (UE_max_no, 2)
        # case1 : volte
        tmp_u = self.UE_pos[np.where(self.UE_cat == 'volte')]  # the positions of users using volte
        tmp_dis = np.sum(tmp_u ** 2, axis = 1)  # the distances of users using volte [ [x1^2 + y1^2], [x2^2 + y2^2], ... ]
        n1 = np.sum(tmp_dis <= (4 * self.BS_radius / 4) ** 2)  # n1 : no. of users whose distance is <= R
        self.volte_dis = n1 / tmp_u.shape[0]  # calculate the ratio
        
        # case2: embb_general
        tmp_u = self.UE_pos[np.where(self.UE_cat == 'embb_general')]  # the positions of users using embb_general
        tmp_dis = np.sum(tmp_u ** 2, axis = 1)
        n2 = np.sum(tmp_dis <= (4 * self.BS_radius / 4) ** 2)
        self.embb_dis = n2 / tmp_u.shape[0]
        
        # case3: urllc
        tmp_u = self.UE_pos[np.where(self.UE_cat == 'urllc')]  # the positions of users using urllc
        tmp_dis = np.sum(tmp_u ** 2, axis = 1)
        n3 = np.sum(tmp_dis <= (4 * self.BS_radius / 4) ** 2)
        self.urllc_dis = n3 / tmp_u.shape[0]

    #=======================================================================================================================================#
    # allocate the bandwidth to each UE (granularity: RB)
    # 2 ways : 
    #     RR : initially allocate the RBs equally among all users in each network slice then use round-robin (RR) to allocate the remaining bandwidth that cannot be evenly divided
    #     RR_nons : allocate the RBs equally among all users across all the network slices
    def scheduling(self):
        self.UE_band = np.zeros(self.UE_max_no)  # store the allocated resource of each UE

        # Method1 : RR
        if self.schedu_method == 'round_robin':
            ser_cat = len(self.ser_cat)
            band_ser_cat = self.band_ser_cat  # bandwidth allocated to each NS in the current window

            # initialize self.ser_schedu_ind every windows
            # self.ser_schedu_ind -> the last user who was assigned a RB in the previous round-robin (RR) allocation phase
            if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):
                self.ser_schedu_ind = [0] * ser_cat

            # consider 1 NS at a time
            for i in range(ser_cat):
                # extract the Active Users' ue_id
                # Active -> using ser_cat i、in the coverage area、the corresponding buffer isn't empty (determine whether index0 is empty)
                UE_index = np.where( (self.UE_cell == 1) &
                                     (self.UE_buffer[0, :] != 0) &
                                     (self.UE_cat == self.ser_cat[i]) )[0]
                UE_Active_No = len(UE_index)
                # if no active users then do nothing
                if UE_Active_No != 0:
                    # Step1 : divide the allocated band into several RBs (1RB = 180kHz)
                    RB_No = band_ser_cat[i] // (180 * 10 ** 3)  
                    RB_round = RB_No // UE_Active_No   
                    self.UE_band[UE_index] += 180 * 10 ** 3 * RB_round  # allocate the RBs among all active users equally
                    # Step2 : allocate the not evenly RBs by RR starts from ser_schedu_ind[i]
                    RB_rem_no = int(RB_No - RB_round * UE_Active_No)  # no. of not evenly divided RBs
                    left_no = np.where(UE_index > self.ser_schedu_ind[i])[0].size  # no. of remaining active users (starts from ser_schedu_ind[i])
                    if left_no >= RB_rem_no:  # no need to roll back to the front
                        UE_act_index = UE_index[np.where(np.greater_equal(UE_index, self.ser_schedu_ind[i]))]
                        UE_act_index = UE_act_index[:RB_rem_no]
                        if UE_act_index.size != 0:
                            self.UE_band[UE_act_index] += 180 * 10 ** 3
                            self.ser_schedu_ind[i] = UE_act_index[-1] + 1
                    else:  # roll back to the front
                        UE_act_index_par1 = UE_index[np.where(UE_index > self.ser_schedu_ind[i])]
                        UE_act_index_par2 = UE_index[0:RB_rem_no - left_no]
                        self.UE_band[np.hstack((UE_act_index_par1, UE_act_index_par2))] += 180 * 10 ** 3
                        self.ser_schedu_ind[i] = UE_act_index_par2[-1] + 1

        # Method2 : RR_nons
        elif self.schedu_method == 'round_robin_nons':
            band_whole = self.band_whole

            # initialize at the begining of the entire simulation
            if self.sys_clock == self.time_subframe:
                self.ser_schedu_ind = 0

            UE_index = np.where((self.UE_buffer[0, :] != 0))[0]  # extract the Active Users' index across all of the NSs
            UE_Active_No = len(UE_index)
            if UE_Active_No != 0:
                # Step1 : divides the band into several RBs and allocate them equally among all active users among all NSs
                RB_No = band_whole // (180 * 10 ** 3)
                RB_round = RB_No // UE_Active_No
                self.UE_band[UE_index] += 180 * 10 ** 3 * RB_round
                # Step2 : allocate the not evenly RBs by RR starts from ser_schedu_ind[i]
                RB_rem_no = RB_No % UE_Active_No
                left_no = np.where(UE_index > self.ser_schedu_ind)[0].size
                if left_no >= RB_rem_no:  # no need to roll back
                    UE_act_index = UE_index[np.where(np.logical_and(np.greater_equal(UE_index, self.ser_schedu_ind),
                                                                    np.less(UE_index, RB_rem_no + self.ser_schedu_ind)))]
                    if UE_act_index.size != 0:
                        self.UE_band[UE_act_index] += 180 * 10 ** 3
                        self.ser_schedu_ind = UE_act_index[-1] + 1
                else:  # need to roll back
                    UE_act_index_par1 = UE_index[np.where(UE_index > self.ser_schedu_ind)]
                    UE_act_index_par2 = UE_index[0:RB_rem_no - left_no]
                    self.UE_band[np.hstack((UE_act_index_par1, UE_act_index_par2))] += 180 * 10 ** 3
                    self.ser_schedu_ind = UE_act_index_par2[-1] + 1

            # reset band_ser_cat every window
            if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):
                self.band_ser_cat = np.zeros(len(self.ser_cat))
            
            # calculate the average allocated band in 1 timeslot of the current window of each NSs
            for i in range(len(self.ser_cat)):
                if (self.sys_clock * 10000) % (self.learning_windows * 10000) == (self.time_subframe * 10000):  # 1st timeslot in the simulation
                    self.band_ser_cat[i] = np.sum(self.UE_band[self.UE_cat == self.ser_cat[i]])
                else:  # not the 1st timeslot in the simulation
                    self.band_ser_cat[i] += np.sum(self.UE_band[self.UE_cat == self.ser_cat[i]])  # accumulate the allocated resources in each timeslots of each NSs
                    if (self.sys_clock * 10000) % (self.learning_windows * 10000) == 0:  # last timeslot in the current learning window
                        lw = (self.learning_windows * 10000) / (self.time_subframe * 10000)  # calculate no. of timeslot in 1 window
                        self.band_ser_cat[i] = self.band_ser_cat[i] / lw  # calculate the average allocated band in a timeslot of each NSs
    
    #=======================================================================================================================================#
    # Calculate Data transmission rate of each UE according to Shannon Thoery
    # Implement packet transmission according to the estimated rate and calculate the reward of the current timeslot
    # update the buffer & latency after the packet transmission of this timeslot
    def provisioning(self):
        UE_index = np.where(self.UE_band != 0)  # extract the UE who's allocated RBs
        self.channel_model()  # calculate the channel lost of all users (output_shape: (UE_max_no, 1))
        # calculate the received power of all users considering large-scale fading & convert the unit to W (output_shape: (UE_max_no, 1))
        rx_power = 10 ** ((self.BS_tx_power - self.chan_loss + self.UE_rx_gain) / 10)  
        rx_power = rx_power.reshape(1, -1)[0]  # output_shape change from (UE_max_no, 1) to (1, UE_max_no) and extract the (UE_max_no) part by [0]
        # calculate the data transmission rate of each user according to Shannon Theory
        rate = np.zeros(self.UE_max_no)  # unit : bit/s
        rate[UE_index] = self.UE_band[UE_index] * np.log10(1 + rx_power[UE_index] / (10 ** (self.noise_PSD / 10) * self.UE_band[UE_index])) * self.dl_mimo
        # update the latency of each non-zero packet (add 0.5ms)
        self.UE_latency[np.where(self.UE_buffer != 0)] += self.time_subframe
        # do the packet transmission 
        for ue_id in UE_index[0]:
            self.UE_buffer[:, ue_id] = bufferUpdate(self.UE_buffer[:, ue_id], rate[ue_id], self.time_subframe)
        # calculate the reward of the current timeslot
        self.store_reward(rate)
        # update UE_buffer & UE_latency
        # 1. packet_size = 0, latency != 0 -> reset the corresponding latency to 0
        # 2. packet_size != 0 -> move the packet forward in the queue
        self.bufferClear()  

    #=======================================================================================================================================#
    # Generate the readtime of each UE according to their belonging NSs & Generate packet according to that readtime
    # if the corresponding queue of UE is full (5) than don't generate a packet
    def activity(self):  # https://www.ngmn.org/fileadmin/user_upload/NGMN_Radio_Access_Performance_Evaluation_Methodology.pdf
        # VoLTE uses the VoIP model
        # embb_general uses the video streaming model
        # urllc uses the FTP2 model
        
        # Generate the readtime of each user according to their belonging NSs
        if self.sys_clock == 0:
            # consider 1 NS at a time
            for ser_name in self.ser_cat:
                ue_index = np.where((self.UE_cat == ser_name) & (self.UE_cell == 1))  # extract the users located in the coverage area
                ue_index_Size = ue_index[0].size
                # case1 : volte
                if ser_name == 'volte':
                    # readtime is randomly generated from a uniform distribution over (0, 160ms)
                    # output_shape (1, ue_index_Size)
                    self.UE_readtime[ue_index] = np.random.uniform(0, 160 * 10 ** (-3), [1, ue_index_Size])  # the silence lasts 160 ms in maximum
                # case2 : embb_general
                elif ser_name == 'embb_general':
                    # readtime is generated from a Pareto distribution with a parameter 1.2 & base = 6ms
                    tmp_readtime = np.random.pareto(1.2, [1, ue_index_Size]) * 6 * 10 ** -3
                    tmp_readtime[tmp_readtime > 12.5 * 10 ** -3] = 12.5 * 10 ** -3  # upper bound is 12.5ms
                    self.UE_readtime[ue_index] = tmp_readtime
                # case3 : urllc
                elif ser_name == 'urllc':
                    # use exponential distribution to simulate suddenly burst packets 
                    # read time is determines much smaller; the spec shows the average time is 180s, but here it is defined as 180 ms
                    self.UE_readtime[ue_index] = np.random.exponential(180 * 10 ** -3, [1, ue_index_Size])  

        # extract the UEs with readtimes lower than 0
        UE_index_readtime = np.where(self.UE_readtime <= 0)[0].tolist()
        
        # Generate packet to the queue of each user according to the readtime 
        # consider 1 UE at a time
        for ue_id in UE_index_readtime:
            if self.UE_buffer[:, ue_id].size - np.count_nonzero(self.UE_buffer[:, ue_id]) != 0:  # The buffer is not full (by 5 - no. of packets in the corresponding queue)
                buf_ind = np.where(self.UE_buffer[:, ue_id] == 0)[0][0]  # find the first empty place to place the newly generated packet start from index0 
                # generate packet according to the UE's belonging NSs
                # case1 : volte
                if self.UE_cat[ue_id] == 'volte':
                    self.UE_buffer[buf_ind, ue_id] = 40 * 8  # packet size = 320 bits
                    self.tx_bit_no[0] += 40 * 8
                    self.UE_readtime[ue_id] = np.random.uniform(0, 160 * 10 ** (-3), 1).squeeze()  # generate a new readtime
                # case2 : embb_general
                elif self.UE_cat[ue_id] == 'embb_general':
                    tmp_buffer_size = np.random.pareto(1.2, 1).squeeze() * 800  # packet size is generated from a Pareto distribution with a parameter 1.2 & base = 800 bits
                    if tmp_buffer_size > 2000:  # upper bound is set to 2000 bits
                        tmp_buffer_size = 2000
                    # tmp_buffer_size = np.random.choice([1*8*10**6, 2*8*10**6, 3*8*10**6, 4*8*10**6, 5*8*10**6])
                    self.UE_buffer[buf_ind, ue_id] = tmp_buffer_size
                    self.tx_bit_no[1] += tmp_buffer_size
                    self.UE_readtime[ue_id] = np.random.pareto(1.2, 1).squeeze() * 6 * 10 ** -3  # generate a new readtime
                    if self.UE_readtime[ue_id] > 12.5 * 10 ** -3:
                        self.UE_readtime[ue_id] = 12.5 * 10 ** -3
                # case3 : urllc
                elif self.UE_cat[ue_id] == 'urllc':
                    # method1 : lognormal distribution
                    # tmp_buffer_size = np.random.lognormal(14.45,0.35,[1,1])
                    # if tmp_buffer_size > 5 * 10 **6:
                    #      tmp_buffer_size > 5 * 10 **6
                    
                    # method2 : small packet size
                    tmp_buffer_size = np.random.choice([6.4*8*10**3, 12.8*8*10**3, 19.2*8*10**3, 25.6*8*10**3, 32*8*10**3])
                    # tmp_buffer_size = np.random.choice([0.3 * 8 * 10 ** 6])
                    # method3 : large packet size
                    # tmp_buffer_size = np.random.choice(
                    #     [0.3 * 8 * 10 ** 6, 0.4 * 8 * 10 ** 6, 0.5 * 8 * 10 ** 6, 0.6 * 8 * 10 ** 6,
                    #      0.7 * 8 * 10 ** 6])
                    self.UE_buffer[buf_ind, ue_id] = tmp_buffer_size
                    self.tx_bit_no[2] += tmp_buffer_size
                    # generate a new readtime
                    # read time is determines much smaller; the spec shows the average time is 180s, but here it is defined as 180 ms
                    self.UE_readtime[ue_id] = np.random.exponential(180 * 10 ** -3, 1).squeeze()
                # update the backup buffer
                self.UE_buffer_backup[buf_ind, ue_id] = self.UE_buffer[buf_ind, ue_id]
                # record the no. of the packets
                self.tx_pkt_no[self.ser_cat.index(self.UE_cat[ue_id])] += 1

            else:  # the corresponding queue is full, don't generate packet this time, generate a new readtime
                if self.UE_cat[ue_id] == 'volte':
                    self.UE_readtime[ue_id] = np.random.uniform(0, 160 * 10 ** (-3), 1).squeeze()
                    self.tx_bit_no[0] += 40 * 8
                elif self.UE_cat[ue_id] == 'embb_general':
                    self.UE_readtime[ue_id] = np.random.pareto(1.2, 1).squeeze() * 6 * 10 ** -3
                    tmp_buffer_size = np.random.pareto(1.2, 1).squeeze() * 800
                    self.tx_bit_no[1] += tmp_buffer_size
                else:  # urllc
                    self.UE_readtime[ue_id] = np.random.exponential(180 * 10 ** -3, 1).squeeze()
                    self.tx_bit_no[2] += np.random.choice([6.4*8*10**3, 12.8*8*10**3, 19.2*8*10**3, 25.6*8*10**3, 32*8*10**3])  # packet size of the dropped packet 
                # record the no. of the dropped packets
                self.drop_pkt_no[self.ser_cat.index(self.UE_cat[ue_id])] += 1

        # update the timing sequence
        self.UE_readtime[np.where(self.UE_cell == 1)] -= self.time_subframe  # update the readtime (-0.5ms)
        self.sys_clock += self.time_subframe  # update the sys_clock
        self.sys_clock = round(self.sys_clock, 4)

    #=======================================================================================================================================#
    # Generate the pending packets & no. of UEs in the coverage area of each NS in the current timeslot
    def get_state(self):
        pkt = self.tx_pkt_no + self.drop_pkt_no
        dis = np.array([self.volte_dis, self.embb_dis, self.urllc_dis])
        total_bits = self.tx_bit_no
        return pkt, total_bits

    #=======================================================================================================================================#
    def store_reward(self, rate):
        se = np.zeros(len(self.ser_cat))
        ee = np.zeros(len(self.ser_cat))
        sys_rate_frame = 0  # accumulate the data transmission rates of every UEs in the current window
        # consider 1 NS at a time
        for ser_name in self.ser_cat:
            ser_index = self.ser_cat.index(ser_name)  # considering NS's index
            ue_index_ = np.where(self.UE_cat == ser_name)  # UEs belonging to the considering NS
            allo_band = np.sum(self.UE_band[ue_index_])  # band allocated to the considering NS in this window
            sum_rate = np.sum(rate[ue_index_])  # sum of data transmission rates of all UEs belonging to the considering NS
            if allo_band != 0:
                sys_rate_frame += sum_rate  # add the sum_rate in the current timeslot to sys_rate_frame
                se[ser_index] = sum_rate / allo_band  # SE in the current timeslot of the considering NS
                ee[ser_index] = se[ser_index] / 10 ** (self.BS_tx_power / 10)  # EE in the current timeslot of the considering NS
        self.sys_se_per_frame += sys_rate_frame / self.band_whole  # SE of the system in the current timeslot

        # no idle situation in this paper
        if sys_rate_frame == 0:
            self.idle_frame += 1
        
        # do nothing
        handling_latency = 2 * 10 ** (-3)
        handling_latency = 0

        # calculate the no. of sucessfully transmitted packets of each UE in the current timeslot
        UE_index = np.where(self.UE_cell == 1)[0]  # extract all UEs in the coverage area
        # consider 1 UE at a time
        for ue_id in UE_index:
            # consider every packets in the corresponding queue
            for i in range(self.UE_latency[:, ue_id].size):  # i = 1~5
                if (self.UE_buffer[i, ue_id] == 0) & (self.UE_latency[i, ue_id] != 0):
                    # case1 : volte SLA -> rate >= 51kbps、latency < 10ms
                    if self.UE_cat[ue_id] == 'volte':
                        cat_index = self.ser_cat.index('volte')
                        if (self.UE_latency[i, ue_id] == self.time_subframe):  # take 1 timeslot to do the transmission
                            if (rate[ue_id] >= 51 * 10 ** 3) & (
                                    self.UE_latency[i, ue_id] < 10 * 10 ** (-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                        else:  # take more than 1 timeslot
                            if (self.UE_buffer_backup[i, ue_id] / self.UE_latency[i, ue_id] >= 51 * 10 ** 3) & (
                                    self.UE_latency[i, ue_id] < 10 * 10 ** (-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                    # case2 : embb_general SLA -> rate >= 100Mbs、latency < 10ms
                    elif self.UE_cat[ue_id] == 'embb_general':
                        cat_index = self.ser_cat.index('embb_general')
                        if (self.UE_latency[i, ue_id] == self.time_subframe):  # take 1 timeslot to do the transmission
                            # if (rate[ue_id] >= 5 * 10 ** 6) & (self.UE_latency[i,ue_id] < 10 * 10 **(-3) - handling_latency):
                            if (rate[ue_id] >= 100 * 10 ** 6) & (
                                    self.UE_latency[i, ue_id] < 10 * 10 ** (-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                        else:  # take more than 1 timeslot
                            if (self.UE_buffer_backup[i, ue_id] / self.UE_latency[i, ue_id] >= 100 * 10 ** 6) & (
                                    self.UE_latency[i, ue_id] < 10 * 10 ** (-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                    # case3 : urllc SLA -> rate >= 10Mbs、latency < 1ms
                    elif self.UE_cat[ue_id] == 'urllc':
                        cat_index = self.ser_cat.index('urllc')
                        if (self.UE_latency[i, ue_id] == self.time_subframe):  # take 1 timeslot to do the transmission
                            if (rate[ue_id] >= 10 * 10 ** 6) & (
                                    self.UE_latency[i, ue_id] < 1 * 10 ** (-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1
                        else: # take more than 1 timeslot
                            if (self.UE_buffer_backup[i, ue_id] / self.UE_latency[i, ue_id] >= 10 * 10 ** 6) & (
                                    self.UE_latency[i, ue_id] < 1 * 10 ** (-3) - handling_latency):
                                self.succ_tx_pkt_no[cat_index] += 1

    #=======================================================================================================================================#
    # return the qoe of each NS & average system SE of 1 timeslot in the current window 
    def get_reward(self):
        se_total = self.sys_se_per_frame / (self.learning_windows / self.time_subframe - self.idle_frame)  # average SE of one timeslot of the system in the current window
        # ee_total = se_total/10**(self.BS_tx_power/10)
        self.tx_pkt_no[np.where(self.tx_pkt_no == 0)] += 1
        qoe = self.succ_tx_pkt_no / (self.tx_pkt_no + self.drop_pkt_no)  # qoe of each NS in the current window
        return qoe, se_total

    #=======================================================================================================================================#
    # update the UE_buffer & UE_latency after each timeslot
    # packets which haven't completely sent will resume there transmissions in the next timeslot 
    # packets which finish there transmission will be clean by setting UE_buffer[finish_packet_index] = 0 & UE_latency[finish_packet_index] = 0
    def bufferClear(self):
        latency = np.sum(self.UE_latency, axis = 0)  # shape : (UE_max_no)
        UE_index = np.where(latency != 0)  # store ue_ids if the corresponding UEs have any packet with latency != 0
        bufSize = self.UE_latency[:, 0].size  # bufSize = UE_max_no

        # consider 1 UE at a time
        for ue_id in UE_index[0]:
            # temp
            buffer_ = self.UE_buffer[:, ue_id].copy()  # self.UE_buffer[:, ue_id] : the Queue corresponds to UE with ue_id
            buffer_bk = self.UE_buffer_backup[:, ue_id].copy()  # self.UE_buffer_backup[:, ue_id] : the backup Queue corresponds to UE with ue_id
            latency_ = self.UE_latency[:, ue_id].copy()  # self.UE_latency[:, ue_id] : the Latency Queue corresponds to UE with ue_id
            
            # Situation1 : packets have finished their transmission but their latency times haven't reset to 0
            ind_1 = np.where( np.logical_and( buffer_ == 0, latency_ != 0 ) )  # find the packets' index which their packet size = 0 & latency = 0
            indSize_1 = ind_1[0].size
            if indSize_1 != 0:
                # reset the corresponding packet size = 0 & latency = 0
                self.UE_latency[ind_1, ue_id] = np.zeros(indSize_1)
                self.UE_buffer_backup[ind_1, ue_id] = np.zeros(indSize_1)
            
            # Situation2 : packets haven't finished their transmission. store there datas to the next timeslot
            ind = np.where(np.logical_and(buffer_ != 0, latency_ != 0))  # find the packets' index which their packet size != 0 & latency != 0
            ind = ind[0]  # take those indices out
            indSize = ind.size
            if indSize != 0:
                # reset 
                self.UE_buffer[:, ue_id] = np.zeros(bufSize)
                self.UE_latency[:, ue_id] = np.zeros(bufSize)
                self.UE_buffer_backup[:, ue_id] = np.zeros(bufSize)
                # store the values back (start from index0)
                # move the unfinished packets from back to front
                self.UE_buffer[:indSize, ue_id] = buffer_[ind]
                self.UE_latency[:indSize, ue_id] = latency_[ind]
                self.UE_buffer_backup[:indSize, ue_id] = buffer_bk[ind]

    #=======================================================================================================================================#
    def countReset(self):
        self.sys_clock = 0
        self.UE_readtime = np.ones(self.UE_max_no)
        self.tx_pkt_no = np.zeros(len(self.ser_cat))
        self.tx_bit_no = np.zeros(len(self.ser_cat))
        self.drop_pkt_no = np.zeros(len(self.ser_cat))
        self.idle_frame = 0
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
# simulate the packet transmission from BS to UE with ue_id, starts from index0 in each queue
# buffer -> UE_buffer[:, ue_id] : packet size of each of the 5 packets in the queue corresponding to the UE with ue_id
# rate -> data transmissio rate of UE with ue_id
def bufferUpdate(buffer, rate, time_subframe):
    bSize = buffer.size
    for i in range(bSize):
        if buffer[i] >= rate * time_subframe:
            buffer[i] -= rate * time_subframe
            rate = 0
            break
        else:
            rate -= buffer[i] / time_subframe
            buffer[i] = 0
    return buffer

#=======================================================================================================================================#
# update the latency of the packets in the queue corresponding to the UE with ue_id by adding 0.5ms (one timeslot) to the packets
# latency -> UE_latency[:, ue_id] : packet latency of each of the 5 packets in the queue corresponding to the UE with ue_id
# buffer -> UE_buffer[:, ue_id] : packet size of each of the 5 packets in the queue corresponding to the UE with ue_id
# isn't used in this env.
def latencyUpdate(latency, buffer, time_subframe):
    lSize = latency.size
    for i in range(lSize):
        if buffer[i] != 0:
            latency[i] += time_subframe
    return latency
