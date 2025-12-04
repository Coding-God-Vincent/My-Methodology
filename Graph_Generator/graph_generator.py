import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Exponential Moving Average, bigger the weight (0~1) smoother the line
def ema(values, weight= 0.9):
    values = np.asarray(values, dtype=float)
    smoothed = np.zeros_like(values)  # 創建一個跟 values 一樣大的 np.zeros
    last = values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        smoothed[i] = last
    return smoothed

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''Uitility'''
# steps = np.arange(10000)
# d2ac_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_csv/utility.csv")
# ganddqn_utility = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/GANDDQN_csv/utility.csv")

# smooth_d2ac = ema(d2ac_utility['Value'], weight= 0.9)
# lower_d2ac = np.minimum(d2ac_utility['Value'], smooth_d2ac)
# upper_d2ac = np.maximum(d2ac_utility['Value'], smooth_d2ac)

# smooth_ganddqn = ema(ganddqn_utility['Value'], weight= 0.9)
# lower_ganddqn = np.minimum(ganddqn_utility['Value'], smooth_ganddqn)
# upper_ganddqn = np.maximum(ganddqn_utility['Value'], smooth_ganddqn)

# plt.figure(0)
# plt.clf()
# plt.title('Utility')
# plt.xlabel('Episode')
# plt.ylabel('utility')
# plt.plot(smooth_d2ac, label= 'D2AC', color= 'red')
# plt.fill_between(x= steps, y1= lower_d2ac, y2= upper_d2ac, color= 'red', alpha= 0.15)
# plt.plot(smooth_ganddqn, label= 'GANDDQN', color= 'blue')
# plt.fill_between(x= steps, y1= lower_ganddqn, y2= upper_ganddqn, color= 'blue', alpha= 0.15)
# plt.legend()
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/Utility')


# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''SE'''
# steps = np.arange(10000)
# d2ac_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_csv/se.csv")
# ganddqn_se = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/GANDDQN_csv/se.csv")

# smooth_d2ac = ema(d2ac_se['Value'], weight= 0.9)
# lower_d2ac = np.minimum(d2ac_se['Value'], smooth_d2ac)
# upper_d2ac = np.maximum(d2ac_se['Value'], smooth_d2ac)

# smooth_ganddqn = ema(ganddqn_se['Value'], weight= 0.9)
# lower_ganddqn = np.minimum(ganddqn_se['Value'], smooth_ganddqn)
# upper_ganddqn = np.maximum(ganddqn_se['Value'], smooth_ganddqn)

# plt.figure(1)
# plt.clf()
# plt.title('SE')
# plt.xlabel('Episode')
# plt.ylabel('SE')
# plt.plot(smooth_d2ac, label= 'D2AC', color= 'red')
# plt.fill_between(x= steps, y1= lower_d2ac, y2= upper_d2ac, color= 'red', alpha= 0.15)
# plt.plot(smooth_ganddqn, label= 'GANDDQN', color= 'blue')
# plt.fill_between(x= steps, y1= lower_ganddqn, y2= upper_ganddqn, color= 'blue', alpha= 0.15)
# plt.legend()
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/SE')

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''QoE of D2AC'''
# steps = np.arange(10000)
# d2ac_embb = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_csv/qoe_embb_general.csv")
# d2ac_urllc = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_csv/qoe_urllc.csv")
# d2ac_volte = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_csv/qoe_volte.csv")

# smooth_embb = ema(d2ac_embb['Value'], weight= 0.9)
# lower_embb = np.minimum(d2ac_embb['Value'], smooth_embb)
# upper_embb = np.maximum(d2ac_embb['Value'], smooth_embb)

# smooth_urllc = ema(d2ac_urllc['Value'], weight= 0.9)
# lower_urllc = np.minimum(d2ac_urllc['Value'], smooth_urllc)
# upper_urllc = np.maximum(d2ac_urllc['Value'], smooth_urllc)

# smooth_volte = ema(d2ac_volte['Value'], weight= 0.9)
# lower_volte = np.minimum(d2ac_volte['Value'], smooth_volte)
# upper_volte = np.maximum(d2ac_volte['Value'], smooth_volte)


# plt.figure(2)
# plt.clf()
# plt.title('D2AC QoE')
# plt.xlabel('Episode')
# plt.ylabel('QoE')
# plt.plot(smooth_embb, label= 'video', color= 'orange')
# plt.fill_between(x= steps, y1= lower_embb, y2= upper_embb, color= 'orange', alpha= 0.15)
# plt.plot(smooth_urllc, label= 'urllc', color= 'green')
# plt.fill_between(x= steps, y1= lower_urllc, y2= upper_urllc, color= 'green', alpha= 0.15)
# plt.plot(smooth_volte, label= 'volte', color= 'blue')
# plt.fill_between(x= steps, y1= lower_volte, y2= upper_volte, color= 'blue', alpha= 0.15)
# plt.legend()
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/D2AC_QoE')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''QoE of GANDDQN'''
# steps = np.arange(10000)
# ganddqn_embb = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/GANDDQN_csv/qoe_embb_general.csv")
# ganddqn_urllc = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/GANDDQN_csv/qoe_urllc.csv")
# ganddqn_volte = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/GANDDQN_csv/qoe_volte.csv")

# smooth_embb = ema(ganddqn_embb['Value'], weight= 0.9)
# lower_embb = np.minimum(ganddqn_embb['Value'], smooth_embb)
# upper_embb = np.maximum(ganddqn_embb['Value'], smooth_embb)

# smooth_urllc = ema(ganddqn_urllc['Value'], weight= 0.9)
# lower_urllc = np.minimum(ganddqn_urllc['Value'], smooth_urllc)
# upper_urllc = np.maximum(ganddqn_urllc['Value'], smooth_urllc)

# smooth_volte = ema(ganddqn_volte['Value'], weight= 0.9)
# lower_volte = np.minimum(ganddqn_volte['Value'], smooth_volte)
# upper_volte = np.maximum(ganddqn_volte['Value'], smooth_volte)


# plt.figure(3)
# plt.clf()
# plt.title('GANDDQN QoE')
# plt.xlabel('Episode')
# plt.ylabel('QoE')
# plt.plot(smooth_embb, label= 'video', color= 'orange')
# plt.fill_between(x= steps, y1= lower_embb, y2= upper_embb, color= 'orange', alpha= 0.15)
# plt.plot(smooth_urllc, label= 'urllc', color= 'green')
# plt.fill_between(x= steps, y1= lower_urllc, y2= upper_urllc, color= 'green', alpha= 0.15)
# plt.plot(smooth_volte, label= 'volte', color= 'blue')
# plt.fill_between(x= steps, y1= lower_volte, y2= upper_volte, color= 'blue', alpha= 0.15)
# plt.legend()
# plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_movingUE_env/Combine/GANDDQN_QoE')


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''ObservationBits'''
steps = np.arange(10000)
d2ac_embb = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/D2AC_csv/observationBits_embb_general.csv")
d2ac_urllc = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/D2AC_csv/observationBits_urllc.csv")
d2ac_volte = pd.read_csv("/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/Combine/D2AC_csv/observationBits_volte.csv")

smooth_embb = ema(d2ac_embb['Value'], weight= 0.9)
lower_embb = np.minimum(d2ac_embb['Value'], smooth_embb)
upper_embb = np.maximum(d2ac_embb['Value'], smooth_embb)

smooth_urllc = ema(d2ac_urllc['Value'], weight= 0.9)
lower_urllc = np.minimum(d2ac_urllc['Value'], smooth_urllc)
upper_urllc = np.maximum(d2ac_urllc['Value'], smooth_urllc)

smooth_volte = ema(d2ac_volte['Value'], weight= 0.9)
lower_volte = np.minimum(d2ac_volte['Value'], smooth_volte)
upper_volte = np.maximum(d2ac_volte['Value'], smooth_volte)


plt.figure(2)
plt.clf()
plt.title('FixedUE_env Observation Bits')
plt.xlabel('Episode')
plt.ylabel('Bits')
plt.plot(smooth_embb, label= 'video', color= 'orange')
plt.fill_between(x= steps, y1= lower_embb, y2= upper_embb, color= 'orange', alpha= 0.15)
plt.plot(smooth_urllc, label= 'urllc', color= 'green')
plt.fill_between(x= steps, y1= lower_urllc, y2= upper_urllc, color= 'green', alpha= 0.15)
plt.plot(smooth_volte, label= 'volte', color= 'blue')
plt.fill_between(x= steps, y1= lower_volte, y2= upper_volte, color= 'blue', alpha= 0.15)
plt.legend()
plt.savefig('/home/super_trumpet/NCKU/Paper/My Methodology/Outcomes/Outcome_fixedUE_env/FixedUE_Observation')
