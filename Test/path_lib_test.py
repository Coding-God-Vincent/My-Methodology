from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


# log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs")
# exp_path = log_path / "D2AC" / "exp1"
# 於 exp_path 中建立一個資料夾 (exp_path 最後要包含該資料夾的名字，本例中的資料夾名稱為 exp1)
# parents= True : 若目標資料夾前面是空的，全幫我補上資料夾
# exist_ok= True : 即使前面資料夾都存在也沒關係
# 常看到 mkdir 中一次出現這兩個參數
# exp_path.mkdir(parents= True, exist_ok= True)



algo_name = 'D2AC'
exp_name = 'exp1'
log_path = Path("/home/super_trumpet/NCKU/Paper/My Methodology/Logs") / algo_name / exp_name / 'tensorboard'
# log_path.mkdir(parents= True, exist_ok= True)

writer = SummaryWriter(log_dir= log_path)

hparams_dict = {
    "lr": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    "env": "YourEnvName",
    "seed": 1,
}

for i in range(10):
    a = 5
    # add_scalar('在 tensor board 上顯示的標籤名', 數值 (csv 的 value 欄位), x 軸 (csv 的 step 欄位))
    b = 10
    writer.add_scalar("a", a, i)
    writer.add_scalar("b", b, i)
final_a = 100
final_b = 101
metric_dict = {
    "final outcome/final_a": final_a,
    "final_outcome/final_b": final_b
}
# 一定要有兩個 hparams_dict & metric_dict 兩個參數
writer.add_hparams(hparam_dict= hparams_dict, metric_dict= metric_dict)