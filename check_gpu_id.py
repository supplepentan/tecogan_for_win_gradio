import torch

# 利用可能なGPUの数を取得
num_gpus = torch.cuda.device_count()

# 利用可能なGPUのIDを列挙
if num_gpus > 0:
    print(f"利用可能なGPUの数: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("利用可能なGPUはありません。")
