import torch
import os
import glob

# 你的路径
file_dir = "/data2/codefile/fjh/data/feats_pgd_answer_clean"

# 获取一个文件
pt_files = glob.glob(os.path.join(file_dir, "*.pt"))

if not pt_files:
    print(f"Error: No .pt files found in {file_dir}")
else:
    sample_file = pt_files[0]
    print(f"Checking file: {sample_file}")
    
    # 加载特征
    data = torch.load(sample_file, map_location='cpu')
    
    if isinstance(data, torch.Tensor):
        print(f"Type: Tensor")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    else:
        print(f"Type: {type(data)}")
        print("Content is not a raw Tensor.")
