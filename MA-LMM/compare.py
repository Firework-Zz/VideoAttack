import torch
import torch.nn.functional as F
import os

# ================= 配置路径 =================
# 原始特征 (Clean)
file_clean = "/data2/codefile/fjh/data/feats_pgd_answer_clean/video7010.pt"
# 攻击后特征 (Adversarial)
file_adv = "/data2/codefile/fjh/data/feats_pgd_answer/video7010.pt"

def compare_features(path1, path2):
    if not os.path.exists(path1):
        print(f"❌ 文件不存在: {path1}")
        return
    if not os.path.exists(path2):
        print(f"❌ 文件不存在: {path2}")
        return

    print(f"Loading files...")
    # 使用 cpu 加载，并转为 float32 以保证计算精度
    feat1 = torch.load(path1, map_location='cpu').float()
    feat2 = torch.load(path2, map_location='cpu').float()

    print(f"\n{'='*20} 基本信息 {'='*20}")
    print(f"Shape Clean: {feat1.shape}")
    print(f"Shape Adv  : {feat2.shape}")
    
    if feat1.shape != feat2.shape:
        print("⚠️ 警告：两个特征文件的形状不一致！后续计算将基于 Flatten 后的数据。")

    # 拉平以便计算全局指标
    f1_flat = feat1.flatten()
    f2_flat = feat2.flatten()
    
    diff = f2_flat - f1_flat
    
    # ================= 计算指标 =================
    # 1. 余弦相似度 (Cosine Similarity)
    # dim=0 表示对拉平后的向量计算
    cosine_sim = F.cosine_similarity(f1_flat.unsqueeze(0), f2_flat.unsqueeze(0)).item()
    
    # 2. L_inf 距离 (最大绝对误差) - 最关键的指标，对应 EPS
    l_inf_dist = torch.max(torch.abs(diff)).item()
    
    # 3. L2 距离 (欧氏距离)
    l2_dist = torch.norm(diff, p=2).item()
    
    # 4. 平均绝对误差 (MAE)
    mae = torch.mean(torch.abs(diff)).item()
    
    # ================= 打印报告 =================
    print(f"\n{'='*20} 对比结果 {'='*20}")
    print(f"Cosine Similarity (越接近1越像) : {cosine_sim:.6f}")
    print(f"L_inf Distance (最大单点扰动)   : {l_inf_dist:.6f}")
    print(f"L2 Distance (整体欧氏距离)      : {l2_dist:.6f}")
    print(f"Mean Absolute Error (平均扰动)  : {mae:.6f}")

    print(f"\n{'='*20} 验证攻击限制 {'='*20}")
    expected_eps = 8 / 255
    print(f"理论限制 (EPS = 8/255) : {expected_eps:.6f}")
    
    if l_inf_dist > expected_eps + 1e-5:
        print(f"🔴 警告: 实际扰动 ({l_inf_dist:.6f}) 超过了限制 ({expected_eps:.6f})！")
        print("   可能原因：保存时的数据类型转换误差，或者攻击代码中的 clamp 逻辑有误。")
    else:
        print(f"🟢 正常: 实际扰动在限制范围内。")

    # ================= 数据预览 =================
    print(f"\n{'='*20} 前10个数值对比 {'='*20}")
    print(f"{'Index':<5} | {'Clean':<10} | {'Adv':<10} | {'Diff':<10}")
    print("-" * 45)
    for i in range(10):
        v1 = f1_flat[i].item()
        v2 = f2_flat[i].item()
        d = v2 - v1
        print(f"{i:<5} | {v1:<10.4f} | {v2:<10.4f} | {d:<10.4f}")

if __name__ == "__main__":
    compare_features(file_clean, file_adv)
