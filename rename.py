import os
import glob

# 你的特征文件所在路径
TARGET_DIR = "/data0/data/msrvtt/msvd_malmm_to_frozen"

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"错误：找不到目录 {TARGET_DIR}")
        return

    # 获取所有 .pt 文件
    files = glob.glob(os.path.join(TARGET_DIR, "*.pt"))
    print(f"找到 {len(files)} 个 .pt 文件，准备处理...")

    count = 0
    for fpath in files:
        dirname, basename = os.path.split(fpath)
        
        # 跳过已经包含 _adv 的文件
        if "_adv.pt" in basename:
            continue
            
        # 构造新文件名：video7010.pt -> video7010_adv.pt
        new_basename = basename.replace(".pt", "_adv.pt")
        new_fpath = os.path.join(dirname, new_basename)
        
        # 重命名
        os.rename(fpath, new_fpath)
        count += 1
        
        if count % 500 == 0:
            print(f"已重命名 {count} 个文件...")

    print(f"完成！共重命名了 {count} 个文件。")
    print(f"现在它们是这样的：{new_basename}")

if __name__ == "__main__":
    main()
