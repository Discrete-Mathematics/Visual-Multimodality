import os
import time
import glob
from datetime import datetime

# ================= 配置区域 =================
# 权重文件所在的文件夹路径 (根据你之前的报错信息，我填好了这个路径，请核对)
TARGET_DIR = "/root/autodl-tmp/FTransUNet/checkpoints_end2end_new"

# 文件匹配模式 (只匹配 epoch 权重，不匹配 best_model)
FILE_PATTERN = "checkpoint_epoch_*.pth"

# 保留最新的几个文件？ (比如保留最新的3个，防止误删正在写的)
KEEP_LAST_N = 3

# 检查间隔 (秒) -> 5分钟 = 300秒
INTERVAL = 1
# ===========================================

def clean_old_checkpoints():
    # 1. 获取所有匹配的权重文件
    search_pattern = os.path.join(TARGET_DIR, FILE_PATTERN)
    files = glob.glob(search_pattern)
    
    # 如果文件数量少于要保留的数量，直接跳过
    if len(files) <= KEEP_LAST_N:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 当前只有 {len(files)} 个文件，无需清理。")
        return

    # 2. 按修改时间排序 (从新到旧)
    # 也就是 files[0] 是最新的，files[-1] 是最旧的
    files.sort(key=os.path.getmtime, reverse=True)

    # 3. 区分要保留的和要删除的
    files_to_keep = files[:KEEP_LAST_N]
    files_to_delete = files[KEEP_LAST_N:]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 发现 {len(files)} 个文件，准备删除 {len(files_to_delete)} 个旧文件...")

    # 4. 执行删除
    for file_path in files_to_delete:
        try:
            # 双重保险：绝对不要删除 best_model.pth (虽然正则没匹配到，但为了安全)
            if "best_model.pth" in file_path:
                continue
                
            os.remove(file_path)
            print(f"  🗑️ 已删除: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ❌ 删除失败 {file_path}: {str(e)}")

def main():
    print(f"🚀 自动清理脚本已启动！")
    print(f"📂 监控目录: {TARGET_DIR}")
    print(f"🛡️ 始终保留最新 {KEEP_LAST_N} 个文件")
    print(f"⏱️ 每 {INTERVAL} 秒检查一次\n")

    while True:
        try:
            clean_old_checkpoints()
        except Exception as e:
            print(f"运行出错: {e}")
        
        # 倒计时等待
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()