import os
import sys
import subprocess

def main():
    print("开始运行主程序...")
    result = subprocess.run([sys.executable, "train-experiment7.py"])

    if result.returncode == 0:
        print("\n主程序已成功完成！")
    else:
        print(f"\n主程序以错误代码 {result.returncode} 退出")

    print("\n准备停止AutoDL实例...")
    os.system("shutdown /s /t 300")

if __name__ == "__main__":
    main()