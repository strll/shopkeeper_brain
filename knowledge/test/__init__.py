import subprocess
import threading
import time

# ================= 配置区 =================
# 替换为你刚才测试用的真实 PDF 路径
TEST_PDF_PATH = r"/knowledge/processor/import_process\temp_dir\万用表的使用.pdf"
OUTPUT_DIR = r"./test_output"


# ==========================================

def check_pytorch_env():
    """第一关：检查底层 PyTorch 是否支持 CUDA"""
    try:
        import torch
        is_available = torch.cuda.is_available()
        print(f"[-] PyTorch 基础环境检查 -> CUDA 可用状态: {is_available}")
        if not is_available:
            print("[!] 警告: 你的 PyTorch 是 CPU 版本或未正确安装 CUDA 驱动。MinerU 绝对无法走 GPU。")
        else:
            print(f"[-] 检测到显卡: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[!] 未安装 PyTorch，无法进行底层环境检查。")


def monitor_gpu_memory(stop_event, memory_records):
    """后台线程：每 0.5 秒记录一次 GPU 显存占用"""
    while not stop_event.is_set():
        try:
            # 调用 nvidia-smi 仅获取显存占用数值 (单位 MB)
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8',
                creationflags=subprocess.CREATE_NO_WINDOW  # Windows 下防止弹出黑框
            )
            # 假设只有单卡，取第一行数据
            mem_used = int(result.strip().split('\n')[0])
            memory_records.append(mem_used)
        except Exception as e:
            # 如果没有安装 nvidia-smi 或者报错，静默忽略
            pass
        time.sleep(0.5)


def run_test():
    check_pytorch_env()
    print("-" * 50)

    # 启动 GPU 监控线程
    stop_event = threading.Event()
    memory_records = []
    monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(stop_event, memory_records))
    monitor_thread.start()

    print(f"[*] 开始运行 mineru 解析任务...\n[*] 正在监控 GPU 显存，请稍候...")
    start_time = time.time()

    try:
        # 执行 mineru 命令
        cmd = ["mineru", "-p", TEST_PDF_PATH, "-o", OUTPUT_DIR, "--source", "local"]
        # 我们这里不捕获输出，直接让它在后台默默跑完，只关心 GPU 状态
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"[*] 解析完成！耗时: {time.time() - start_time:.2f} 秒")
    except subprocess.CalledProcessError:
        print("[!] mineru 命令执行失败，请检查 PDF 路径是否正确。")
    finally:
        # 停止监控
        stop_event.set()
        monitor_thread.join()

    # 分析监控结果
    print("-" * 50)
    if not memory_records:
        print("[!] 无法获取 nvidia-smi 数据，请确认你的系统是 NVIDIA 显卡且驱动正常。")
        return

    baseline_mem = memory_records[0]
    peak_mem = max(memory_records)
    mem_increase = peak_mem - baseline_mem

    print(f"[数据] 初始显存占用: {baseline_mem} MB")
    print(f"[数据] 峰值显存占用: {peak_mem} MB")
    print(f"[数据] 解析期间显存净增加: {mem_increase} MB")

    print("\n>>> 最终结论 <<<")
    # 通常加载 Layout 或 OCR 模型至少需要几百 MB 到几个 G 的显存
    if mem_increase > 300:
        print("✅ 恭喜！检测到显存显著飙升，MinerU 成功调用了 GPU！")
    else:
        print("❌ 未检测到显存显著增加 (增量小于 300MB)。MinerU 当前正在使用纯 CPU 进行解析。")


if __name__ == "__main__":
    run_test()