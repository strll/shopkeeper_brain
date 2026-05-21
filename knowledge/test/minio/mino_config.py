import os
from pathlib import Path


def format_size(size_in_bytes):
    """将字节数转换为人类可读的格式 (KB, MB, GB)"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0


def check_model_directory(dir_path_str):
    dir_path = Path(dir_path_str)
    print(f"\n{'=' * 60}")
    print(f"🔍 正在检查目录: {dir_path}")
    print(f"{'=' * 60}")

    if not dir_path.exists():
        print("❌ 结论: 路径根本不存在！可能是盘符或拼写错误。")
        return

    if not dir_path.is_dir():
        print("❌ 结论: 路径存在，但它是一个文件而不是文件夹！")
        return

    total_size = 0
    file_count = 0

    print(f"{'文件名'.ljust(40)} | {'文件大小'}")
    print("-" * 60)

    # 递归遍历目录下的所有文件
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            total_size += file_size
            file_count += 1

            # 格式化打印排版
            display_name = file_path.name
            if len(display_name) > 38:
                display_name = display_name[:35] + "..."
            print(f"{display_name.ljust(40)} | {format_size(file_size)}")

    print("-" * 60)
    print(f"📊 统计汇总:")
    print(f"   文件总数: {file_count} 个")
    print(f"   占用空间: {format_size(total_size)}")

    # 智能诊断结论
    if total_size == 0:
        print("\n⚠️ 诊断结论: 这是一个空文件夹！MinerU 找不到模型，所以触发了重新下载。")
    elif total_size < 500 * 1024 * 1024:  # 小于 500MB
        print("\n⚠️ 诊断结论: 文件夹总大小异常小！完整的模型通常需要几个 GB。")
        print("   这说明你可能只下载了 README 或 config 等小文件，核心的权重大文件（如 .safetensors 或 .bin）缺失。")
    else:
        print("\n✅ 诊断结论: 文件夹体积看起来正常（达到了 GB 级别）。")
        print(
            "   如果代码仍然提示下载，请对照日志看看是不是具体缺失了某个特定名称的文件，或者文件哈希值校验不通过（说明下载过程中损坏了）。")


if __name__ == "__main__":
    # 这是从你的 mineru.json 中提取的两个目标路径
    path_pipeline = r"E:\mineru_models\models\OpenDataLab\PDF-Extract-Kit-1___0"
    path_vlm = r"E:\mineru_models\models\OpenDataLab\MinerU2___5-Pro-2604-1___2B"

    check_model_directory(path_pipeline)
    check_model_directory(path_vlm)