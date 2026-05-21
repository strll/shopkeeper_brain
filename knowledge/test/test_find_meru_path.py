import json
from pathlib import Path


def get_mineru_models_dir():
    # 1. 获取当前系统用户的家目录 (等同于 C:\Users\你的用户名)
    home_dir = Path.home()

    # 2. MinerU 的配置文件通常叫这两个名字之一
    config_files = ["mineru.json", "magic-pdf.json"]

    for config_name in config_files:
        config_path = home_dir / config_name

        # 判断文件是否存在
        if config_path.exists():
            print(f"✅ 找到配置文件: {config_path}")
            try:
                # 3. 读取并解析 JSON 文件
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                    # 4. 提取 models-dir 字段
                    models_dir = config_data.get("models-dir")

                    if models_dir:
                        print(f"\n🎉 提取成功！你的 '--source local' 对应的模型真实路径是:")
                        print(f"👉 {models_dir}")
                        return models_dir
                    else:
                        print("❌ 配置文件中存在，但没有找到 'models-dir' 字段。")
                        return None

            except json.JSONDecodeError:
                print(f"❌ 解析 JSON 失败，请检查 {config_path} 里的格式是否报错。")
                return None
            except Exception as e:
                print(f"❌ 读取文件时发生未知错误: {e}")
                return None

    print(f"❌ 检索完毕。未能在 {home_dir} 目录下找到 MinerU 的配置文件。")
    return None


if __name__ == "__main__":
    print("开始检测 MinerU 本地模型路径...\n" + "-" * 40)
    get_mineru_models_dir()
    print("-" * 40)