import json
import os

# 定义你的存档文件路径 (使用 r 前缀避免 Windows 路径中的转义字符冲突)
file_paths = [
    r"C:\Users\12903\AppData\Roaming\SlayTheSpire2\steam\76561198980205808\profile2\saves\current_run.save",
    r"C:\Users\12903\AppData\Roaming\SlayTheSpire2\steam\76561198980205808\profile2\saves\current_run.save.backup"
]

# === 在这里设置你想要修改的数值 ===
NEW_CURRENT_HP = 130  # 当前生命值
NEW_MAX_HP = 158  # 最大生命值
NEW_GOLD = 6000  # 金币数量


# =================================

def modify_save_file(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}")
        return

    try:
        # 1. 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 查找并修改 players 字段下的数值
        if "players" in data:
            for player in data["players"]:
                # 更新数值
                player["current_hp"] = NEW_CURRENT_HP
                player["max_hp"] = NEW_MAX_HP
                player["gold"] = NEW_GOLD

            print(f"✅ 成功修改文件: {file_path}")
        else:
            print(f"⚠️ 文件中未找到 'players' 字段: {file_path}")
            return

        # 3. 将修改后的数据写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            # indent=2 保持 JSON 格式美观，ensure_ascii=False 避免中文被转义
            json.dump(data, f, indent=2, ensure_ascii=False)

    except json.JSONDecodeError:
        print(f"❌ 解析 JSON 失败，文件可能已损坏: {file_path}")
    except Exception as e:
        print(f"❌ 处理文件 {file_path} 时发生未知错误: {e}")


# 遍历并处理两个文件
for path in file_paths:
    modify_save_file(path)

print("执行完毕！")