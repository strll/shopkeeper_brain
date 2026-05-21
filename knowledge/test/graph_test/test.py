# ==========================================
# 1. 定义节点函数 (每个节点只关心自己那一摊事)
# ==========================================
from knowledge.test.graph_test.MiniLangGraphEngine import MiniLangGraphEngine


def wash_vegetables(state):
    print("   👩‍🍳 动作：正在洗菜...")
    # 只需要返回状态的【增量更新】，不要直接修改传入的 state
    return {"vegetables_clean": True}

def cut_meat(state):
    print("   👩‍🍳 动作：正在切肉...")
    return {"meat_cut": True}

def cook_food(state):
    print("   👩‍🍳 动作：正在炒菜...")
    # 模拟根据前面的状态做出反应
    if state.get("vegetables_clean") and state.get("meat_cut"):
        return {"meal_ready": True, "dish_name": "辣椒炒肉"}
    else:
        return {"meal_ready": False, "error": "食材没准备好！"}

# ==========================================
# 2. 拼装图计算工作流
# ==========================================
workflow = MiniLangGraphEngine()

# 添加节点
workflow.add_node("node_wash", wash_vegetables)
workflow.add_node("node_cut", cut_meat)
workflow.add_node("node_cook", cook_food)

# 定义边 (画路线图)
workflow.add_edge("node_wash", "node_cut")   # 洗完菜去切肉
workflow.add_edge("node_cut", "node_cook")   # 切完肉去炒菜
workflow.add_edge("node_cook", "END")        # 炒完菜结束

# ==========================================
# 3. 运行！
# ==========================================
initial_state = {"vegetables_clean": False, "meat_cut": False}
final_state = workflow.run(start_node="node_wash", initial_state=initial_state)

print("\n最终产出状态：", final_state)