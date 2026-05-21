class MiniLangGraphEngine:
    def __init__(self):
        self.nodes = {}  # 存放节点的盒子：{节点名: 节点函数}
        self.edges = {}  # 存放路线图的盒子：{当前节点: 下一个节点}
        self.state = {}  # 【核心】全局状态字典

    # 1. 注册节点
    def add_node(self, name, func):
        self.nodes[name] = func

    # 2. 注册普通边 (固定路线)
    def add_edge(self, from_node, to_node):
        self.edges[from_node] = to_node

    # 3. 引擎启动（这就是图计算的底层逻辑！）
    def run(self, start_node, initial_state):
        print("🚀 图引擎启动！")
        self.state = initial_state
        current_node = start_node
        superstep_count = 1

        # 【魔法就在这里】
        # 这就是一个典型的 Event Loop（事件循环），也就是 Pregel 模型里的“回合制”
        while current_node != "END":
            print(f"\n--- 🔄 第 {superstep_count} 回合 (Superstep) 开始 ---")
            print(f"📍 当前停靠节点: [{current_node}]")

            # 第一步：获取当前节点的任务函数，并把【当前状态】传给它
            node_func = self.nodes[current_node]
            # 节点执行后，必须返回一个“状态增量（Update）”
            state_update = node_func(self.state)

            # 第二步：【状态归约/合并】
            # 引擎负责把节点返回的新数据，合并到全局 State 中
            if state_update:
                self.state.update(state_update)
            print(f"📦 节点执行完毕，当前全局 State 变为: {self.state}")

            # 第三步：【全局同步屏障与路由跳转】
            # 在这里，节点必须停下来。由引擎查表决定下一步去哪
            # 完全没有 if...else 控制流，全靠查配置表！
            next_node = self.edges.get(current_node, "END")
            print(f"➡️ 根据配置的路线图，下一步前往: [{next_node}]")

            current_node = next_node
            superstep_count += 1

            # 💡 这里就是“持久化/检查点”的最佳位置！
            # 如果你想暂停程序，只需要在这里把 self.state 保存到数据库即可。

        print("\n🛑 遇到 END 节点，引擎执行结束！")
        return self.state