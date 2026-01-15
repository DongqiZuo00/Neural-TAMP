# Neural-TAMP 白皮书（基于 `main.py` 的全流程梳理）

## 1. 系统总览（从主入口理解整体框架）

`main.py` 定义了一个**批量数据生成管线**，核心目标是：

- 在 ProcTHOR 场景中批量采样场景
- 通过 Oracle 构建语义图（SceneGraph）
- 生成可执行任务指令
- 通过分层规划器将指令分解为可执行动作序列
- 保存数据、可视化结果与日志

整个流程围绕“场景 → 感知 → 记忆 → 任务 → 规划 → 数据输出”的闭环构建。

参考：`main.py`（初始化、循环、数据保存与统计流程）【F:main.py†L1-L158】

---

## 2. `main.py` 主流程分解（从入口到输出的清晰脉络）

### 2.1 入口与环境准备

- 修正 Python 搜索路径，保证本地模块可导入。
- 检查 `OPENAI_API_KEY`，提示规划器可能失败。
- 初始化输出目录，清空旧数据。

参考：`main.py` 路径修正与输出目录准备【F:main.py†L8-L39】

### 2.2 核心模块初始化

初始化系统所需的关键组件：

- `ProcTHOREnv`：仿真环境控制器
- `OracleInterface`：从环境直接读取场景真值图
- `GraphManager`：记忆模块（图结构重构与严格关系推断）
- `BEVVisualizer`：鸟瞰图可视化
- `TaskGenerator`：任务生成器
- `TaskDecomposer`：分层规划器

参考：`main.py` 模块初始化【F:main.py†L41-L53】

### 2.3 批量采样配置

- 目标样本数 `TOTAL_SAMPLES=50`
- 从 10000 个场景中随机抽取 200 个候选

参考：`main.py` 参数配置【F:main.py†L58-L65】

### 2.4 主循环：逐场景处理流程

每个候选场景的处理流程包括：

**A. 场景加载**

调用 `env.change_scene(idx)` 切换场景。

参考：`main.py` 场景加载【F:main.py†L78-L83】

**B. 场景筛选**

只保留包含至少两个房间的场景，提高任务复杂度。

参考：`main.py` 场景筛选【F:main.py†L85-L90】

**C. 感知与记忆构建**

1. `OracleInterface` 输出带 Room 结构的层级图
2. `GraphManager` 用严格规则重建同房间几何关系

参考：`main.py` 感知/记忆构建【F:main.py†L92-L98】

**D. 任务生成**

`TaskGenerator` 根据图生成 Pick & Place 指令及元数据。

参考：`main.py` 任务生成【F:main.py†L99-L107】

**E. 分层规划**

`TaskDecomposer` 将自然语言任务拆解为子目标，再生成原子动作序列。

参考：`main.py` 分层规划【F:main.py†L109-L119】

**F. 数据保存与可视化**

保存 Ground Truth BEV 与 AI 语义地图，并将任务、计划、可视化信息写入 JSON。

参考：`main.py` 数据保存【F:main.py†L121-L149】

### 2.5 结束与统计

停止环境，输出统计与数据路径。

参考：`main.py` 结束与统计【F:main.py†L151-L158】

---

## 3. 数据与控制流图（核心对象的流转路径）

1. **环境**：`ProcTHOREnv` → 场景状态
2. **感知**：`OracleInterface` → `SceneGraph`
3. **记忆**：`GraphManager` → 关系重构的 `SceneGraph`
4. **任务**：`TaskGenerator` → instruction + metadata
5. **规划**：`TaskDecomposer` → action list
6. **输出**：`BEVVisualizer` + JSON 日志

参考：`main.py` 主流程【F:main.py†L41-L149】

---

## 4. 全部类的逐一讲解（Class-by-Class）

以下为仓库中全部 Python 类，逐个说明职责、关键方法、输入输出与协作对象。

### 4.1 `ProcTHOREnv`（环境封装）

**文件**：`src/env/procthor_wrapper.py`

**职责**：

- 封装 AI2-THOR/ProcTHOR 控制器
- 负责加载场景、重置、采集观察、保存 GT 视角

**关键方法**：

- `reset()`：重置场景并随机选择可达点传送机器人
- `change_scene(index)`：切换场景并调用 `reset()`
- `get_observation()`：返回当前 RGB / depth / pose
- `save_ground_truth_bev()`：添加第三人称顶视相机并保存图像

参考：`ProcTHOREnv` 定义与方法实现【F:src/env/procthor_wrapper.py†L7-L117】

与主流程关系：`main.py` 初始化与调用环境完成场景切换与 GT 输出。【F:main.py†L44-L47】【F:main.py†L78-L127】

---

### 4.2 `OracleInterface`（真值场景图提取器）

**文件**：`src/perception/oracle_interface.py`

**职责**：

- 从 THOR 元数据中提取**带房间层级**的 `SceneGraph`
- 将房间建为节点，物体加入房间归属，建立 `contains` 边

**关键方法**：

- `_calculate_polygon_area()`：计算房间多边形面积
- `get_hierarchical_graph()`：构建房间-物体层级图

参考：`OracleInterface` 主要逻辑【F:src/perception/oracle_interface.py†L5-L88】

与主流程关系：负责产生高质量 `SceneGraph`，供记忆和任务生成使用。【F:main.py†L93-L98】

---

### 4.3 `GraphManager`（记忆/图结构管理）

**文件**：`src/memory/graph_manager.py`

**职责**：

- 存储全局图 `global_graph`
- 根据房间与几何关系重新计算 `on/inside` 边
- 保留房间 `contains` 关系，剔除跨房间边

**关键方法**：

- `override_global_graph()`：覆盖图并触发严格重构
- `_recompute_edges_strict()`：同房间/几何关系筛选
- `get_rag_context()`：语义检索相关节点
- `save_snapshot()`：保存记忆快照

参考：`GraphManager` 逻辑【F:src/memory/graph_manager.py†L6-L88】

与主流程关系：承接 Oracle 图并进行结构强化。【F:main.py†L96-L98】

---

### 4.4 `SceneGraph` / `Node` / `Edge`（图数据结构）

**文件**：`src/core/graph_schema.py`

#### `Node`

- 描述节点实体，包含 `id/label/pos/bbox/state/geometry/room_id`
- `room_id` 标识房间归属

参考：`Node` 定义【F:src/core/graph_schema.py†L4-L26】

#### `Edge`

- 描述关系边（source、target、relation）

参考：`Edge` 定义【F:src/core/graph_schema.py†L28-L41】

#### `SceneGraph`

- 存储 `nodes/edges/robot_pose`
- 提供序列化与文本格式输出接口

参考：`SceneGraph` 定义【F:src/core/graph_schema.py†L43-L70】

---

### 4.5 `TaskGenerator`（任务生成器）

**文件**：`src/task/task_generator.py`

**职责**：

- 基于场景图生成复杂的 Pick & Place 指令
- 过滤不可行任务
- 根据“距离、靠墙、拥挤度”进行对抗性排序

**关键方法**：

- `generate()`：主入口，返回 `(instruction, metadata)`
- `_scan_objects()`：从图中筛选可拾取物体与容器
- `_check_hard_constraints()`：硬约束检查
- `_calculate_adversarial_reward()`：计算难度评分
- `_generate_long_instruction()`：生成 >60 字符指令

参考：`TaskGenerator` 逻辑与约束【F:src/task/task_generator.py†L6-L224】

与主流程关系：为每个场景生成任务指令和结构化元数据。【F:main.py†L99-L107】

---

### 4.6 `TaskDecomposer`（分层规划器）

**文件**：`src/planning/decomposer.py`

**职责**：

- 两阶段规划：
  1. 任务 → 子目标
  2. 子目标 → 原子动作

**关键方法**：

- `plan()`：调用 LLM，两阶段生成 actions

参考：`TaskDecomposer` 实现【F:src/planning/decomposer.py†L6-L56】

与主流程关系：将任务指令变为动作序列，生成 dataset 的“ground truth plan”。【F:main.py†L109-L119】

---

### 4.7 `PromptBuilder`（规划 Prompt 生成器）

**文件**：`src/planning/prompt_builder.py`

**职责**：

- 构造 LLM prompt
- 提供两类 prompt：任务分解、动作生成

**关键方法**：

- `build_decomposition_prompt()`
- `build_action_prompt()`
- `_graph_to_text()`：将场景图转文字描述

参考：`PromptBuilder` 逻辑【F:src/planning/prompt_builder.py†L4-L114】

---

### 4.8 `LLMInterface`（LLM 调用封装）

**文件**：`src/planning/llm_interface.py`

**职责**：

- 封装 OpenAI API 调用
- 统一 JSON 格式输出

**关键方法**：

- `predict(system_prompt, user_prompt)`：发送请求并解析 JSON

参考：`LLMInterface` 实现【F:src/planning/llm_interface.py†L6-L44】

---

### 4.9 `BEVVisualizer`（鸟瞰图可视化）

**文件**：`src/utils/visualizer.py`

**职责**：

- 根据 `SceneGraph` 绘制房间多边形、对象点位、边关系和机器人姿态
- 生成语义鸟瞰图（用于数据对比/可视化）

**关键方法**：

- `render(scene_graph, filename)`

参考：`BEVVisualizer` 绘图逻辑【F:src/utils/visualizer.py†L7-L100】

与主流程关系：在样本生成时保存 AI 语义地图。【F:main.py†L121-L128】

---

### 4.10 `SemanticMatcher`（语义概念匹配器）

**文件**：`src/utils/semantic_matcher.py`

**职责**：

- 用 SentenceTransformer 计算标签语义相似度
- 判断一个物体是否为“锚点物体”或“容器”

**关键方法**：

- `is_anchor()`
- `is_container()`
- `_check_similarity()`

参考：`SemanticMatcher` 定义【F:src/utils/semantic_matcher.py†L14-L74】

与主流程关系：用于 `GraphManager` 推断 “on/inside”。【F:src/memory/graph_manager.py†L60-L63】

---

### 4.11 `VLMInterface`（视觉-语言模型感知模块）

**文件**：`src/perception/vlm_interface.py`

**职责**：

- 利用 Qwen2-VL 解析图像并生成 SceneGraph
- 强化 JSON 输出结构化要求

**关键方法**：

- `_build_system_prompt()`：构造严格 JSON schema
- `parse()`：图像 → 文本 → SceneGraph
- `_text_to_graph()`：解析模型输出

参考：`VLMInterface` 实现【F:src/perception/vlm_interface.py†L22-L179】

> 注意：该模块当前不在 `main.py` 的管线中使用，但提供了视觉感知替代路径。

---

### 4.12 `SpatialLifter`（2D → 3D 空间投影）

**文件**：`src/perception/spatial_lifter.py`

**职责**：

- 将 2D bbox 中心点通过深度图与相机内参投影到 3D
- 更新 SceneGraph 中节点位置

**关键方法**：

- `lift_to_3d(scene_graph, depth_map, robot_pose)`

参考：`SpatialLifter` 实现【F:src/perception/spatial_lifter.py†L14-L91】

---

## 5. 总结（系统能力与定位）

**Neural-TAMP 的核心定位**是：

- 使用“真值感知 + 结构化记忆 + 对抗任务生成 + 分层规划”
- 构建可以训练/验证神经规划器的数据集
- 兼具“可执行动作序列”和“可视化语义地图”输出

此流程以 `main.py` 串联全部核心模块，并通过 `SceneGraph` 贯穿感知、任务生成与规划。

参考：`main.py` 与 `SceneGraph` 结构【F:main.py†L41-L149】【F:src/core/graph_schema.py†L43-L70】
