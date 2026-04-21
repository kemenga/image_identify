## 项目上下文摘要（大图扫描性能）

生成时间：2026-04-21 17:05:00 CST

### 1. 相似实现分析

- **实现1**: `main.py:16`
  - 模式：默认运行配置集中在 `RUN_CONFIG`
  - 可复用：无参数运行时直接读取 `RUN_CONFIG["image"]`
  - 需注意：用户会直接修改这里指向 `seed_dataset` 的高分辨率样本
- **实现2**: `document_detector.py:583`
  - 模式：`_enumerate_region_anomalies(...)` 做全图多尺度滑窗
  - 可复用：候选生成、鲁棒分数和上下文分数逻辑保持不变
  - 需注意：大图上 24 像素窗口数量会达到十几万，单窗口又会调用 `Canny/Laplacian/CLAHE`
- **实现3**: `document_detector.py:662`
  - 模式：`_enumerate_text_window_anomalies(...)` 同样用 24 像素全图滑窗生成文本热点
  - 可复用：后续贴合文字组件、证件区域候选和报告过滤逻辑
  - 需注意：这里和区域异常扫描有相同性能瓶颈
- **实现4**: `test_detector.py`
  - 模式：使用 `unittest` 和真实样例做回归
  - 可复用：可新增私有辅助函数测试，确保大图扫描会降采样并受窗口预算约束

### 2. 项目约定

- **命名约定**: Python 使用 `snake_case`，可调参数沿用全大写类属性
- **文件组织**: 性能修复留在 `document_detector.py`，入口层不新增公开参数
- **导入顺序**: 标准库、第三方、本地模块
- **代码风格**: 中文注释解释约束，保持既有候选结构不变

### 3. 可复用组件清单

- `document_detector.py::_region_feature_vector`
- `document_detector.py::_region_context_score`
- `document_detector.py::_robust_scores`
- `document_detector.py::_snap_text_cluster_to_components`
- `main.py::RUN_CONFIG`

### 4. 测试策略

- **测试框架**: `unittest`
- **测试模式**: 新增大图扫描预算单测 + 真实大图冒烟 + 全量回归
- **参考文件**: `test_detector.py`
- **覆盖要求**: 大图会自动降采样；扫描窗口数不超过预算；原有 26 项回归不退化

### 5. 依赖和集成点

- **外部依赖**: `opencv-python`、`numpy`
- **内部依赖**: `UniversalTamperDetector.detect(...)` 调用 `TraditionalTamperDetector.detect(...)`
- **集成方式**: 不改公开 API，新增文档规则层可调参数自动进入 `main.py` 的 `--document-*` 参数
- **配置来源**: `TraditionalTamperDetector.tunable_defaults()`

### 6. 技术选型理由

- **为什么用这个方案**: 大图卡顿根因是全图密集滑窗数量过大，最小改动是对滑窗输入做自动缩放并限制窗口预算
- **优势**: 不改变候选类型和报告结构，旧样例结果风险较小
- **劣势和风险**: 对超大图中极小像素级异常的敏感度会下降，但当前需求优先保证工具可用和可交互

### 7. 关键风险点

- **边界条件**: 用户把扫描上限调得过小会降低召回
- **性能瓶颈**: `Canny/Laplacian/CLAHE` 仍是每个窗口的主要成本，但窗口数量已受控
- **评测影响**: 需要跑现有回归，防止截图、票据、身份证、新样例命中退化
