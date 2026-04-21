## 项目上下文摘要（标准答案误报抑制）

生成时间：2026-04-20 16:16:13 CST

### 1. 相似实现分析

- **实现1**: `detector.py::UniversalTamperDetector.detect`
  - 模式：读取图片、调用文档检测器、融合 ELA/噪声证据、输出最终检测和报告。
  - 可复用：`candidate_regions` 已保存完整候选池，适合在最终输出前增加报告级过滤。
  - 需注意：不能删除内部候选，否则后续调试和评测调参会缺少信息。
- **实现2**: `detector.py::_build_final_detections`
  - 模式：当前先保留文档层主框，再按分数补满到 `max_detections`。
  - 可复用：已有 `_bbox_iou`、`_bbox_overlap_ratio` 和去重逻辑。
  - 需注意：误报主要来自“补满”行为，需要把 `max_detections` 改成上限而不是目标数量。
- **实现3**: `detector.py::build_report`
  - 模式：当前 `top_candidates` 直接按候选原始分数截取。
  - 可复用：报告结构可保留 `candidate_count`，新增过滤后数量字段。
  - 需注意：用户选择“报告也过滤”，所以 `top_candidates` 也不能继续展示明显正确文字误报。
- **实现4**: `evaluation.py::evaluate_dataset`
  - 模式：用最终 `detections` 与标准答案框计算召回和误报。
  - 可复用：无需修改评测计算，只要最终检测框更干净即可反映误报下降。
  - 需注意：验收硬门槛为总召回率不低于 `0.875`、总误报不超过 `10`。

### 2. 项目约定

- **命名约定**: Python 函数使用 `snake_case`，内部过滤函数使用 `_` 前缀。
- **文件组织**: 检测输出过滤放在融合层 `detector.py`；评测指标仍由 `evaluation.py` 负责。
- **导入顺序**: 标准库、第三方库、项目内部模块分组导入，不新增依赖。
- **代码风格**: 中文注释解释策略意图；避免把调参信息写成变更记录式注释。

### 3. 可复用组件清单

- `detector.py::TamperRegion`: 最终输出和候选池的统一数据结构。
- `detector.py::EvidenceBundle`: 候选 `detail.evidence` 中已有局部 ELA/噪声分。
- `detector.py::_bbox_iou`: 最终去重使用的 IoU 计算。
- `evaluation.py::bbox_iou`: 测试中验证候选与标准框命中关系。
- `test_detector.py::sample_image_path`: 测试样例路径解析。

### 4. 测试策略

- **测试框架**: Python 标准库 `unittest`。
- **测试模式**: 真实样例回归、标准答案评测、CLI 子进程、报告结构断言。
- **参考文件**: `test_detector.py`。
- **覆盖要求**: 总召回率 `>= 0.875`，总误报 `<= 10`，并新增每类误报抑制断言。

### 5. 依赖和集成点

- **外部依赖**: OpenCV、NumPy、Pillow，保持不新增。
- **内部依赖**: `build_report` 需要复用和 `detect` 相同的报告级过滤逻辑。
- **集成方式**: 不改变 CLI 参数和 Python API 参数，改变的是最终输出筛选策略。
- **配置来源**: 继续使用现有 `max_detections` 作为最多输出数量上限。

### 6. 技术选型理由

- **为什么用这个方案**: 当前误报来自最终输出阶段的候选补满，最小风险修复是在融合层增加报告级过滤，而不是重写文档层规则。
- **优势**: 保留完整候选池，减少用户可见误报，同时不破坏标准答案召回基准。
- **劣势和风险**: 规则仍依赖当前样例分布，后续新增样例时需要继续用 `evaluation.py` 评估阈值。

### 7. 关键风险点

- **边界条件**: 空白图可能没有可报告候选，应返回合法状态；缺图继续抛出清晰异常。
- **性能瓶颈**: 过滤只遍历候选列表，不增加图像级重计算。
- **回归风险**: 身份证姓名旧断言与标准答案不一致，本轮以标准答案评测和“正确文字不误判”为准调整断言。
- **工具限制**: 当前会话仍未暴露 `sequential-thinking`、`shrimp-task-manager`、`desktop-commander`、`context7` 和 `github.search_code`，继续用本地只读分析、自动测试和评测报告替代并留痕。
