## 项目上下文摘要（新增样本漏检修复）

生成时间：2026-04-21 15:10:29 CST

### 1. 相似实现分析

- **实现1**: `test_detector.py::test_dataset_evaluation_recall_threshold`
  - 模式：对 `evaluate_dataset` 的结果做全量门槛校验。
  - 可复用：继续使用 `evaluate_dataset` 和 `matches` 结构统计命中与误报。
  - 需注意：新增 `image.png`、`image4.png`、`image5.png` 后，全量召回已降到 `0.6667`，不能再把“稳定样例通过门槛”和“漏检样例基线”混在一起。
- **实现2**: `test_detector.py::test_reportable_outputs_suppress_correct_text_false_positives`
  - 模式：按图片名拆分报告，对关键样例做显式断言。
  - 可复用：适合把新样本的当前召回、误报和候选类型写成硬回归。
  - 需注意：该测试原本只覆盖旧样例，需要新增三张样本的独立断言。
- **实现3**: `test_detector.py::test_high_report_threshold_keeps_ground_truth_candidates`
  - 模式：在真实样例上叠加 `REPORT_CONFIDENCE_THRESHOLD`，复用 `bbox_iou` 和标准答案框做质量回归。
  - 可复用：说明项目已经接受“真实样例 + 标准答案图 + IoU 断言”的回归方式。
  - 需注意：本轮不修改 `detector.py`，只能围绕现状补齐基线测试和修复计划说明。
- **实现4**: `evaluation.py::evaluate_dataset`
  - 模式：统一生成 `ground_truth_boxes`、`detections`、`matches`、`recall` 和 `false_positive_count`。
  - 可复用：新增样本的显式评测断言应直接绑定这个输出协议，避免重复实现评测逻辑。
  - 需注意：CLI 与 Python API 都复用同一份统计结构，测试和文档要同步更新。

### 2. 项目约定

- **命名约定**: Python 测试方法继续使用 `test_...`，图片样例名沿用 `image.png`、`image4.png`、`image5.png`。
- **文件组织**: 回归测试集中在 `test_detector.py`；用户说明写入 `README.md`；架构梳理写入 `项目梳理.md`；上下文与验收记录写入项目本地 `.codex/`。
- **导入顺序**: 标准库、第三方、本地模块分组导入；本轮无需新增第三方依赖。
- **代码风格**: 继续使用 `unittest`、`self.assert...` 和 `self.subTest`，文档全部使用简体中文。

### 3. 可复用组件清单

- `test_detector.py::sample_image_path`：统一解析样例图片路径。
- `test_detector.py::_ground_truth_boxes`：复用标准答案框抽取逻辑。
- `evaluation.py::evaluate_dataset`：输出数据集评测报告。
- `evaluation.py::extract_ground_truth_boxes`：从 `xxx检测结果.png` 抽取标准框。
- `evaluation.py::bbox_iou`：统一候选框与标准框的重叠计算。
- `detector.py::UniversalTamperDetector.detect`：返回 `detections` 和 `candidate_regions`，供“最终结果断言”和“候选池覆盖断言”同时使用。

### 4. 新样本目标

- `image.png`
  - 目标：把多字段发票/清单型样例纳入显式评测，记录当前“5/7 命中”的基线，并锁定候选池里 `text_region`、`text_noise_anomaly`、`region_anomaly` 的协同覆盖。
- `image4.png`
  - 目标：记录当前最终输出未命中标准框的现状，同时确认候选池已经具备 `digit_window`、`text_noise_anomaly`、`text_region` 三类入口，避免后续修复前候选来源退化。
- `image5.png`
  - 目标：把“最终输出碎片化、候选池只有 `region_anomaly` 粗覆盖”的现状显式写成回归，为后续从大框收敛到小框提供起点。

### 5. 当前基线观察

- `image.png`
  - 评测基线：`recall=0.7143`、`false_positive_count=0`
  - 漏检位置：`ground_truth_index` 为 `3`、`5` 的两个标准框仍未进入最终输出
  - 候选池观察：7 个标准框都能在候选池中找到重叠候选，其中 5 个已达到 `IoU >= 0.3`
- `image4.png`
  - 评测基线：`recall=0.0`、`false_positive_count=2`
  - 最终输出：`digit_window` + `text_noise_anomaly`
  - 候选池观察：3 个标准框都已有候选重叠，最佳 IoU 分别约为 `0.1118`、`0.1535`、`0.3906`
- `image5.png`
  - 评测基线：`recall=0.0`、`false_positive_count=2`
  - 最终输出：两个 `text_noise_anomaly`
  - 候选池观察：3 个标准框都只被同一个大 `region_anomaly` 弱覆盖，最佳 IoU 约为 `0.0117`、`0.0569`、`0.0204`

### 6. 测试策略

- **稳定样例门槛**: 继续对 `300.png`、`400.png`、截图、票据、发票、身份证做召回和误报门槛校验，维持旧回归的稳定性。
- **新增样例显式评测**: 单独断言 `image.png`、`image4.png`、`image5.png` 的当前召回、误报和最终输出类型，作为漏检修复基线。
- **候选池覆盖**: 使用 `candidate_regions + bbox_iou` 对三张新样例逐个标准框做候选覆盖断言，锁定“是否已经有可用候选”的起点。
- **CLI 对齐**: 继续验证 `evaluation.py` CLI 写出的报告中包含三张新样例，并沿用稳定样例门槛统计。

### 7. 依赖和集成点

- **外部依赖**: OpenCV、NumPy、Pillow；本轮不新增依赖。
- **内部依赖**: `test_detector.py` 依赖 `detector.py`、`evaluation.py` 和 `data/` 下标准答案图。
- **集成方式**: Python API 用于候选池覆盖断言，CLI 用于评测报告写出回归。
- **配置来源**: 本轮不改检测参数，全部沿用当前默认值。

### 8. 验证命令

1. `python -m unittest test_detector.UniversalTamperDetectorTest.test_dataset_evaluation_recall_threshold test_detector.UniversalTamperDetectorTest.test_new_samples_are_explicitly_tracked_in_dataset_evaluation test_detector.UniversalTamperDetectorTest.test_new_samples_candidate_pool_cover_repair_targets test_detector.UniversalTamperDetectorTest.test_evaluation_cli_writes_dataset_report`
2. `python -m unittest test_detector.py`
3. `python evaluation.py --data-dir ./data --report /tmp/evaluation_report_new_samples.json --iou-threshold 0.3 --max-detections 8`

### 9. 结果占位说明

- 后续修复 `detector.py` 时，需在验证报告中按“当前基线 -> 修复后结果”的格式回填三张样本的变化。
- 推荐占位模板：
  - `image.png`：`recall [当前 0.7143] -> [待回填]`
  - `image4.png`：`recall [当前 0.0] -> [待回填]`
  - `image5.png`：`recall [当前 0.0] -> [待回填]`
- 如果候选池策略发生变化，还要同步回填“最佳候选类型”和“最佳 IoU”是否提升。

### 10. 工具偏差说明

- 仓库规范提到的 `desktop-commander`、`context7`、`github.search_code`、`sequential-thinking`、`shrimp-task-manager` 当前会话未提供。
- 本轮使用本地代码检索、真实样例运行和现有评测脚本完成等效分析，并把结果记录在本文件与验收报告中。
