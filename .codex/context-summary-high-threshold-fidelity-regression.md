## 项目上下文摘要（高阈值保真回归测试与文档）

生成时间：2026-04-21 11:21:06 CST

### 1. 相似实现分析

- **实现1**: `test_detector.py` 中 `test_single_report_threshold_can_reduce_output_boxes`
  - 模式：用真实截图样例验证极高 `REPORT_CONFIDENCE_THRESHOLD` 会收紧最终输出。
  - 可复用：通过 `UniversalTamperDetector(detector_overrides=...)` 显式进入报告阈值调参路径。
  - 需注意：该用例只验证数量减少，没有验证保留下来的候选是否命中标准答案。
- **实现2**: `test_detector.py` 中 `test_reportable_outputs_suppress_correct_text_false_positives`
  - 模式：调用 `evaluate_dataset`，按标准答案 IoU 检查误报、召回和关键类型。
  - 可复用：`bbox_iou` 与标准答案匹配口径，适合判断候选质量。
  - 需注意：默认评测路径没有显式传入 `REPORT_CONFIDENCE_THRESHOLD`，不能覆盖用户调高阈值后的路径。
- **实现3**: `test_detector.py` 中 `test_report_threshold_lowering_increases_reportable_box_count`
  - 模式：用合成候选覆盖报告级阈值单调性。
  - 可复用：直接围绕 `REPORT_CONFIDENCE_THRESHOLD` 的行为写回归断言。
  - 需注意：合成候选不覆盖截图、票据、300、400 这些真实样例的候选质量。
- **实现4**: `detector.py` 中 `_reportable_candidates`
  - 模式：显式传入阈值时进入 `single_threshold_mode`，先做宽松合法性过滤，再按报告置信度、阈值和去重选候选。
  - 可复用：测试可通过真实 API 触发完整最终输出路径。
  - 需注意：当前宽松模式里高置信通用噪声候选可能比专用正确候选更容易留下。

### 2. 项目约定

- **命名约定**: Python 测试方法使用 `test_...`，候选类型沿用 `time_group`、`digit_window`、`text_noise_anomaly`。
- **文件组织**: 回归测试集中在 `test_detector.py`；用户说明写入 `README.md`；架构和流程说明写入 `项目梳理.md`；过程记录写入项目本地 `.codex/`。
- **导入顺序**: 标准库、第三方、本地模块分组导入，当前无需新增导入。
- **代码风格**: 继续使用 `unittest`、`self.subTest` 和 `self.assert...` 断言。

### 3. 可复用组件清单

- `test_detector.py::sample_image_path`: 统一查找根目录或 `data/` 下的样例图。
- `evaluation.py::extract_ground_truth_boxes`: 从标准答案图中抽取标准框。
- `evaluation.py::bbox_iou`: 判断检测框是否命中标准答案。
- `detector.py::UniversalTamperDetector`: 通过 `detector_overrides` 覆盖报告级阈值。

### 4. 测试策略

- **测试框架**: Python `unittest`。
- **测试模式**: 使用真实样例和标准答案图做高阈值保真回归。
- **参考文件**: `test_detector.py` 中样例直测、数据集评测和阈值测试。
- **覆盖要求**: 高阈值路径至少覆盖 `截图.png` 保留 `time_group`、`票据.png` 保留 `digit_window`、`300.png` 和 `400.png` 保留 `text_noise_anomaly`，并要求保留下来的候选都命中标准答案。

### 5. 依赖和集成点

- **外部依赖**: OpenCV、NumPy 已由现有测试使用，本轮不新增依赖。
- **内部依赖**: `test_detector.py` 调用 `detector.py`、`evaluation.py` 和 `data/` 标准答案图。
- **集成方式**: 真实检测结果进入 IoU 质量断言，覆盖候选生成、报告级过滤和最终输出组合链路。
- **配置来源**: `REPORT_CONFIDENCE_THRESHOLD` 通过 `detector_overrides` 显式传入。

### 6. 技术选型理由

- **为什么用真实样例**: 用户关注高阈值后保留下来的是否是正确候选，必须绑定真实标准答案，而不是只看候选数量。
- **优势**: 能直接暴露截图、票据、短信样例中高置信错误候选抢占的问题。
- **劣势和风险**: 当前允许范围不包含 `detector.py`，新增测试可能先失败，用于推动后续生产逻辑修复。

### 7. 关键风险点

- **并发问题**: 无并发。
- **边界条件**: 标准答案图缺失时测试会失败，符合当前数据集硬回归定位。
- **性能瓶颈**: 新测试会额外跑 4 张真实样例，完整测试耗时会增加。
- **安全考虑**: 本轮不涉及认证、鉴权或加密。

### 8. 续补检查记录

生成时间：2026-04-21 11:34:24 CST

- **发现缺口**: 当前 `test_high_report_threshold_keeps_ground_truth_candidates` 已使用 `95.0` 阈值并断言截图、票据、`300.png`、`400.png` 的候选类型，但没有把“命中标准答案、无无重叠错误候选”落实为 IoU 断言。
- **补强方案**: 在 `test_detector.py` 中复用 `extract_ground_truth_boxes` 与 `bbox_iou`，新增高阈值候选到标准答案、标准答案到候选的双向 `IoU >= 0.3` 断言。
- **当前观测**: 阈值 `95.0` 下，截图、票据、`300.png` 和 `400.png` 的当前输出均能命中对应标准答案，`400.png` 两个输出分别覆盖两个标准答案框。
