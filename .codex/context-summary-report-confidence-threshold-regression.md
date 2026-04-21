## 项目上下文摘要（REPORT_CONFIDENCE_THRESHOLD 回归测试与文档）

生成时间：2026-04-21 10:17:22 CST

### 1. 相似实现分析

- **实现1**: `test_detector.py` 中 `test_detector_accepts_tunable_overrides`
  - 模式：通过 `detector_overrides` 验证融合层参数可由构造器覆盖。
  - 可复用：`UniversalTamperDetector(detector_overrides=...)`。
  - 需注意：只验证参数接线，不验证参数对最终输出数量的行为。
- **实现2**: `test_detector.py` 中 `test_cli_accepts_tunable_overrides`
  - 模式：使用 `subprocess.run` 执行 CLI，检查参数覆盖提示与报告写出。
  - 可复用：CLI 子进程测试结构、临时目录和报告读取方式。
  - 需注意：该用例关注 CLI 透传，不适合精确断言候选筛选链路。
- **实现3**: `test_detector.py` 中 `test_single_report_threshold_can_reduce_output_boxes`
  - 模式：使用真实样例验证极高 `REPORT_CONFIDENCE_THRESHOLD` 会减少最终 `detections`。
  - 可复用：阈值调高减少输出框的断言口径。
  - 需注意：真实图像样例受候选生成影响，无法单独定位 `_report_limit`、去重和 `max_detections` 对最终数量的影响。
- **实现4**: `detector.py` 中 `_reportable_candidates`
  - 模式：先按 `_is_reportable_candidate` 过滤，再按 `_report_confidence` 排序，随后应用 `min_confidence`、`_report_limit`、去重和 `max_items`。
  - 可复用：直接调用该方法可以用合成候选覆盖报告级筛选链路。
  - 需注意：本轮不修改 `detector.py`，只补测试暴露当前链路约束。

### 2. 项目约定

- **命名约定**: 测试方法使用 `test_...`，辅助函数使用下划线前缀，检测类型沿用 `time_group`、`digit_window`、`text_region`、`text_noise_anomaly` 等英文标识。
- **文件组织**: 单元测试集中在仓库根目录 `test_detector.py`；用户说明文档在 `README.md`；项目结构说明在 `项目梳理.md`；本轮记录写入 `.codex/`。
- **导入顺序**: 标准库、第三方库、本地模块分组导入。
- **代码风格**: 使用 `unittest` 断言；说明性文本和注释使用简体中文。

### 3. 可复用组件清单

- `detector.py::TamperRegion`: 构造合成候选，避免依赖图像候选生成波动。
- `detector.py::UniversalTamperDetector._reportable_candidates`: 直接验证最终报告筛选链路。
- `detector.py::UniversalTamperDetector._report_confidence`: 计算阈值分界点，避免在测试中硬编码内部加权细节。
- `test_detector.py::sample_image_path`: 真实样例检测测试继续复用现有路径解析。

### 4. 测试策略

- **测试框架**: Python 标准库 `unittest`。
- **测试模式**: 对真实样例保留已有冒烟测试；对阈值语义新增合成候选链路测试。
- **参考文件**: `test_detector.py`。
- **覆盖要求**: 覆盖阈值调高会减少输出框、阈值调低会增加或至少不减少输出框，并验证低阈值时输出数量只受 `max_items/max_detections` 上限约束。

### 5. 依赖和集成点

- **外部依赖**: 测试文件已有 `cv2`、`numpy`，本轮不新增外部依赖。
- **内部依赖**: `test_detector.py` 从 `detector.py` 导入 `UniversalTamperDetector`，本轮新增导入 `TamperRegion`。
- **集成方式**: 合成候选直接进入 `_reportable_candidates`，覆盖 `_is_reportable_candidate`、`_report_confidence`、`_report_limit`、去重和 `max_items` 的组合效果。
- **配置来源**: `REPORT_CONFIDENCE_THRESHOLD` 可通过构造器覆盖和 CLI `--detector-report-confidence-threshold` 设置。

### 6. 技术选型理由

- **为什么用合成候选**: 需求关注报告级阈值语义，而非图像候选生成质量；合成候选能稳定复现多个可报告候选被同一阈值控制的场景。
- **优势**: 测试快、定位准、不会被样例图细节或 OpenCV 版本差异放大。
- **劣势和风险**: 该测试不验证整图检测真实候选数量，只验证最终报告筛选链路；真实图像候选不足时仍需另行补候选生成测试。

### 7. 关键风险点

- **并发问题**: 无并发逻辑。
- **边界条件**: 阈值位于候选置信度之间、阈值为 `0.0`、`max_items` 小于候选数时的数量关系。
- **性能瓶颈**: 合成候选数量很小，无性能风险。
- **安全考虑**: 本轮不涉及认证、鉴权、加密或外部网络访问。

### 8. 工具可用性说明

- 当前会话未提供 `sequential-thinking`、`shrimp-task-manager`、`desktop-commander`、`context7`、`github.search_code` 工具。
- 已使用本地只读命令 `rg`、`sed`、`git diff` 替代完成代码检索与差异识别，并在操作日志中记录该限制。
