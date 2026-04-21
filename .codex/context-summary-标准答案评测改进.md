## 项目上下文摘要（标准答案评测改进）

生成时间：2026-04-20 15:35:32 CST

### 1. 相似实现分析

- **实现1**: `detector.py`
  - 模式：`UniversalTamperDetector.detect` 统一调用文档规则检测器，再融合 ELA 与噪声证据。
  - 可复用：`EvidenceBundle`、`_bbox_score`、`_convert_region`、`_build_final_detections`。
  - 需注意：截图和票据已有高置信专用规则，新增候选不能抢占第一主框。
- **实现2**: `document_detector.py`
  - 模式：先生成多来源候选，再通过 `_candidate_threshold` 和 `_select_regions` 统一筛选。
  - 可复用：`TamperRegion` 数据结构、候选 `detail` 字段、文本组件与局部噪声一致性思路。
  - 需注意：当前文档层最多输出 5 个主检测框，评测模式需要在融合层补足 `max_detections=8`。
- **实现3**: `test_detector.py`
  - 模式：使用 `unittest`、真实样例图片和本地 CLI 子进程做回归验证。
  - 可复用：`sample_image_path`、`overlaps`、临时目录和 JSON 报告断言方式。
  - 需注意：新增评测测试应沿用现有本地子进程验证，不引入远程或人工验收。
- **实现4**: `evaluation.py`
  - 模式：从 `data/xxx.png` 配对 `data/xxx检测结果.png`，抽取彩色标注框并用 IoU 统计召回。
  - 可复用：`discover_image_pairs`、`extract_ground_truth_boxes`、`evaluate_dataset`、`bbox_iou`。
  - 需注意：尺寸不一致图必须走 ORB 仿射对齐，失败时抛出明确错误而不是跳过。

### 2. 项目约定

- **命名约定**: 文件名使用小写英文或中文文档名；函数与变量使用 Python `snake_case`；类使用 `PascalCase`。
- **文件组织**: CLI 在 `main.py`，融合逻辑在 `detector.py`，文档规则在 `document_detector.py`，评测逻辑放在独立 `evaluation.py`。
- **导入顺序**: 标准库、第三方库、项目内部模块分组导入。
- **代码风格**: Python 3 类型标注，数据结构使用 `dataclass(slots=True)`，注释使用简体中文解释设计意图。

### 3. 可复用组件清单

- `detector.py:UniversalTamperDetector`: 统一 API 与 CLI 检测入口。
- `detector.py:EvidenceBundle`: 已有 ELA 与噪声证据图，适合补充证据图驱动候选。
- `document_detector.py:TamperRegion`: 文档候选的数据协议，融合层需要兼容其字段。
- `evaluation.py:bbox_iou`: 评测与调试共享 IoU 计算。
- `test_detector.py:sample_image_path`: 测试中定位根目录或 `data/` 样例。

### 4. 测试策略

- **测试框架**: Python 标准库 `unittest`。
- **测试模式**: 样例图回归测试、CLI 子进程测试、评测报告结构测试。
- **参考文件**: `test_detector.py`。
- **覆盖要求**: 标准答案抽框、尺寸不一致自动对齐、数据集召回阈值、CLI 评测报告、既有截图/票据/身份证/热力图/缺图/空白图回归。

### 5. 依赖和集成点

- **外部依赖**: OpenCV、NumPy、Pillow，保持不新增依赖。
- **内部依赖**: `evaluation.py` 调用 `UniversalTamperDetector`；`detector.py` 调用 `TraditionalTamperDetector` 和 `preprocessing.load_image`。
- **集成方式**: `evaluation.py` 作为独立 CLI，同时被 `test_detector.py` 直接导入。
- **配置来源**: 评测 CLI 参数 `--data-dir`、`--report`、`--iou-threshold`、`--max-detections`。

### 6. 技术选型理由

- **为什么用这个方案**: 用户已有人工标注结果图，直接从标注图抽框可以形成可重复的本地召回基准。
- **优势**: 不依赖 OCR、深度学习模型或人工复核；可用真实样例持续驱动阈值调参。
- **劣势和风险**: 标注图是半透明色块而非严格矩形框，红色区域可能产生碎片；尺寸不一致图依赖 ORB 对齐质量。

### 7. 关键风险点

- **并发问题**: 无长期进程或并发写入，主要风险是测试临时文件路径冲突，使用 `TemporaryDirectory` 避免。
- **边界条件**: 标准答案缺失、标注颜色缺失、ORB 特征不足、映射后框越界、空白图无文本。
- **性能瓶颈**: 数据集评测会对每张图完整跑检测，当前样例少，耗时可接受；滑窗补充候选限制在已有文本框内。
- **安全考虑**: 本任务不引入认证、鉴权、加密或网络访问能力。

### 8. 工具替代记录

- 用户要求使用 `sequential-thinking`、`shrimp-task-manager`、`desktop-commander`、`context7` 和 `github.search_code`，但当前会话未暴露这些工具入口。
- 已尝试开启两个子 agent；两个子 agent 均因上游 `502 Bad Gateway` 失败，无可用输出。
- 本轮采用本地命令、代码阅读、真实样例评测和 `unittest` 作为替代验证路径，并在操作日志与验收报告中留痕。
