## 项目上下文摘要（证据热力图导出）

生成时间：2026-04-20 11:16:01 CST

### 1. 相似实现分析

- **实现1**: `detector.py`
  - 模式：`UniversalTamperDetector.detect` 统一编排检测、可视化和报告写出
  - 可复用：`EvidenceBundle.ela_map`、`EvidenceBundle.noise_map`、`build_report`
  - 需注意：默认检测行为不能因为未传辅助图目录而变化
- **实现2**: `main.py`
  - 模式：`argparse` 定义 CLI 参数，传给 `UniversalTamperDetector.detect`
  - 可复用：已有 `--image`、`--output`、`--report` 参数结构
  - 需注意：新增参数必须使用中文帮助文本
- **实现3**: `test_detector.py`
  - 模式：`unittest` + `tempfile.TemporaryDirectory` 验证输出文件
  - 可复用：CLI 冒烟测试和 `cv2.imread` 文件读取校验
  - 需注意：新测试应覆盖默认空字典和显式导出两条路径

### 2. 项目约定

- **命名约定**: Python 使用 `snake_case`，报告字段使用小写下划线
- **文件组织**: 辅助证据图属于融合层能力，放在 `detector.py`
- **导入顺序**: 标准库、第三方库、本地模块
- **代码风格**: 中文注释和帮助文本，4 空格缩进

### 3. 可复用组件清单

- `EvidenceBundle`: 已包含 ELA 和噪声二维证据图
- `UniversalTamperDetector.LOCAL_ELA_WEIGHT`: 融合热力图权重来源
- `UniversalTamperDetector.LOCAL_NOISE_WEIGHT`: 融合热力图权重来源
- `build_report`: JSON 报告统一出口

### 4. 测试策略

- **测试框架**: `unittest`
- **测试模式**: API 文件输出测试、CLI 文件输出测试、默认报告字段测试、全量回归
- **参考文件**: `test_detector.py`
- **覆盖要求**: 3 张热力图存在、可读、报告记录路径、默认不生成时为空字典

### 5. 依赖和集成点

- **外部依赖**: `opencv-python`、`numpy`、`Pillow`
- **内部依赖**: `main.py -> detector.py`
- **集成方式**: 新增 `evidence_output_dir` 参数向下透传
- **配置来源**: CLI 参数和 Python API 参数

### 6. 技术选型理由

- **为什么用这个方案**: 证据图已经在检测流程中存在，直接导出成本最低且和现有评分一致
- **优势**: 不重复计算、不新增依赖、API 和 CLI 同步可用
- **劣势和风险**: JET 伪彩色图只用于辅助观察，不能替代原始数值分析

### 7. 关键风险点

- **边界条件**: 证据图可能出现常量或非有限值，需要归一化前清理
- **I/O 风险**: 输出目录不可写或图片写入失败时必须抛出清晰异常
- **兼容风险**: 新增报告字段必须不破坏已有字段和默认测试
