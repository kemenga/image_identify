## 项目上下文摘要（开放许可数据集生成）

生成时间：2026-04-21 16:08:00 CST

### 1. 相似实现分析

- **实现1**: `evaluation.py:18`
  - 模式：通过 `discover_image_pairs(...)` 自动发现原图与 `检测结果.png` 标签图
  - 可复用：当前评测口径、红框标签抽取和 `IoU >= 0.3` 验收链路
  - 需注意：现有实现只支持同目录配对，需要兼容新数据目录结构
- **实现2**: `main.py:16`
  - 模式：统一在入口文件里定义默认配置和可调参数，再通过 CLI 触发
  - 可复用：`argparse` 风格、默认值配置模式、命令行帮助文案写法
  - 需注意：本轮新增脚本应保持同样的简洁 CLI 风格，不改现有公开调用方式
- **实现3**: `preprocessing.py:33`
  - 模式：在预处理阶段完成图片读取、灰度化和文本行检测
  - 可复用：`load_image(...)`、`detect_text_lines(...)`、`segment_line_characters(...)`
  - 需注意：文档类原图筛选要复用现有文本检测逻辑，而不是重新发明一套文字检测
- **实现4**: `test_detector.py:1`
  - 模式：使用 `unittest` + `tempfile.TemporaryDirectory()` 做本地可重复验证
  - 可复用：临时目录验证、CLI 冒烟测试、结构化断言风格
  - 需注意：网络搜索与下载不应进入单元测试，测试应基于本地合成图像离线完成

### 2. 项目约定

- **命名约定**: Python 使用 `snake_case`；中文文档、日志、报告统一使用简体中文
- **文件组织**: 根目录放 CLI/评测/检测逻辑，测试文件放根目录，工作留痕放 `.codex/`
- **导入顺序**: 标准库、第三方、本地模块
- **代码风格**: 4 空格缩进，中文注释只解释意图和约束，不写修改说明

### 3. 可复用组件清单

- `preprocessing.py::load_image`
- `preprocessing.py::detect_text_lines`
- `preprocessing.py::segment_line_characters`
- `evaluation.py::discover_image_pairs`
- `evaluation.py::evaluate_dataset`
- `test_detector.py` 的 `unittest` 与 CLI 冒烟模式

### 4. 测试策略

- **测试框架**: `unittest`
- **测试模式**: 离线单元测试 + 本地 CLI 冒烟 + 在线 4 图烟雾生成
- **参考文件**: `test_detector.py`
- **覆盖要求**:
  - `search` 产出带许可和下载链接的素材清单
  - `download` 后图片可读且有去重
  - `tamper` 生成图与原图同尺寸
  - `label` 同时产出红框 PNG 和 JSON
  - `evaluation.py --data-dir ./seed_dataset` 可完成评测

### 5. 依赖和集成点

- **外部依赖**: `opencv-python`、`numpy`、`Pillow`
- **网络集成**: Openverse API、Wikimedia Commons MediaWiki API
- **内部集成**: 新脚本与 `preprocessing.py`、`evaluation.py` 协同工作
- **配置来源**: 新脚本内部默认关键词、目录和许可白名单

### 6. 技术选型理由

- **为什么用这个方案**: 你当前缺的是“可重复获取原图并本地篡改”的数据生成能力，最适合用一个独立 CLI 管线解决
- **优势**: 不影响现有检测主流程，可追溯、可重跑、可继续扩量
- **劣势和风险**: 网络接口结果可能波动；完全自动筛掉“原图已被编辑”的素材只能做到启发式近似

### 7. 关键风险点

- **许可风险**: 必须显式过滤掉 `NC`、`ND`、许可缺失样本
- **素材风险**: 需尽量排除标题/描述就提示“已编辑”的图片
- **评测兼容**: 新目录结构不能破坏现有 `data/` 评测口径
- **稳定性**: 单元测试不能依赖联网，联网步骤要放在手工/CLI 冒烟验证
