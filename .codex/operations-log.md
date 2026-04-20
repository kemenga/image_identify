# operations-log

日期：2026-04-20 09:49:54 CST
执行者：Codex

## 编码前检查 - skimage兼容修复

时间：2026-04-20 09:49:54 CST

- 已查阅上下文摘要文件：`.codex/context-summary-skimage兼容修复.md`
- 将使用以下可复用组件：
  - `preprocessing.py`：保持原有文本分割与字符切分入口不变
  - `document_detector.py` 现有 `_edge_feature`、`_jpeg_feature`、`_clahe_feature`：沿用纯 `cv2/numpy` 特征抽取模式
  - `detector.py`：保持融合输出协议不变
- 将遵循命名约定：Python `snake_case` + `@dataclass(slots=True)`
- 将遵循代码风格：中文注释、4 空格缩进、导入顺序与现有文件一致
- 确认不重复造轮子，证明：已检查 `preprocessing.py`、`document_detector.py`、`detector.py`，项目内不存在可直接复用的 LBP 或骨架函数

## 过程记录

- 09:40 工具：`rg`
  参数：检索 `skimage`、`local_binary_pattern`、`skeletonize` 的实际使用点
  输出摘要：确认仅 `document_detector.py` 的 `_texture_feature` 与 `_stroke_feature` 依赖 `skimage`
- 09:43 工具：`sed` / `nl`
  参数：阅读 `main.py`、`detector.py`、`document_detector.py`、`preprocessing.py`、`test_detector.py`
  输出摘要：确认项目分层、测试框架与输出协议，找到 3 处可复用模式
- 09:46 工具：`python -m pip show`
  参数：检查 `numpy` 与 `scikit-image` 版本
  输出摘要：当前环境为 `numpy 2.4.4`、`scikit-image 0.24.0`，入口报错为二进制 ABI 不兼容
- 09:50 工具：`apply_patch`
  参数：替换 `document_detector.py` 中的 `skimage` 调用，清理 `requirements.txt`
  输出摘要：新增纯 `opencv/numpy` 的 uniform LBP 与骨架细化实现，移除 `scikit-image`、`scipy` 依赖声明
- 09:52 工具：`python -m unittest test_detector.py`
  参数：执行本地回归测试
  输出摘要：6 项测试全部通过，覆盖短信截图、票据、身份证、CLI 输出、缺图异常与空白图回退
- 09:53 工具：`python main.py --image ./截图.png --output ./tmp_result.png --report ./tmp_report.json`
  参数：执行 CLI 冒烟验证
  输出摘要：成功生成结果图与 JSON 报告，入口不再因 `skimage` 导入失败而中断
- 09:55 工具：`rg` / `sed` / `nl`
  参数：梳理 `main.py`、`detector.py`、`preprocessing.py`、`document_detector.py`、`test_detector.py`、`README.md`
  输出摘要：确认仓库实际是文档型图像篡改检测工具，核心为传统视觉规则引擎而非深度学习项目
- 09:57 工具：`apply_patch`
  参数：新增 `项目梳理.md`，在 `README.md` 增加文档入口
  输出摘要：补充了项目目标、分层结构、核心流程、样例说明与当前代码特点，便于后续维护和交接
- 09:58 工具：`python -m unittest test_detector.py`
  参数：执行本地回归验证
  输出摘要：6 项测试全部通过，文档整理未影响现有检测行为

## 编码后声明 - skimage兼容修复

时间：2026-04-20 09:53:30 CST

### 1. 复用了以下既有组件

- `preprocessing.py`：继续提供文本行与字符切分，不改动调用协议
- `document_detector.py` 既有特征函数：沿用 `_edge_feature`、`_jpeg_feature`、`_clahe_feature` 的纯 `cv2/numpy` 思路
- `detector.py`：保持最终检测结果、报告结构与可视化流程不变

### 2. 遵循了以下项目约定

- 命名约定：新增函数采用 `snake_case`，与 `document_detector.py` 现有私有方法一致
- 代码风格：中文注释说明设计意图，导入顺序保持“标准库/第三方/本地模块”
- 文件组织：只在特征提取层替换实现，没有改动上层 API 和测试入口

### 3. 对比了以下相似实现

- `document_detector.py::_edge_feature`：同样使用局部图块、统计特征、固定维度输出，我的方案保持这一约束不变
- `document_detector.py::_jpeg_feature`：同样通过纯 `cv2/numpy` 计算稳定特征，我的方案沿用纯本地算子而非新增重依赖
- `preprocessing.py::segment_line_characters`：同样强调小块图像的边界裁切与鲁棒性，我的 LBP 实现使用反射边界以降低边缘抖动

### 4. 未重复造轮子的证明

- 检查了 `preprocessing.py`、`document_detector.py`、`detector.py`，仓库内不存在现成的 LBP 或骨架细化工具
- 如果存在类似功能，我的差异化价值是：直接移除 ABI 脆弱依赖，保证当前入口在本地环境可运行

### 5. 工具偏差说明

- 仓库规范中提到的 `desktop-commander`、`context7`、`github.search_code`、`sequential-thinking`、`shrimp-task-manager` 当前环境未提供
- 已使用本地代码检索、现有测试与实际运行结果完成等效上下文分析和验证，并在本日志中留痕

## 编码前检查 - 项目梳理

时间：2026-04-20 09:55:18 CST

- 已查阅上下文摘要文件：`.codex/context-summary-项目梳理.md`
- 将使用以下可复用组件：
  - `main.py`：作为入口层说明来源
  - `detector.py`：作为统一对外 API 和报告结构说明来源
  - `document_detector.py`：作为规则引擎与候选生成说明来源
  - `preprocessing.py`：作为文本行与字符切分说明来源
- 将遵循命名约定：中文文档描述项目结构，代码引用保持原始标识符
- 将遵循代码风格：仓库内文档使用简体中文，描述聚焦职责和集成关系
- 确认不重复造轮子，证明：已检查现有 `README.md`，其内容偏使用说明，尚未覆盖完整架构梳理

## 编码后声明 - 项目梳理

时间：2026-04-20 09:57:00 CST

### 1. 复用了以下既有组件

- `main.py`：梳理 CLI 入口职责和参数
- `detector.py`：梳理统一检测器、融合逻辑、报告输出
- `document_detector.py`：梳理候选规则类型与核心检测流程
- `preprocessing.py`：梳理文本行和字符切分职责

### 2. 遵循了以下项目约定

- 命名约定：新增文档文件名和内容均使用中文，代码标识符保持原样引用
- 代码风格：文档内容以简体中文为主，聚焦“做什么”和“怎么串起来”
- 文件组织：未改动核心代码结构，只补充项目说明文档

### 3. 对比了以下相似实现

- `README.md`：原本更偏“如何运行”，新增文档补的是“代码结构和项目本质”
- `test_detector.py`：从测试反推项目支持的主要场景和验收目标
- `detector.py::build_report`：据此明确了输出 JSON 的结构说明

### 4. 未重复造轮子的证明

- 检查了 `README.md` 与 `.codex` 现有文件，尚无面向仓库阅读者的系统性项目梳理文档
- 本次新增文档用于降低阅读门槛，不引入新的实现逻辑

## 编码前检查 - 证据热力图导出

时间：2026-04-20 11:16:01 CST

- 已查阅上下文摘要文件：`.codex/context-summary-证据热力图导出.md`
- 将使用以下可复用组件：
  - `EvidenceBundle.ela_map`：生成 `ela_heatmap.png`
  - `EvidenceBundle.noise_map`：生成 `noise_heatmap.png`
  - `UniversalTamperDetector.LOCAL_ELA_WEIGHT` 与 `LOCAL_NOISE_WEIGHT`：生成融合热力图
  - `build_report`：追加 `evidence_artifacts` 报告字段
- 将遵循命名约定：新增参数和报告字段使用 `evidence_output_dir`、`evidence_artifacts`
- 将遵循代码风格：中文注释与 CLI 帮助文本，4 空格缩进
- 确认不重复造轮子，证明：已检查 `detector.py`，现有 `ela_map` 和 `noise_map` 已满足输入，不需要重算或新增依赖

## 过程记录 - 证据热力图导出

- 11:17 工具：`apply_patch`
  参数：修改 `detector.py`
  输出摘要：新增 `evidence_output_dir` API 参数、`evidence_artifacts` 结果字段、三类证据热力图导出逻辑和报告字段
- 11:18 工具：`apply_patch`
  参数：修改 `main.py`
  输出摘要：新增 CLI 参数 `--evidence-output-dir`，并在生成辅助图时打印输出目录
- 11:19 工具：`apply_patch`
  参数：修改 `test_detector.py`
  输出摘要：新增 API 和 CLI 热力图输出测试，并断言默认报告字段为空字典
- 11:20 工具：`apply_patch`
  参数：修改 `README.md` 与 `项目梳理.md`
  输出摘要：补充辅助证据图的调用方式、输出文件和报告字段说明
- 11:21 工具：`python -m unittest test_detector.py`
  参数：执行本地回归测试
  输出摘要：8 项测试全部通过，覆盖原有 6 项和新增 2 项证据热力图输出能力

## 编码后声明 - 证据热力图导出

时间：2026-04-20 11:21:00 CST

### 1. 复用了以下既有组件

- `EvidenceBundle.ela_map`：用于生成 ELA 热力图
- `EvidenceBundle.noise_map`：用于生成噪声热力图
- `UniversalTamperDetector.LOCAL_ELA_WEIGHT` 与 `LOCAL_NOISE_WEIGHT`：用于融合热力图权重
- `build_report`：用于统一输出 `evidence_artifacts`

### 2. 遵循了以下项目约定

- 命名约定：新增字段使用 `evidence_output_dir` 和 `evidence_artifacts`
- 代码风格：新增注释和 CLI 帮助文本使用简体中文
- 文件组织：辅助证据图导出位于融合层 `detector.py`，入口层只负责参数透传

### 3. 对比了以下相似实现

- `detector.py::detect`：沿用原有输出图和报告写出流程，只增加可选辅助图分支
- `main.py::parse_args`：沿用现有 `argparse` 参数定义风格
- `test_detector.py::test_cli_writes_output_and_report`：沿用临时目录和 CLI 子进程验证方式

### 4. 未重复造轮子的证明

- 检查了 `detector.py`，已有 ELA 和噪声二维证据图，无需重建计算链路
- 检查了 `requirements.txt`，现有 `opencv-python` 已可完成伪彩色热力图生成，无需新增依赖

## 编码前检查 - 文字噪声一致性检测

时间：2026-04-20 11:32:00 CST

- 已查阅上下文摘要文件：`.codex/context-summary-文字噪声一致性检测.md`
- 将使用以下可复用组件：
  - `document_detector.py::_text_component_boxes`：提取文字组件作为噪声比较单元
  - `document_detector.py::_robust_scores`：计算文字组件噪声离群分
  - `document_detector.py::_candidate_threshold`：接入新候选阈值
  - `document_detector.py::_selection_priority`：控制新候选排序权重
- 将遵循命名约定：新增候选类型 `text_noise_anomaly`
- 将遵循代码风格：中文注释、保守阈值、与现有候选生成阶段一致
- 确认不重复造轮子，证明：已检查现有风格、区域和文本候选，尚无“文字组件间噪声一致性”比较逻辑

## 过程记录 - 文字噪声一致性检测

- 15:00 工具：`find` / `rg` / `sed`
  参数：检查 `data/` 样例、现有候选生成函数、候选阈值和最终排序逻辑
  输出摘要：确认 `data/` 包含 300、发票、截图、票据、身份证等样例，现有检测缺少文字组件间噪声一致性比较
- 15:01 工具：`python`
  参数：运行当前检测器遍历 `data/*.png`
  输出摘要：确认旧逻辑主要输出时间、文字块、区域滑窗等候选，发票样例没有噪声一致性专用候选
- 15:02 工具：`apply_patch`
  参数：修改 `document_detector.py`
  输出摘要：新增 `text_noise_anomaly` 候选类型，基于文字组件高频残差、拉普拉斯响应和背景残差做全图/同行鲁棒离群比较
- 15:03 工具：`apply_patch`
  参数：修改 `test_detector.py`
  输出摘要：样例路径兼容根目录与 `data/`，新增数据样例噪声候选测试
- 15:04 工具：`python -m unittest test_detector.py`
  参数：执行本地回归测试
  输出摘要：9 项测试全部通过
- 15:04 工具：`python main.py --image ./data/发票.png --output ./tmp_noise_result.png --report ./tmp_noise_report.json --evidence-output-dir ./tmp_noise_maps`
  参数：执行发票样例冒烟验证
  输出摘要：第一检测项为“噪声篡改”，框为 `(753, 422, 194, 53)`，同时生成证据热力图

## 编码后声明 - 文字噪声一致性检测

时间：2026-04-20 15:04:55 CST

### 1. 复用了以下既有组件

- `_text_component_boxes`：提取文字组件作为噪声比较对象
- `_robust_scores`：对文字噪声特征做鲁棒离群评分
- `_candidate_threshold`：为 `text_noise_anomaly` 设置独立阈值
- `_selection_priority`：控制新候选不压过时间、数字和证件精确规则

### 2. 遵循了以下项目约定

- 命名约定：新增候选类型为 `text_noise_anomaly`
- 代码风格：新增注释为简体中文，说明噪声一致性比较意图
- 文件组织：新增逻辑集中在 `document_detector.py` 候选生成阶段，未改动上层 API

### 3. 对比了以下相似实现

- `_enumerate_text_block_regions`：同样使用文字组件，但本次新增的是噪声特征离群比较
- `_enumerate_text_window_anomalies`：同样使用鲁棒热点思想，但本次只在文字组件集合内比较
- `_build_type_scores`：复用了“全局分 + 行内分”的上下文融合思路

### 4. 未重复造轮子的证明

- 检查了 `document_detector.py` 中已有风格、区域、文字块和证件规则，未发现文字组件间噪声一致性比较
- 本次没有新增依赖，没有引入 OCR 或模型，仅补充现有传统视觉规则引擎的一个证据源
