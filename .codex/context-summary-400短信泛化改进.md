## 项目上下文摘要（400短信泛化改进）

生成时间：2026-04-21 09:39:20 CST

### 1. 相似实现分析

- **实现1**: `detector.py:293`
  - 模式：在融合层基于证据图补充候选，而不是直接改文档规则主流程
  - 可复用：`_build_evidence_text_noise_candidates(...)`
  - 需注意：新增短信类候选必须继续走最终报告过滤，不能直接写进输出
- **实现2**: `detector.py:364`
  - 模式：在文本块内部做二次细化，输出更小的局部框
  - 可复用：`_find_noisy_component_clusters_in_text_box(...)`
  - 需注意：需要控制单块最多输出数量，避免退化成整行误报
- **实现3**: `detector.py:735`
  - 模式：候选先做报告级排序和场景过滤，再决定最终输出
  - 可复用：`_reportable_candidates(...)`
  - 需注意：短信类通用候选不能抢掉截图时间框、票据数字框等专用场景主框
- **实现4**: `test_detector.py:224`
  - 模式：通过 `evaluate_dataset(...)` 固化数据集召回和误报门槛
  - 可复用：`test_dataset_evaluation_recall_threshold`
  - 需注意：本轮需要把 `400.png` 明确纳入硬回归，而不是只依赖自动发现

### 2. 项目约定

- **命名约定**: Python 使用 `snake_case`，检测类型沿用现有英文标识，例如 `text_noise_anomaly`
- **文件组织**: `main.py` 做入口与调参，`detector.py` 做融合与报告过滤，`document_detector.py` 做规则候选，`evaluation.py` 做标准答案评测
- **导入顺序**: 标准库、第三方、本地模块
- **代码风格**: 中文注释与文档，4 空格缩进，`@dataclass(slots=True)` 用于结果结构

### 3. 可复用组件清单

- `detector.py::_build_evidence_text_noise_candidates`：补充融合层文字噪声细化候选
- `detector.py::_find_noisy_component_clusters_in_text_box`：在文本块内部提取多个局部噪声峰值
- `detector.py::_reportable_candidates`：做报告级排序、限流和去重
- `detector.py::_best_candidate_of_type`：专用场景主候选选择
- `detector.py::_is_special_scene_conflicting_refinement`：避免短信细化候选抢占时间框或数字框
- `evaluation.py::evaluate_dataset`：统一计算召回、误报和 IoU

### 4. 测试策略

- **测试框架**: `unittest`
- **测试模式**: 单元测试 + CLI 冒烟 + 数据集评测
- **参考文件**: `test_detector.py`
- **覆盖要求**:
  - `400.png` 必须逐处命中两个标准框
  - `截图.png` 第一主框仍为 `time_group`
  - `票据.png` 第一主框仍为 `digit_window`
  - 总体 `Recall@IoU0.3 >= 0.90`
  - 总误报数 `<= 3`

### 5. 依赖和集成点

- **外部依赖**: `opencv-python`、`numpy`、`Pillow`
- **内部依赖**: `detector.py` 依赖 `document_detector.py` 的候选池和文本组件
- **集成方式**: `main.py` 创建 `UniversalTamperDetector`，`evaluation.py` 复用同一检测入口
- **配置来源**: `main.py` 顶部 `RUN_CONFIG`、`DETECTOR_PARAM_DEFAULTS`、`DOCUMENT_PARAM_DEFAULTS`

### 6. 技术选型理由

- **为什么用这个方案**: 当前项目不引入 OCR 或深度学习，只能在现有传统视觉框架内提升短信类局部数字改动的泛化
- **优势**: 不改公开 API，不新增依赖，直接复用现有证据图、文本组件和报告过滤体系
- **劣势和风险**: 若过滤过松，容易重新引入正常文字误报；若过滤过严，又会压掉 `400.png` 这种多处小改动

### 7. 关键风险点

- **边界条件**: 同图多处改动时不能只保留一个最佳窗口
- **场景冲突**: 通用噪声细化候选可能抢掉截图时间框或票据数字框
- **性能瓶颈**: 文本块内部组件聚类和多候选排序会增加一定计算量
- **工具偏差**: 仓库规范提到的 `desktop-commander`、`context7`、`github.search_code`、`sequential-thinking` 当前环境未提供，本轮采用本地代码检索、现有测试和数据集评测做等效验证
