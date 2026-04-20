## 项目上下文摘要（文字噪声一致性检测）

生成时间：2026-04-20 11:32:00 CST

### 1. 相似实现分析

- **实现1**: `document_detector.py::_enumerate_text_block_regions`
  - 模式：基于文本组件聚类生成 `text_region` 候选
  - 可复用：`_text_component_boxes`、`_snap_text_cluster_to_components`、`_merge_text_region_candidates`
  - 需注意：候选分数进入统一阈值和最终裁决
- **实现2**: `document_detector.py::_enumerate_text_window_anomalies`
  - 模式：滑窗提特征，用鲁棒分数筛热点，再贴合文字组件
  - 可复用：`_robust_scores`、`_bbox_overlap_ratio`、`_center_distance`
  - 需注意：热点窗口容易误报，因此需要文本组件约束
- **实现3**: `document_detector.py::_build_type_scores`
  - 模式：先算全局离群，再按行内上下文融合
  - 可复用：全局和局部上下文结合的打分思想
  - 需注意：文本行字符切分在身份证场景较弱，因此噪声检测应优先用组件盒而非只用字符盒

### 2. 项目约定

- **命名约定**: Python 使用 `snake_case`，候选类型使用小写下划线
- **文件组织**: 文档规则候选生成集中在 `document_detector.py`
- **导入顺序**: 不新增依赖，继续使用 `cv2` 和 `numpy`
- **代码风格**: 中文注释说明检测意图和风险控制

### 3. 可复用组件清单

- `preprocessing.detect_text_lines`: 文本行上下文
- `document_detector._text_component_boxes`: 文本组件候选
- `document_detector._robust_scores`: 鲁棒离群评分
- `document_detector._candidate_threshold`: 统一阈值入口
- `document_detector._selection_priority`: 最终候选排序入口

### 4. 测试策略

- **测试框架**: `unittest`
- **测试模式**: 私有候选生成验证 + 现有样例回归
- **参考文件**: `test_detector.py`
- **覆盖要求**: 新候选能在数据样例产生，原有时间/票据/身份证断言不回退

### 5. 依赖和集成点

- **外部依赖**: `opencv-python`、`numpy`、`Pillow`
- **内部依赖**: `TraditionalTamperDetector.detect` 第三阶段候选生成
- **集成方式**: 新增 `text_noise_anomaly` 候选类型，与其他候选一起排序筛选
- **配置来源**: 类常量阈值，不新增配置文件

### 6. 技术选型理由

- **为什么用这个方案**: 用户指出篡改通常只改少量文字，局部文字噪声相对其他文字不一致是有效证据
- **优势**: 不依赖 OCR，不需要模型，直接复用已有文本组件和鲁棒统计
- **劣势和风险**: 扫描件、压缩边缘或印章附近可能也有噪声差异，因此必须保持阈值保守并给专用语义规则更高优先级

### 7. 关键风险点

- **误报风险**: 大量背景纹理或表格线可能被当作文字组件
- **边界条件**: 文本组件数量太少时不能可靠比较，需要直接跳过
- **排序风险**: 新候选不能压过时间、数字、证件精确规则
