## 项目上下文摘要（skimage兼容修复）

生成时间：2026-04-20 09:49:54 CST

### 1. 相似实现分析

- **实现1**: `preprocessing.py`
  - 模式：基础图像处理统一使用 `cv2` + `numpy`
  - 可复用：`load_image`、`detect_text_lines`、`segment_line_characters`
  - 需注意：项目默认依赖本地轻量视觉算子，不依赖训练模型
- **实现2**: `document_detector.py`
  - 模式：字符风格向量由多个特征函数拼接组成
  - 可复用：`_edge_feature`、`_jpeg_feature`、`_clahe_feature`
  - 需注意：特征维度必须稳定，否则离群评分会漂移
- **实现3**: `detector.py`
  - 模式：文档检测结果会与 ELA、噪声图做二次融合
  - 可复用：`_noise_evidence`、`_bbox_score`、`_build_final_detections`
  - 需注意：上游候选排序变化会直接影响最终输出框

### 2. 项目约定

- **命名约定**: Python 使用 `snake_case`，数据模型使用 `@dataclass(slots=True)`
- **文件组织**: `preprocessing.py` 负责基础分割，`document_detector.py` 负责候选生成，`detector.py` 负责融合与输出
- **导入顺序**: 先标准库，再第三方库，再项目内模块
- **代码风格**: 4 空格缩进，中文注释说明意图与约束

### 3. 可复用组件清单

- `preprocessing.py`：图像读取、文本行检测、字符切分
- `document_detector.py`：候选排序、区域合并、特征抽取主流程
- `detector.py`：统一输出结构、报告生成、结果可视化

### 4. 测试策略

- **测试框架**: `unittest`
- **测试模式**: 本地图像样例回归 + CLI 冒烟
- **参考文件**: `test_detector.py`
- **覆盖要求**: 正常检测、CLI 输出、缺图异常、空白图回退

### 5. 依赖和集成点

- **外部依赖**: `opencv-python`、`Pillow`、`numpy`
- **内部依赖**: `main.py -> detector.py -> document_detector.py/preprocessing.py`
- **集成方式**: 直接函数调用
- **配置来源**: `requirements.txt`

### 6. 技术选型理由

- **为什么用这个方案**: 当前报错来自 `scikit-image` 二进制扩展与 `numpy` ABI 不兼容，直接移除该依赖比要求用户重建环境更稳
- **优势**: 运行环境更简单，避免本地 ABI 漂移导致入口直接崩溃
- **劣势和风险**: 自实现 LBP 与骨架细化会与 `skimage` 存在细微数值差异，需要依赖现有回归测试兜底

### 7. 关键风险点

- **边界条件**: 小字符块的纹理和骨架特征容易受边界填充策略影响
- **性能瓶颈**: 形态学骨架细化是循环实现，但输入块很小，预计可接受
- **兼容风险**: 如果未来重新引入 `skimage`，需要避免和当前实现重复
