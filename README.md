# 文档优先的通用图像篡改检测

日期：2026-04-15 09:33（UTC+8）
执行者：Codex

本目录提供一个独立的本地 Python 检测工具，首版优先覆盖短信截图、票据、身份证/证件等文档型图片。实现采用传统计算机视觉方法：以文档规则定位为主，辅以 ELA 和噪声不一致性证据，不包含训练流程或外部模型下载。

如果你想先快速了解仓库在做什么、各文件职责怎么分，可以直接看 [项目梳理.md](./项目梳理.md)。

当前版本还会比较同一张图片中不同文字组件的噪声一致性。当少量文字的高频噪声、拉普拉斯响应或背景残差明显偏离其余文字时，会生成“噪声篡改”候选，用于补强局部文字改动场景。
对于 `data/300.png`、`data/400.png` 这类短信样例，当前实现已经支持在同一张图里把多个真实小改动逐处输出，而不是退化成整行大框。

## 适用范围

- 短信、聊天截图中的时间、金额、短文本篡改
- 票据、小票中的数字修改
- 身份证、证件中的姓名、号码、地址、照片区域篡改
- 发票、票据中少量文字噪声与整体文字噪声不一致的局部篡改
- 文档图上的通用局部异常回退检测

## 快速开始

```bash
pip install -r requirements.txt
python main.py --image ./data/截图.png --output ./detected_result.png --report ./detected_report.json
python main.py --image ./data/截图.png --output ./detected_result.png --report ./detected_report.json --evidence-output-dir ./evidence_maps
python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8
python dataset_builder.py --dataset-root ./seed_dataset search --docs-target 2 --natural-target 2
python dataset_builder.py --dataset-root ./seed_dataset download --docs-target 2 --natural-target 2
python dataset_builder.py --dataset-root ./seed_dataset tamper --seed 42
python dataset_builder.py --dataset-root ./seed_dataset label
python dataset_builder.py --dataset-root ./seed_dataset verify
python evaluation.py --data-dir ./seed_dataset --report ./seed_dataset/manifests/evaluation_report.json --iou-threshold 0.3 --max-detections 2
```

如果你要直接调融合权重、文字噪声阈值、数字窗口阈值等内部参数，可以先看完整 CLI：

```bash
python main.py --help
```

`main.py` 现在已经接入融合层和文档规则层的全部可调参数，例如：

```bash
python main.py \
  --image ./data/400.png \
  --report ./detected_report.json \
  --detector-global-evidence-weight 1.05 \
  --document-text-noise-threshold 5.1 \
  --document-method-weight-stroke 1.4
```

命令执行后，终端会打印本次实际生效的参数覆盖项，方便记录调参组合。

如果你主要想控制“最终结果里保留多少个框”，现在可以优先只调一个融合层参数：`REPORT_CONFIDENCE_THRESHOLD`。它是最终输出的单一置信度闸门，会按统一的报告级置信度同时过滤结果图里的最终 `detections` 和 JSON 里的 `top_candidates`。
- 调高：输出框更少或不增加，更保守
- 调低：输出框更多或不减少，更激进
- 设为 `0.0`：表示不再按报告级置信度剔除候选，最终数量只应继续受“同一物理区域去重”和 `--max-detections`/`MAX_REPORT_CANDIDATES` 这类显式数量上限影响

高阈值模式不只是“框更少”，还会优先保留更可信的主候选：
- 截图场景优先保留 `time_group`
- 票据场景优先保留 `digit_window`
- 身份证和发票优先保留 `*_precise`、`invoice_*`、`embedded_id_photo` 这类语义更强的候选
- `300.png`、`400.png` 这类短信样例继续优先保留真实的 `text_noise_anomaly`

高阈值的语义不是“只要置信度数值高就保留”，而是“在收紧输出数量时优先保留正确候选”。因此高阈值回归要求：截图样例仍保留命中标准答案的 `time_group`，票据样例仍保留命中标准答案的 `digit_window`，`data/300.png` 和 `data/400.png` 仍保留命中标准答案的 `text_noise_anomaly`；同时，被高阈值留下的可见框不应是与标准答案无重叠的错误候选。
验收时会同时检查候选类型和标准答案 IoU：`test_high_report_threshold_keeps_ground_truth_candidates` 使用 `IoU >= 0.3` 的既有评测口径，要求每个高阈值输出框都命中标准答案，并要求每个标准答案框都被高阈值输出覆盖。因此“同类型但位置错误”的候选不能通过高阈值保真回归。

例如：

```bash
python main.py \
  --image ./data/400.png \
  --report ./detected_report.json \
  --detector-report-confidence-threshold 68
```

## 开放许可数据集生成

仓库现在额外提供 `dataset_builder.py`，用于自动搜索开放许可原图，再在本地生成篡改图和标签。它不会下载已经篡改过的图片，默认只接受可修改和再分发的许可：

- `CC0`
- `Public Domain`
- `CC BY`
- `CC BY-SA`

以下情况会被默认拒绝：

- `NC`
- `ND`
- 许可缺失
- 标题或描述中包含 `edited`、`photoshop`、`composite`、`collage`、`manipulation`、`render`、`AI-generated`

默认目录结构如下：

- `seed_dataset/manifests/`
- `seed_dataset/originals/docs/`
- `seed_dataset/originals/natural/`
- `seed_dataset/tampered/`
- `seed_dataset/labels_png/`
- `seed_dataset/labels_json/`

子命令职责如下：

- `search`：调用 Openverse 和 Wikimedia Commons 搜索原图候选，并写出 `search_manifest.json`
- `download`：下载候选原图，做分辨率过滤、许可过滤、感知哈希去重，以及文档类文字密度过滤
- `tamper`：对每张原图固定生成 3 张本地篡改图
- `label`：生成 `xxx检测结果.png` 和同名 JSON 标签
- `verify`：检查尺寸一致性、标签路径、标注框越界和评测配对可用性

文档类默认篡改方法：

- `text_patch_replace`
- `cross_doc_splice`
- `copy_move_token`

自然图默认篡改方法：

- `copy_move_region`
- `cross_image_splice`
- `erase_and_fill`

标签命名规则示例：

- 原图：`doc_0001_orig.png`
- 篡改图：`doc_0001_t01_text_patch_replace.png`
- PNG 标签：`doc_0001_t01_text_patch_replace检测结果.png`
- JSON 标签：`doc_0001_t01_text_patch_replace.json`

当前 `evaluation.py` 已兼容 `seed_dataset/` 结构，直接传 `--data-dir ./seed_dataset` 就会自动读取 `tampered/` 和 `labels_png/` 中的配对样本。

## Python API

```python
from detector import UniversalTamperDetector

detector = UniversalTamperDetector()
result = detector.detect(
    image_path="data/截图.png",
    output_path="detected_result.png",
    report_path="detected_report.json",
    evidence_output_dir="evidence_maps",
)

print(result.status)
print(result.evidence)
print(result.evidence_artifacts)
for item in result.detections:
    print(item.label, item.bbox, item.score)
```

## CLI 参数

- `--image`：输入图片路径
- `--output`：输出高亮结果图路径
- `--report`：输出 JSON 报告路径
- `--max-detections`：最多输出的检测框数量，默认 5
- `--evidence-output-dir`：辅助证据热力图输出目录；传入后生成 ELA、噪声和融合热力图
- `--detector-*`：覆盖融合层参数，例如 `--detector-global-evidence-weight`
- `--detector-report-confidence-threshold`：覆盖 `REPORT_CONFIDENCE_THRESHOLD`，用于单点控制最终输出框数量
- `--document-*`：覆盖文档规则层参数，例如 `--document-text-noise-threshold`
- `--document-global-scan-max-side` / `--document-global-scan-max-windows`：控制大图全图滑窗的最长边缩放和窗口预算；大图卡顿时优先调低这两个值

## 标准答案评测

`evaluation.py` 会把 `data/xxx.png` 与 `data/xxx检测结果.png` 自动配对，文件名中已有 `检测结果` 的图片不会作为原图参与评测。当前默认采用召回优先口径，`IoU >= 0.3` 即视为标准答案框被命中，每张图最多统计 8 个检测框。

```bash
python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8
```

标准答案图中优先抽取红色标注框；当标准答案图和原图尺寸不一致时，会先用 ORB 特征匹配和仿射变换自动对齐，再结合与红框重叠的黄色或橙色外层标注稳定框范围。对齐不足 50 个匹配点、RANSAC 内点比例低于 `0.55` 或缩放比例不在 `0.5-2.0` 时会直接报错，避免静默跳过问题样例。

评测报告包含每张图的标准框、检测框、命中情况、召回率、误报数和平均最佳 IoU。当前 `data/` 中只要存在对应 `xxx检测结果.png`，就会自动进入量化统计，因此 `400.png` 这类新增短信变体也会自动进入硬回归。

从 2026-04-21 起，`image.png`、`image4.png`、`image5.png` 也被纳入显式评测断言，并且已经进入与旧样例同一套硬门槛。当前本地基线是：

- `image.png`：7 个标准框全部命中，最终输出会同时使用 `invoice_token_patch` 与原有 `invoice_large_text_field`。
- `image4.png`：3 个标准框全部命中，最终结果会保留 1 个 `digit_window` 和 2 个 `component_patch_overlay`。
- `image5.png`：3 个标准框全部命中，最终结果收敛为 3 个 `message_bubble_patch`。
- 全集评测：`Recall@IoU0.3 = 0.9630`，总误报数 `= 2`。

## 输出说明

- 结果图只绘制最终筛选出的检测框，适合人工验收。
- 辅助证据热力图仅在传入 `--evidence-output-dir` 时生成，用于观察检测依据。
- JSON 报告固定包含：
  - `status`
  - `reason`
  - `detections`
  - `candidate_count`
  - `reportable_candidate_count`
  - `top_candidates`
  - `evidence`
  - `evidence_artifacts`

`evidence` 中默认保留三类全局证据：

- `ela_score`
- `noise_score`
- `document_score`

每个检测框的 `detail.evidence` 中还会保留局部 ELA、局部噪声以及全局证据分，便于回看候选为什么被保留。

当前默认输出采用“召回不降 + 报告过滤”的平衡策略：内部仍保留完整候选池用于调试，但最终 `detections` 和 `top_candidates` 只展示通过报告级过滤的候选，尽量避免把正确文字直接画框或写进报告。报告级过滤的数量语义统一由 `REPORT_CONFIDENCE_THRESHOLD` 控制：调高阈值只能让输出数量减少或保持不变，调低阈值只能让输出数量增加或保持不变；其他链路只负责候选合法性、同一区域去重和显式上限截断，不能再额外承担“隐藏阈值”的数量控制职责。
当显式传入该阈值时，单阈值模式还会对候选质量做额外加权，高阈值下优先保留时间框、数字框、证件精确字段、发票主字段和真实短信噪声框，而不是把纯粹分值高但语义较弱的碎片热点排到前面。

围绕新增漏检样本，当前文档型候选策略可以概括成三层：

- 专用候选：`time_group`、`digit_window` 这类语义最强的场景规则，优先解决截图时间和票据数字。
- 文本候选：`text_region`、`text_noise_anomaly` 负责承接发票、多字段清单和局部文字噪声异常，是 `image.png`、`image4.png`、`image5.png` 当前最主要的候选来源。
- 兜底候选：`region_anomaly` 仍保留在候选池里兜底，但当前新样例的最终输出已经不再依赖它。

从验收口径看，`REPORT_CONFIDENCE_THRESHOLD` 同时承担数量和质量约束：数量上必须单调收紧，质量上必须避免高置信错误候选压过已知正确场景候选。也就是说，高阈值过滤后的 `detections` 和 `top_candidates` 应该更像“可信候选清单”，而不是“错误候选也能凭高分留下”的原始排序结果。

当传入 `--max-detections 8` 进行评测时，`max_detections` 表示“最多输出数量上限”，而不是“补满到 8 个”。融合层会优先保留真正高可信的候选，例如对大文本块内部的噪声图热点，会输出更小的“噪声篡改”细化框来覆盖只修改少量文字的场景；低可信正常文字候选则不会进入最终报告。
短信类场景目前采用“文字组件簇细化 + 报告级过滤”的组合策略：先在文本块内部找多个局部噪声峰值，再结合截图时间框、票据数字框等专用规则做场景抑制，尽量兼顾 `400.png` 这种多处数字改动召回和旧样例的误报控制。

如果需要做深入调试，Python API 返回值中的 `candidate_regions` 仍会保留完整候选池；报告中的 `top_candidates` 则只展示经过过滤后的可报告候选。

`evidence_artifacts` 会记录辅助证据图路径。默认不生成辅助图时该字段为空字典；传入输出目录后会包含：

- `ela_heatmap`：`ela_heatmap.png`，ELA 压缩误差热力图
- `noise_heatmap`：`noise_heatmap.png`，噪声不一致性热力图
- `fused_heatmap`：`fused_heatmap.png`，按 ELA 和噪声权重融合后的热力图

热力图使用 OpenCV `COLORMAP_JET` 着色，红色表示当前证据图中的相对高值区域，蓝色表示相对低值区域。

检测框的 `label` 可能包括：

- `时间篡改`
- `数字篡改`
- `文字篡改`
- `噪声篡改`
- `局部篡改`

## 当前局限

- 首版是文档优先方案，不承诺自然场景照片篡改的高精度检测。
- 检测结果适合作为辅助判断，不应单独作为取证结论。
- 规则阈值当前内置在代码中，首版未做复杂配置外置。
# image_identify
