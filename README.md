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
- `--document-*`：覆盖文档规则层参数，例如 `--document-text-noise-threshold`

## 标准答案评测

`evaluation.py` 会把 `data/xxx.png` 与 `data/xxx检测结果.png` 自动配对，文件名中已有 `检测结果` 的图片不会作为原图参与评测。当前默认采用召回优先口径，`IoU >= 0.3` 即视为标准答案框被命中，每张图最多统计 8 个检测框。

```bash
python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8
```

标准答案图中优先抽取红色标注框；当标准答案图和原图尺寸不一致时，会先用 ORB 特征匹配和仿射变换自动对齐，再结合与红框重叠的黄色或橙色外层标注稳定框范围。对齐不足 50 个匹配点、RANSAC 内点比例低于 `0.55` 或缩放比例不在 `0.5-2.0` 时会直接报错，避免静默跳过问题样例。

评测报告包含每张图的标准框、检测框、命中情况、召回率、误报数和平均最佳 IoU。当前 `data/` 中只要存在对应 `xxx检测结果.png`，就会自动进入量化统计，因此 `400.png` 这类新增短信变体也会自动进入硬回归。

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

当前默认输出采用“召回不降 + 报告过滤”的平衡策略：内部仍保留完整候选池用于调试，但最终 `detections` 和 `top_candidates` 只展示通过报告级过滤的高可信候选，尽量避免把正确文字直接画框或写进报告。

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
