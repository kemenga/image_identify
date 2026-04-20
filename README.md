# 文档优先的通用图像篡改检测

日期：2026-04-15 09:33（UTC+8）
执行者：Codex

本目录提供一个独立的本地 Python 检测工具，首版优先覆盖短信截图、票据、身份证/证件等文档型图片。实现采用传统计算机视觉方法：以文档规则定位为主，辅以 ELA 和噪声不一致性证据，不包含训练流程或外部模型下载。

如果你想先快速了解仓库在做什么、各文件职责怎么分，可以直接看 [项目梳理.md](./项目梳理.md)。

当前版本还会比较同一张图片中不同文字组件的噪声一致性。当少量文字的高频噪声、拉普拉斯响应或背景残差明显偏离其余文字时，会生成“噪声篡改”候选，用于补强局部文字改动场景。

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
```

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

## 输出说明

- 结果图只绘制最终筛选出的检测框，适合人工验收。
- 辅助证据热力图仅在传入 `--evidence-output-dir` 时生成，用于观察检测依据。
- JSON 报告固定包含：
  - `status`
  - `reason`
  - `detections`
  - `candidate_count`
  - `top_candidates`
  - `evidence`
  - `evidence_artifacts`

`evidence` 中默认保留三类全局证据：

- `ela_score`
- `noise_score`
- `document_score`

每个检测框的 `detail.evidence` 中还会保留局部 ELA、局部噪声以及全局证据分，便于回看候选为什么被保留。

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
