# 验证报告 - 新增样本漏检修复

生成时间：2026-04-21 15:02:08 CST
执行者：Codex

## 需求审查清单

- 目标：修复 `image.png`、`image4.png`、`image5.png` 的候选池漏检，并保持旧 6 张样例不退化
- 范围：`detector.py` 候选生成与排序、`test_detector.py` 回归断言、文档与 `.codex` 留痕
- 交付物：代码修复、测试加固、文档更新、操作日志和验证报告
- 审查要点：新样本逐框命中、`Recall@IoU0.3 >= 0.95`、总误报数 `<= 6`
- 依赖与风险：继续使用 OpenCV/NumPy/Pillow，不引入 OCR、模型或新依赖
- 结论留痕：`evaluation_report.json`、`.codex/operations-log.md` 与本报告均已回填

## 技术维度评分

- 代码质量：93/100
  - 原因：沿用现有融合层与报告过滤框架，只在候选池和排序条件上做中等重构，改动面集中且职责清晰
- 测试覆盖：96/100
  - 原因：完整 `22` 项 `unittest` 通过，并新增了 `image.png`、`image4.png`、`image5.png` 的显式回归和候选池覆盖断言
- 规范遵循：89/100
  - 原因：中文文档、`.codex` 留痕和本地验证已补齐；仓库规范点到的外部工具当前环境未提供，已在上下文摘要中做偏差说明

## 战略维度评分

- 需求匹配：97/100
  - 原因：`image.png`、`image4.png`、`image5.png` 已全部逐框命中，旧样例主框顺序和 CLI/API 行为保持不变
- 架构一致：95/100
  - 原因：继续坚持“文档规则 + 证据融合 + 报告过滤”的既有路线，没有推翻重写或引入新依赖
- 风险评估：91/100
  - 原因：当前 9 图基线已稳定通过，但 `身份证.png` 仍只有 `recall=0.8`，后续新增样本时仍需继续以数据集评测守住回归

## 综合评分

- 综合评分：94/100
- 建议：通过

## 本地验证结果

1. 执行：`/opt/anaconda3/bin/python -m unittest test_detector.py`
   结果：通过，`Ran 22 tests in 340.345s`
2. 执行：`/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
   结果：通过，`image_count=9`、`ground_truth_count=27`、`hit_count=26`、`recall=0.9630`、`false_positive_count=2`

## 关键样例结果

- `image.png`：`gt=7`、`det=8`、`recall=1.0`、`fp=1`、`average_best_iou=0.4895`
- `image4.png`：`gt=3`、`det=3`、`recall=1.0`、`fp=0`、`average_best_iou=0.4950`
- `image5.png`：`gt=3`、`det=3`、`recall=1.0`、`fp=0`、`average_best_iou=0.6115`
- `截图.png`：第一主框保持 `time_group`
- `票据.png`：第一主框保持 `digit_window`
- `300.png` / `400.png`：继续保持 `text_noise_anomaly` 主导

## 可重复验证步骤

1. 执行 `/opt/anaconda3/bin/python -m unittest test_detector.py`
2. 执行 `/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
3. 如需人工检查结果图，执行 `/opt/anaconda3/bin/python main.py --image ./data/image5.png --output ./detected_result.png --report ./detected_report.json`

## 风险与补偿计划

- `身份证.png` 当前 `recall=0.8`，说明旧样例里仍有一个标准框未被最终结果命中；后续再扩样本时应继续以全量评测和专项断言同步跟进
- 当前 `image.png` 与 `发票.png` 各有 `1` 个误报框，后续如需继续压误报，优先调 `invoice_token_patch` 的报告排序与限流，而不是直接移除候选生成入口
