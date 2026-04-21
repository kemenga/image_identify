# 验证报告 - 400短信泛化改进

生成时间：2026-04-21 09:39:20 CST
执行者：Codex

## 需求审查清单

- 目标：提升 `400.png` 这类短信变体的泛化能力，同时保持旧场景稳定
- 范围：融合层短信噪声细化逻辑、报告级过滤、测试与文档
- 交付物：代码修复、测试加固、文档更新、日志与验证报告
- 审查要点：`400.png` 逐处命中、`截图.png` 与 `票据.png` 专用主框不回退、总体召回与误报达标
- 依赖与风险：继续沿用 OpenCV/NumPy/Pillow，不新增模型和第三方依赖
- 结论留痕：本报告、`.codex/operations-log.md` 与 `evaluation_report.json` 已记录结果

## 技术维度评分

- 代码质量：94/100
  - 原因：沿用现有融合层候选和报告过滤框架，只在短信类细化与场景冲突抑制处增强，修改面集中
- 测试覆盖：96/100
  - 原因：`16` 项 `unittest` 全部通过，并补强了 `400.png` 的显式回归和更严格的数据集门槛
- 规范遵循：90/100
  - 原因：中文文档、日志和 `.codex` 留痕已补齐；部分仓库规范点到的外部工具当前环境未提供，已做偏差说明

## 战略维度评分

- 需求匹配：97/100
  - 原因：`400.png` 两处真实改动已稳定命中，且旧样例的时间框和数字框优先级已恢复
- 架构一致：95/100
  - 原因：仍然保持“文档规则 + 证据融合 + 报告过滤”的既有路线，没有引入新依赖或破坏接口
- 风险评估：91/100
  - 原因：当前数据集通过，但短信类阈值仍有继续调参空间，新增样例时仍需以评测回归约束

## 综合评分

- 综合评分：94/100
- 建议：通过

## 本地验证结果

1. 执行：`/opt/anaconda3/bin/python -m unittest test_detector.py`
   结果：通过，`Ran 16 tests in 134.358s`
2. 执行：`/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
   结果：通过，`image_count=6`、`ground_truth_count=14`、`hit_count=13`、`recall=0.9286`、`false_positive_count=0`
3. 关键样例：
   - `400.png`：2 个标准框全部命中，最终输出 2 个 `text_noise_anomaly`
   - `截图.png`：第一主框恢复为 `time_group`
   - `票据.png`：第一主框保持为 `digit_window`

## 可重复验证步骤

1. 执行 `/opt/anaconda3/bin/python -m unittest test_detector.py`
2. 执行 `/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
3. 如需人工查看 `400.png`，执行 `/opt/anaconda3/bin/python main.py --image ./data/400.png --output ./detected_result.png --report ./detected_report.json`

## 风险与补偿计划

- 当前短信类泛化主要依赖文字组件噪声一致性和报告过滤，尚未覆盖更多排版风格差异极大的真实样本
- 补偿建议：后续继续把新增短信变体以 `xxx检测结果.png` 方式补进 `data/`，直接纳入 `evaluation.py` 的自动回归
