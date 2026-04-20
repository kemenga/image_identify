# 验证报告

生成时间：2026-04-20 15:04:55 CST
执行者：Codex

## 需求审查清单

- 目标：根据用户提示，比较少量文字和其余文字的噪声差异，提高局部文字篡改识别能力
- 范围：新增文档检测候选类型、测试和说明文档，不改动对外 API
- 交付物：`text_noise_anomaly` 候选逻辑、测试用例、README 更新、项目梳理更新、验证报告
- 审查要点：新候选能在 `data/` 样例中产生，原有截图/票据/身份证回归不失败
- 依赖与风险：不新增依赖，复用 OpenCV 和 NumPy
- 结论留痕：本报告和 `.codex/operations-log.md` 已记录时间戳与验证结果

## 技术维度评分

- 代码质量：92/100
  - 原因：新增逻辑复用现有候选框架，独立成函数组，阈值和排序接入点清晰
- 测试覆盖：91/100
  - 原因：新增数据样例噪声候选测试，并保留原有 8 项回归
- 规范遵循：92/100
  - 原因：中文注释、中文文档、`.codex` 记录均已补齐；专用 MCP 工具当前环境不可用，已通过本地等效命令完成

## 战略维度评分

- 需求匹配：94/100
  - 原因：实现了“少量文字噪声 vs 其余文字噪声”的全图与同行对比
- 架构一致：93/100
  - 原因：新能力接入文档规则层，不影响上层 API 和报告协议
- 风险评估：88/100
  - 原因：噪声一致性对扫描质量和压缩差异敏感，已用保守过滤和排序权重降低误报，但仍需更多标注样例继续调参

## 综合评分

- 综合评分：92/100
- 建议：通过

## 本地验证结果

1. 执行：`/opt/anaconda3/bin/python -m unittest test_detector.py`
   结果：通过，`Ran 9 tests in 46.121s`
2. 执行：`/opt/anaconda3/bin/python main.py --image ./data/发票.png --output ./tmp_noise_result.png --report ./tmp_noise_report.json --evidence-output-dir ./tmp_noise_maps`
   结果：通过，第一检测项为“噪声篡改”，检测框为 `(753, 422, 194, 53)`
3. 执行：遍历 `data/*.png` 的候选摘要
   结果：300、发票、截图、票据、身份证样例均产生了 `text_noise_anomaly` 候选，原有主规则未整体回退

## 可重复验证步骤

1. 执行 `/opt/anaconda3/bin/python -m unittest test_detector.py`
2. 执行 `/opt/anaconda3/bin/python main.py --image ./data/发票.png --output ./detected_result.png --report ./detected_report.json --evidence-output-dir ./evidence_maps`
3. 检查终端输出中是否包含“噪声篡改”
4. 检查 `detected_report.json` 的 `detections` 和 `top_candidates` 中是否包含 `text_noise_anomaly`

## 风险与补偿计划

- 当前没有人工标注的标准答案集合，不能给出真实准确率指标
- 后续建议把 `data/` 中每张图的期望篡改框整理成 JSON 标注，再增加 IoU 指标测试
- 如果误报集中在标点或表格线，可继续提高单组件面积下限或要求同行支持数量
