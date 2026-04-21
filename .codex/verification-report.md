# 验证报告

生成时间：2026-04-20 09:53:30 CST
执行者：Codex

## 需求审查清单

- 目标：修复 `main.py` 因 `skimage` / `numpy` 二进制不兼容导致的启动失败
- 范围：仅调整本地检测实现与依赖声明，不改变 CLI 参数和输出协议
- 交付物：代码修复、依赖清理、上下文摘要、操作日志、验证报告
- 审查要点：入口可运行、现有测试通过、检测结果主流程不回退
- 依赖与风险：已评估 `skimage` 移除对特征数值的影响，使用现有样例回归验证
- 结论留痕：本报告和 `.codex/operations-log.md` 已记录时间戳与验证结果

## 技术维度评分

- 代码质量：95/100
  - 原因：用纯 `opencv/numpy` 替代脆弱的二进制依赖，保持了原有特征接口和输出维度
- 测试覆盖：93/100
  - 原因：现有 `unittest` 6 项回归全部通过，覆盖主样例、CLI、异常路径与空白图回退
- 规范遵循：91/100
  - 原因：中文说明、日志与 `.codex` 记录已补齐；部分仓库规范提到的专用工具在当前环境不可用，已做偏差说明

## 战略维度评分

- 需求匹配：97/100
  - 原因：直接消除了当前启动阻塞点，用户无需先处理环境 ABI
- 架构一致：94/100
  - 原因：继续沿用项目现有的传统视觉路线，没有改动上层 API 和测试组织
- 风险评估：90/100
  - 原因：主要风险是自实现 LBP 与骨架细化的数值细节差异；现有回归样例未发现行为回退

## 综合评分

- 综合评分：94/100
- 建议：通过

## 本地验证结果

1. 执行：`/opt/anaconda3/bin/python -m unittest test_detector.py`
   结果：通过，`Ran 6 tests in 14.974s`
2. 执行：`/opt/anaconda3/bin/python main.py --image ./截图.png --output ./tmp_result.png --report ./tmp_report.json`
   结果：通过，成功生成结果图与 JSON 报告，状态为 `detected`

## 可重复验证步骤

1. 执行 `pip install -r requirements.txt`
2. 执行 `/opt/anaconda3/bin/python -m unittest test_detector.py`
3. 执行 `/opt/anaconda3/bin/python main.py --image ./截图.png --output ./detected_result.png --report ./detected_report.json`

## 风险与补偿计划

- 当前未额外新增针对自实现 LBP / 骨架细化的单元测试
- 补偿建议：后续可为 `_uniform_lbp` 与 `_skeletonize_mask` 增加维度和基础统计稳定性测试，降低未来回归成本

---

## 增量验证 - 400短信泛化改进

生成时间：2026-04-21 09:39:20 CST

### 需求审查清单

- 目标：让 `400.png` 这类短信变体也能稳定命中，同时不破坏截图、票据、身份证等旧场景
- 范围：融合层短信细化候选、报告级过滤、测试门槛和文档
- 交付物：代码增强、测试更新、文档更新、独立验证报告
- 审查要点：`400.png` 双命中、总召回达标、误报继续受控
- 依赖与风险：沿用现有 OpenCV/NumPy/Pillow 方案，不新增模型依赖

### 评分

- 代码质量：94/100
- 测试覆盖：96/100
- 规范遵循：90/100
- 需求匹配：97/100
- 架构一致：95/100
- 风险评估：91/100
- 综合评分：94/100
- 建议：通过

### 本地验证结果

1. `/opt/anaconda3/bin/python -m unittest test_detector.py`
   - 结果：通过，`Ran 16 tests in 134.358s`
2. `/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
   - 结果：通过，`recall=0.9286`、`false_positive_count=0`

### 结论

- `400.png` 已纳入硬回归，并可逐处输出两个真实篡改位置
- `截图.png` 的 `time_group` 和 `票据.png` 的 `digit_window` 主框优先级已恢复

---

## 增量验证 - 单阈值最终输出控制

生成时间：2026-04-21 10:31:15 CST

### 需求审查清单

- 目标：让用户主要只调 `REPORT_CONFIDENCE_THRESHOLD` 就能控制最终输出框数量
- 范围：融合层最终候选筛选、`main.py` 默认调参区、测试和文档
- 交付物：单阈值模式实现、回归测试、README 和项目梳理说明、操作日志
- 审查要点：低阈值不再被类型限额压成一两个框，高阈值能收紧输出，默认评测不回退

### 评分

- 代码质量：93/100
- 测试覆盖：95/100
- 规范遵循：90/100
- 需求匹配：96/100
- 架构一致：92/100
- 风险评估：90/100
- 综合评分：93/100
- 建议：通过

### 本地验证结果

1. `/opt/anaconda3/bin/python -m unittest test_detector.py`
   - 结果：通过，`Ran 19 tests in 139.099s`
2. `/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
   - 结果：通过，`recall=0.9286`
3. `data/400.png` 阈值阶梯：
   - `0 -> 11` 个框
   - `50 -> 9` 个框
   - `60 -> 3` 个框
   - `68 -> 3` 个框
   - `80 -> 3` 个框
   - `95 -> 2` 个框
   - `999 -> 0` 个框

### 结论

- `REPORT_CONFIDENCE_THRESHOLD` 现在在显式调参路径中是最终输出数量的主控制项
- 默认 API 未显式传入该参数时继续使用原报告过滤，避免评测误报回升

---

## 增量验证 - 高阈值保真修复

生成时间：2026-04-21 11:39:32 CST

### 需求审查清单

- 目标：确保高阈值过滤后优先留下正确主候选，而不是错误的高分候选
- 范围：单阈值模式下的最终排序与阈值分数、回归测试、文档说明
- 交付物：`detector.py` 排序修复、高阈值专项测试、文档和日志
- 审查要点：`截图 -> time_group`、`票据 -> digit_window`、`300/400 -> text_noise_anomaly`，默认评测不回退

### 评分

- 代码质量：94/100
- 测试覆盖：96/100
- 规范遵循：90/100
- 需求匹配：97/100
- 架构一致：93/100
- 风险评估：91/100
- 综合评分：94/100
- 建议：通过

### 本地验证结果

1. `/opt/anaconda3/bin/python -m unittest test_detector.UniversalTamperDetectorTest.test_high_report_threshold_keeps_ground_truth_candidates`
   - 结果：通过
2. `/opt/anaconda3/bin/python -m unittest test_detector.py`
   - 结果：通过，`Ran 20 tests in 181.595s`
3. `/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
   - 结果：通过，`recall=0.9286`
4. 高阈值样例输出：
   - `截图.png @95 -> ['time_group']`
   - `票据.png @95 -> ['digit_window']`
   - `300.png @95 -> ['text_noise_anomaly']`
   - `400.png @95 -> ['text_noise_anomaly', 'text_noise_anomaly']`

### 结论

- 高阈值过滤现在优先留下正确主候选，而不再让碎片热点、冲突噪声框或错误时间框抢位
- 单阈值模式和默认严格模式已同时通过回归

---

## 增量验证 - REPORT_CONFIDENCE_THRESHOLD 回归测试与文档

生成时间：2026-04-21 10:20:00 CST

### 需求审查清单

- 目标：围绕“单阈值控制最终框数量”补回归测试，并更新 README/项目梳理中的新语义。
- 范围：仅修改 `test_detector.py`、`README.md`、`项目梳理.md` 和 `.codex/` 本轮记录文件。
- 交付物：新增合成候选回归测试、文档语义说明、上下文摘要、操作日志和本验证报告。
- 审查要点：阈值调高减少输出；阈值调低增加或不减少输出；低阈值不被隐藏类型限额压成固定一两个框。
- 依赖与风险：`detector.py` 不在本轮允许修改范围内，新增测试当前暴露生产链路未满足新语义。

### 评分

- 代码质量：88/100
- 测试覆盖：92/100
- 规范遵循：86/100
- 需求匹配：90/100
- 架构一致：84/100
- 风险评估：70/100
- 综合评分：78/100
- 建议：退回生产实现修复

### 本地验证结果

1. `/opt/anaconda3/bin/python -m unittest test_detector.UniversalTamperDetectorTest.test_report_threshold_lowering_increases_reportable_box_count test_detector.UniversalTamperDetectorTest.test_low_report_threshold_is_limited_by_max_items_not_type_caps test_detector.UniversalTamperDetectorTest.test_single_report_threshold_can_reduce_output_boxes`
   - 结果：失败，`Ran 3 tests in 9.360s`，2 项新增回归失败。
   - `test_report_threshold_lowering_increases_reportable_box_count`：低阈值期望 5 个候选，实际 2 个。
   - `test_low_report_threshold_is_limited_by_max_items_not_type_caps`：`max_items=3` 期望 3 个候选，实际 2 个。
2. `/opt/anaconda3/bin/python -m py_compile test_detector.py`
   - 结果：通过。

### 结论

- 测试和文档已补齐用户要求的新语义。
- 当前生产行为仍不满足该语义，需后续在 `detector.py` 修复 `_report_limit` 等隐藏数量限制后再让新增回归通过。

---

## 增量验证 - 高阈值保真回归测试与文档

生成时间：2026-04-21 11:24:00 CST

### 需求审查清单

- 目标：补充高阈值质量回归，确保高阈值过滤后保留下来的是正确候选，而不是错误候选。
- 范围：仅修改 `test_detector.py`、`README.md`、`项目梳理.md` 和 `.codex/` 本轮记录文件。
- 交付物：新增真实样例高阈值保真测试、README 高阈值语义说明、项目梳理高阈值验收口径、上下文摘要、操作日志和验证报告。
- 审查要点：截图保留命中标准答案的 `time_group`；票据保留命中标准答案的 `digit_window`；`300.png` 和 `400.png` 保留命中标准答案的 `text_noise_anomaly`；高阈值留下的可见框不应与标准答案完全无重叠。
- 依赖与风险：`detector.py` 不在本轮允许修改范围内，新增测试会暴露当前生产筛选逻辑仍未满足高阈值保真语义。

### 评分

- 代码质量：90/100
- 测试覆盖：94/100
- 规范遵循：88/100
- 需求匹配：96/100
- 架构一致：86/100
- 风险评估：72/100
- 综合评分：79/100
- 建议：退回生产实现修复

### 本地验证结果

1. `/opt/anaconda3/bin/python -m py_compile test_detector.py`
   - 结果：通过。
2. `/opt/anaconda3/bin/python -m unittest test_detector.UniversalTamperDetectorTest.test_high_report_threshold_keeps_ground_truth_candidates`
   - 结果：失败，`Ran 1 test in 10.445s`，4 个子样例失败。
   - `截图.png`：保留了与标准答案无重叠的 `text_noise_anomaly` 和 `region_anomaly` 错误候选。
   - `票据.png`：除正确 `digit_window` 外，还保留了多个与标准答案无重叠的 `digit_window` 错误候选。
   - `300.png`：保留了与标准答案无重叠的 `time_group` 错误候选。
   - `400.png`：除命中标准答案的 `text_noise_anomaly` 外，还保留了一个与标准答案无重叠的 `text_noise_anomaly` 错误候选。

### 结论

- 本轮测试和文档已覆盖用户指定的高阈值保真语义。
- 当前生产行为未通过新增回归，说明显式高阈值路径仍偏向原始报告置信度排序，未充分抑制高置信错误候选。
- 后续需要在允许修改 `detector.py` 时调整高阈值路径的质量约束，例如把专用场景候选保护、标准候选合法性和错误候选抑制纳入 `single_threshold_mode`，同时保持数量单调性。
