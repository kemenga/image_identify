## 验证报告（REPORT_CONFIDENCE_THRESHOLD 回归测试与文档）

生成时间：2026-04-21 10:20:00 CST

### 需求审查清单

- **目标**：围绕“单阈值控制最终框数量”补充回归测试，并更新文档说明 `REPORT_CONFIDENCE_THRESHOLD` 的新语义。
- **范围**：仅修改 `test_detector.py`、`README.md`、`项目梳理.md` 和 `.codex/` 本轮记录文件。
- **交付物**：新增阈值单调性测试、补充 CLI/README/项目梳理说明、上下文摘要、操作日志和验证报告。
- **审查要点**：阈值调高会减少输出框；阈值调低会增加或至少不减少输出框；低阈值不应被隐藏类型限额压成固定一两个框。
- **依赖与风险**：当前 `detector.py` 不在本轮允许修改范围内；新增测试会暴露现有 `_report_limit` 仍限制低阈值输出的问题。
- **结论留痕**：本报告和 `.codex/operations-log.md` 已记录本地验证结果。

### 技术维度评分

- **代码质量**：88/100
  - 原因：测试使用合成 `TamperRegion` 直接覆盖报告筛选链路，定位准确；但生产实现尚未修复，新增测试当前失败。
- **测试覆盖**：92/100
  - 原因：覆盖了阈值调高减少输出、阈值调低增加输出、低阈值只受显式上限约束三类关键语义。
- **规范遵循**：86/100
  - 原因：限制在用户指定文件范围内补测试和文档；当前会话缺少项目要求的专用工具，已记录替代方案。

### 战略维度评分

- **需求匹配**：90/100
  - 原因：测试和文档准确表达新语义，能够复现用户描述的隐藏限额问题。
- **架构一致**：84/100
  - 原因：未引入新依赖，沿用现有 `unittest` 和报告筛选函数；但生产实现仍与文档语义不一致。
- **风险评估**：70/100
  - 原因：新增回归当前失败，说明合入前必须修复 `detector.py` 的报告筛选链路，否则文档与行为不一致。

### 综合评分

- **综合评分**：78/100
- **建议**：退回生产实现修复

### 本地验证结果

1. 执行：`/opt/anaconda3/bin/python -m unittest test_detector.UniversalTamperDetectorTest.test_report_threshold_lowering_increases_reportable_box_count test_detector.UniversalTamperDetectorTest.test_low_report_threshold_is_limited_by_max_items_not_type_caps test_detector.UniversalTamperDetectorTest.test_single_report_threshold_can_reduce_output_boxes`
   - 结果：失败，`Ran 3 tests in 9.360s`，其中 2 项新增回归失败。
   - 失败1：`test_report_threshold_lowering_increases_reportable_box_count` 期望低阈值返回 5 个候选，实际返回 2 个。
   - 失败2：`test_low_report_threshold_is_limited_by_max_items_not_type_caps` 期望 `max_items=3` 时返回 3 个候选，实际返回 2 个。
2. 执行：`/opt/anaconda3/bin/python -m py_compile test_detector.py`
   - 结果：通过，测试文件语法有效。

### 可重复验证步骤

1. 执行 `/opt/anaconda3/bin/python -m unittest test_detector.UniversalTamperDetectorTest.test_report_threshold_lowering_increases_reportable_box_count test_detector.UniversalTamperDetectorTest.test_low_report_threshold_is_limited_by_max_items_not_type_caps test_detector.UniversalTamperDetectorTest.test_single_report_threshold_can_reduce_output_boxes`
2. 执行 `/opt/anaconda3/bin/python -m py_compile test_detector.py`

### 风险与补偿计划

- 风险：当前生产链路仍会在 `REPORT_CONFIDENCE_THRESHOLD=0.0` 时通过 `_report_limit` 等隐藏规则压缩输出数量，导致用户看到的框仍可能只有一两个。
- 补偿计划：下一步需要在 `detector.py` 中调整报告筛选责任边界，让 `_is_reportable_candidate` 只判断候选合法性、去重只合并同一物理区域、`max_detections` 只作为显式上限，数量收缩统一由 `REPORT_CONFIDENCE_THRESHOLD` 控制。
- 附加说明：本轮按用户要求未修改 `detector.py`，因此未尝试修复生产行为。
