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

## 增量验证 - 新增样本漏检修复

生成时间：2026-04-21 15:42:00 CST

### 需求审查清单

- 目标：修复 `image.png`、`image4.png`、`image5.png` 的漏检，并保持旧样例稳定
- 范围：`detector.py`、`test_detector.py`、`README.md`、`项目梳理.md`、`.codex/` 记录
- 交付物：候选生成改造、回归测试、文档更新、验证报告
- 审查要点：9 图评测 `Recall@IoU0.3 >= 0.95`，总误报数 `<= 6`，新样例逐框命中
- 依赖与风险：继续使用 OpenCV/NumPy/Pillow，不引入 OCR、模型或新依赖

### 评分

- 代码质量：93/100
- 测试覆盖：96/100
- 规范遵循：89/100
- 需求匹配：97/100
- 架构一致：95/100
- 风险评估：91/100
- 综合评分：94/100
- 建议：通过

### 本地验证结果

1. `/opt/anaconda3/bin/python -m unittest test_detector.py`
   - 结果：通过，`Ran 22 tests in 340.345s`
2. `/opt/anaconda3/bin/python evaluation.py --data-dir ./data --report ./evaluation_report.json --iou-threshold 0.3 --max-detections 8`
   - 结果：通过，`recall=0.9630`、`false_positive_count=2`

### 结论

- `image.png`、`image4.png`、`image5.png` 已全部进入硬回归并达到 `recall=1.0`
- 旧样例主框顺序保持稳定：截图仍为 `time_group`，票据仍为 `digit_window`
- 当前仍建议后续继续观察 `身份证.png` 的剩余漏框和 `invoice_token_patch` 的少量误报

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

---

## 增量验证 - 开放许可数据集生成

生成时间：2026-04-21 16:34:00 CST

### 需求审查清单

- 目标：新增一条独立的数据生成管线，自动搜索开放许可原图，再本地生成篡改图与标签，不下载任何已篡改图片。
- 范围：新增 `dataset_builder.py`、新增 `test_dataset_builder.py`、扩展 `evaluation.py` 的样本发现逻辑，并更新 README、项目梳理与 `.codex` 留痕。
- 交付物：开放许可搜索/下载/篡改/打标/校验脚本，离线单测，`seed_dataset/` 目录兼容，以及真实联网烟雾验证结果。
- 审查要点：许可过滤正确；下载后尺寸与去重有效；篡改图与标签同尺寸；`evaluation.py --data-dir ./seed_dataset` 的内部配对逻辑可用；旧检测功能不回退。
- 依赖与风险：外部素材搜索依赖 Openverse 与 Wikimedia Commons；真实高分辨率样本评测耗时较长，不适合作为每轮阻塞式验收。

### 评分

- 代码质量：92/100
- 测试覆盖：93/100
- 规范遵循：89/100
- 需求匹配：95/100
- 架构一致：91/100
- 风险评估：88/100
- 综合评分：91/100
- 建议：通过

### 本地验证结果

1. `python -m py_compile dataset_builder.py test_dataset_builder.py evaluation.py`
   - 结果：通过。
2. `python -m unittest test_dataset_builder.py`
   - 结果：通过，`Ran 3 tests`。
   - 覆盖点：搜索 manifest 结构、下载去重、篡改/打标/校验链路、`seed_dataset/` 评测兼容。
3. `python -m unittest test_detector.py test_dataset_builder.py`
   - 结果：通过，`Ran 25 tests in 397.394s`。
   - 说明：新增数据管线未破坏现有检测器回归。
4. 真实联网烟雾：
   - `python dataset_builder.py --dataset-root ./seed_dataset search --docs-target 2 --natural-target 2 --page-size 8 --max-pages 2 --oversample-factor 5`
   - `python dataset_builder.py --dataset-root ./seed_dataset download --docs-target 2 --natural-target 2`
   - `python dataset_builder.py --dataset-root ./seed_dataset tamper --seed 42`
   - `python dataset_builder.py --dataset-root ./seed_dataset label`
   - `python dataset_builder.py --dataset-root ./seed_dataset verify`
   - 结果：通过，生成 20 个候选、4 张原图、12 张篡改图、12 张标签图；`verify_report.json` 显示 `passed=True`、`pair_count=12`、`extracted_ground_truth_box_count=12`。
5. `python evaluation.py --data-dir ./seed_dataset --report ./seed_dataset/manifests/evaluation_smoke_report.json --iou-threshold 0.3 --max-detections 2`
   - 结果：可启动，但在 12 张高分辨率真实图上运行时间过长，本轮为避免阻塞被中断。
   - 补偿证据：离线单测已实际调用 `evaluate_dataset(...)` 验证 `seed_dataset/` 结构兼容；`verify` 也已对全部真实配对执行 `extract_ground_truth_boxes(...)` 反向抽框。

### 战略评估

- 需求匹配：脚本能力与目录结构完全按计划落地，且未污染现有 `data/` 回归集。
- 架构一致：新能力独立于主检测入口，复用了已有预处理和评测逻辑，没有引入新依赖。
- 风险评估：外部接口波动和高分辨率真实图评测耗时仍需关注，但不影响当前“可生成、可标注、可校验”的主目标。

### 结论

- 本轮实现已满足“开放许可原图搜索与本地篡改数据集生成”主需求。
- `dataset_builder.py` 已可用于继续扩到默认 20 张原图 / 60 张篡改图规模。
- 当前剩余风险主要是：如果后续要把真实高分辨率全量评测纳入每轮阻塞验收，建议再加一层评测前缩图或分批运行策略。

---

## 增量验证 - 大图扫描性能

生成时间：2026-04-21 17:22:00 CST

### 需求审查清单

- 目标：修复无参数运行 `main.py` 时在高分辨率 `seed_dataset` 图片上长时间卡在 `region_anomaly` 全图滑窗的问题。
- 范围：修改 `document_detector.py` 的密集滑窗扫描策略，新增测试；不改变 `UniversalTamperDetector.detect(...)` 和 CLI 公开参数。
- 交付物：大图自动降采样、扫描窗口预算、可调参数接入、回归测试和验证记录。
- 审查要点：大图检测能快速返回；旧样例检测不退化；新增参数可调；无参数运行入口可正常结束。
- 依赖与风险：超大图密集扫描降采样会牺牲极小像素异常的敏感度，但换来可交互的运行时间。

### 评分

- 代码质量：91/100
- 测试覆盖：92/100
- 规范遵循：90/100
- 需求匹配：96/100
- 架构一致：92/100
- 风险评估：88/100
- 综合评分：92/100
- 建议：通过

### 本地验证结果

1. `python -m py_compile main.py document_detector.py test_detector.py`
   - 结果：通过。
2. `python -m unittest test_detector.UniversalTamperDetectorTest.test_detector_accepts_tunable_overrides test_detector.UniversalTamperDetectorTest.test_large_image_dense_scan_is_downscaled_and_budgeted test_detector.UniversalTamperDetectorTest.test_cli_writes_output_and_report`
   - 结果：通过，`Ran 3 tests in 3.501s`。
3. 真实大图冒烟：
   - 命令：直接调用检测器处理 `seed_dataset/tampered/doc_0001_t01_text_patch_replace.png`
   - 结果：约 `7.364s` 返回，状态 `detected`，检测框数量 `8`。
4. `python -m unittest test_detector.py test_dataset_builder.py`
   - 结果：通过，`Ran 26 tests in 278.069s`。
5. `/opt/anaconda3/bin/python /Users/helen/Desktop/work/claude_to_codex/image_deep_learning_codex/main.py`
   - 结果：正常结束，生成 `detected_result.png` 和 `detected_report.json`，不再需要手动 `Ctrl+C`。

### 结论

- 本轮已经修复高分辨率图片导致全图滑窗过慢的问题。
- 新增文档规则层参数已自动进入 `main.py` 调参体系：
  - `GLOBAL_SCAN_MAX_SIDE`
  - `GLOBAL_SCAN_MAX_WINDOWS`
  - `GLOBAL_SCAN_MIN_WINDOW`
  - `GLOBAL_SCAN_MIN_STRIDE`
- 后续如果要追求更高召回，可以适当调高 `GLOBAL_SCAN_MAX_SIDE` 或 `GLOBAL_SCAN_MAX_WINDOWS`；如果要更快，可以调低这两个参数。
