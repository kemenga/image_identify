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
