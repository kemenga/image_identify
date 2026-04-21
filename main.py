from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector import UniversalTamperDetector, build_report


BASE_DIR = Path(__file__).resolve().parent
BASE_DETECTOR_DEFAULTS = UniversalTamperDetector.tunable_defaults()
BASE_DOCUMENT_DEFAULTS = UniversalTamperDetector.document_tunable_defaults()

# 直接手动调参区
# 如果你更习惯直接改文件而不是传命令行参数，优先改这里。
RUN_CONFIG = {
    "image": str(BASE_DIR / "data/300.png"),
    "output": str(BASE_DIR / "detected_result.png"),
    "report": str(BASE_DIR / "detected_report.json"),
    "max_detections": 5,
    "evidence_output_dir": None,
}

# 融合层参数。
# 可直接修改数值，例如：
# "GLOBAL_EVIDENCE_WEIGHT": 1.05
DETECTOR_PARAM_DEFAULTS = dict(BASE_DETECTOR_DEFAULTS)

# 文档规则层参数。
# 可直接修改数值，例如：
# "TEXT_NOISE_THRESHOLD": 5.1
# "METHOD_WEIGHT_STROKE": 1.4
DOCUMENT_PARAM_DEFAULTS = dict(BASE_DOCUMENT_DEFAULTS)

DETECTOR_PARAM_DEFAULTS["GLOBAL_EVIDENCE_WEIGHT"] = 1.05
DOCUMENT_PARAM_DEFAULTS["TEXT_NOISE_THRESHOLD"] = 5
DOCUMENT_PARAM_DEFAULTS["METHOD_WEIGHT_STROKE"] = 1.4


def _cli_flag(prefix: str, key: str) -> str:
    return f"--{prefix.replace('_', '-')}{key.lower().replace('_', '-')}"


def _argparse_type(default_value: object):
    if isinstance(default_value, bool):
        return str
    if isinstance(default_value, int):
        return int
    if isinstance(default_value, float):
        return float
    return str


def _add_tunable_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    defaults: dict[str, object],
    group_title: str,
) -> None:
    group = parser.add_argument_group(group_title)
    for key, default_value in defaults.items():
        group.add_argument(
            _cli_flag(prefix, key),
            dest=f"{prefix}{key}",
            type=_argparse_type(default_value),
            default=None,
            help=f"覆盖默认参数 {key}，当前默认值: {default_value}",
        )


def _collect_overrides(
    args: argparse.Namespace,
    prefix: str,
    defaults: dict[str, object],
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key in defaults:
        value = getattr(args, f"{prefix}{key}")
        if value is not None:
            overrides[key] = value
    return overrides


def _changed_values(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    changed: dict[str, object] = {}
    for key, value in current.items():
        if baseline.get(key) != value:
            changed[key] = value
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文档优先的通用图像篡改检测")
    parser.add_argument(
        "--image",
        default=str(RUN_CONFIG["image"]),
        help="输入图片路径",
    )
    parser.add_argument(
        "--output",
        default=str(RUN_CONFIG["output"]),
        help="输出高亮图片路径",
    )
    parser.add_argument(
        "--report",
        default=str(RUN_CONFIG["report"]),
        help="输出 JSON 报告路径",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=int(RUN_CONFIG["max_detections"]),
        help="最多输出的检测框数量",
    )
    parser.add_argument(
        "--evidence-output-dir",
        default=RUN_CONFIG["evidence_output_dir"],
        help="辅助证据热力图输出目录；传入后生成 ELA、噪声和融合热力图",
    )
    _add_tunable_args(
        parser=parser,
        prefix="detector_",
        defaults=DETECTOR_PARAM_DEFAULTS,
        group_title="融合层可调参数",
    )
    _add_tunable_args(
        parser=parser,
        prefix="document_",
        defaults=DOCUMENT_PARAM_DEFAULTS,
        group_title="文档规则层可调参数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_overrides = dict(DETECTOR_PARAM_DEFAULTS)
    detector_overrides.update(
        _collect_overrides(
            args=args,
            prefix="detector_",
            defaults=DETECTOR_PARAM_DEFAULTS,
        )
    )
    document_detector_overrides = dict(DOCUMENT_PARAM_DEFAULTS)
    document_detector_overrides.update(
        _collect_overrides(
            args=args,
            prefix="document_",
            defaults=DOCUMENT_PARAM_DEFAULTS,
        )
    )
    changed_detector_overrides = _changed_values(detector_overrides, BASE_DETECTOR_DEFAULTS)
    changed_document_overrides = _changed_values(document_detector_overrides, BASE_DOCUMENT_DEFAULTS)
    detector = UniversalTamperDetector(
        max_output_detections=args.max_detections,
        detector_overrides=detector_overrides,
        document_detector_overrides=document_detector_overrides,
    )
    result = detector.detect(
        image_path=args.image,
        output_path=args.output,
        report_path=args.report,
        max_detections=args.max_detections,
        evidence_output_dir=args.evidence_output_dir,
    )

    print("检测完成")
    print(f"状态: {result.status}")
    if result.reason:
        print(f"说明: {result.reason}")
    print(
        "全局证据: "
        f"ELA={result.evidence['ela_score']:.4f} | "
        f"噪声={result.evidence['noise_score']:.4f} | "
        f"文档规则={result.evidence['document_score']:.4f}"
    )
    for index, detection in enumerate(result.detections, start=1):
        print(
            f"检测{index}: {detection.label} | 行={detection.line_index} "
            f"| 评分={detection.score:.4f} | 框={detection.bbox}"
        )
    if not result.detections:
        print("未输出高亮框。")
    print(f"输出图片: {args.output}")
    print(f"检测报告: {args.report}")
    if result.evidence_artifacts:
        print(f"证据热力图目录: {args.evidence_output_dir}")
    if changed_detector_overrides:
        print(f"融合层参数覆盖: {json.dumps(changed_detector_overrides, ensure_ascii=False)}")
    if changed_document_overrides:
        print(f"文档规则层参数覆盖: {json.dumps(changed_document_overrides, ensure_ascii=False)}")

    report = build_report(result, max_report_candidates=int(detector.MAX_REPORT_CANDIDATES))
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
