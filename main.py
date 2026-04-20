from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector import UniversalTamperDetector, build_report


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文档优先的通用图像篡改检测")
    parser.add_argument(
        "--image",
        default=str(BASE_DIR / "data/截图.png"),
        help="输入图片路径",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "detected_result.png"),
        help="输出高亮图片路径",
    )
    parser.add_argument(
        "--report",
        default=str(BASE_DIR / "detected_report.json"),
        help="输出 JSON 报告路径",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=5,
        help="最多输出的检测框数量",
    )
    parser.add_argument(
        "--evidence-output-dir",
        default=None,
        help="辅助证据热力图输出目录；传入后生成 ELA、噪声和融合热力图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = UniversalTamperDetector(max_output_detections=args.max_detections)
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

    report = build_report(result)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
