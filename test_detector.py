from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detector import UniversalTamperDetector
from evaluation import bbox_iou, evaluate_dataset, extract_ground_truth_boxes


BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_ARTIFACT_NAMES = {
    "ela_heatmap": "ela_heatmap.png",
    "noise_heatmap": "noise_heatmap.png",
    "fused_heatmap": "fused_heatmap.png",
}


def sample_image_path(image_name: str) -> Path:
    root_path = BASE_DIR / image_name
    if root_path.exists():
        return root_path
    return BASE_DIR / "data" / image_name


def overlaps(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = lhs
    bx, by, bw, bh = rhs
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    return not (ax2 <= bx or bx2 <= ax or ay2 <= by or by2 <= ay)


class UniversalTamperDetectorTest(unittest.TestCase):
    def _detect(self, image_name: str):
        detector = UniversalTamperDetector()
        return detector.detect(str(sample_image_path(image_name)))

    def test_screenshot_detects_time_region(self) -> None:
        result = self._detect("截图.png")
        self.assertEqual(result.status, "detected")
        self.assertTrue(result.detections)
        self.assertEqual(result.detections[0].label, "时间篡改")
        self.assertTrue(overlaps(result.detections[0].bbox, (584, 898, 157, 49)))
        self.assertIn("ela_score", result.evidence)
        self.assertIn("noise_score", result.evidence)
        self.assertIn("document_score", result.evidence)
        self.assertEqual(result.evidence_artifacts, {})

    def test_receipt_detects_digit_region_first(self) -> None:
        result = self._detect("票据.png")
        self.assertEqual(result.status, "detected")
        self.assertTrue(result.detections)
        self.assertEqual(result.detections[0].label, "数字篡改")
        self.assertTrue(overlaps(result.detections[0].bbox, (216, 858, 19, 39)))

    def test_id_card_covers_major_regions(self) -> None:
        result = self._detect("身份证.png")
        self.assertEqual(result.status, "detected")
        self.assertTrue(result.detections)
        boxes = [detection.bbox for detection in result.detections]
        self.assertTrue(any(overlaps(box, (355, 626, 135, 30)) for box in boxes))
        self.assertTrue(any(overlaps(box, (818, 758, 150, 29)) for box in boxes))
        self.assertTrue(any(overlaps(box, (730, 320, 320, 400)) for box in boxes))
        self.assertFalse(any(overlaps(box, (356, 361, 34, 37)) for box in boxes))

    def test_cli_writes_output_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "result.png"
            report_path = temp_path / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BASE_DIR / "main.py"),
                    "--image",
                    str(sample_image_path("截图.png")),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(BASE_DIR),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.exists())
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "detected")
            self.assertIn("evidence", report)
            self.assertEqual(report["evidence_artifacts"], {})
            self.assertIn("top_candidates", report)
            self.assertIn("reportable_candidate_count", report)
            self.assertTrue(report["detections"])
            self.assertLessEqual(len(report["top_candidates"]), report["reportable_candidate_count"])

    def test_detector_accepts_tunable_overrides(self) -> None:
        detector = UniversalTamperDetector(
            max_output_detections=6,
            detector_overrides={
                "GLOBAL_EVIDENCE_WEIGHT": 1.1,
                "MAX_REPORT_CANDIDATES": 3,
            },
            document_detector_overrides={
                "TEXT_NOISE_THRESHOLD": 5.2,
                "METHOD_WEIGHT_STROKE": 1.6,
                "SPECIAL_RULE_TYPES": "digit_window,time_group",
            },
        )
        self.assertEqual(detector.GLOBAL_EVIDENCE_WEIGHT, 1.1)
        self.assertEqual(detector.MAX_REPORT_CANDIDATES, 3)
        self.assertEqual(detector.document_detector.TEXT_NOISE_THRESHOLD, 5.2)
        self.assertEqual(detector.document_detector.METHOD_WEIGHTS["stroke"], 1.6)
        self.assertEqual(detector.document_detector.SPECIAL_RULE_TYPES, {"digit_window", "time_group"})

    def test_cli_accepts_tunable_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            report_path = temp_path / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BASE_DIR / "main.py"),
                    "--image",
                    str(sample_image_path("截图.png")),
                    "--report",
                    str(report_path),
                    "--detector-global-evidence-weight",
                    "1.05",
                    "--document-text-noise-threshold",
                    "5.1",
                    "--document-method-weight-stroke",
                    "1.4",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(BASE_DIR),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report_path.exists())
            self.assertIn("融合层参数覆盖", completed.stdout)
            self.assertIn("文档规则层参数覆盖", completed.stdout)

    def test_api_writes_evidence_heatmaps(self) -> None:
        detector = UniversalTamperDetector()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = detector.detect(
                str(sample_image_path("截图.png")),
                evidence_output_dir=temp_dir,
            )
            self.assertEqual(set(result.evidence_artifacts), set(EVIDENCE_ARTIFACT_NAMES))
            for artifact_key, file_name in EVIDENCE_ARTIFACT_NAMES.items():
                artifact_path = Path(result.evidence_artifacts[artifact_key])
                self.assertEqual(artifact_path.name, file_name)
                self.assertTrue(artifact_path.exists())
                heatmap = cv2.imread(str(artifact_path))
                self.assertIsNotNone(heatmap)
                self.assertEqual(heatmap.shape[:2], cv2.imread(str(sample_image_path("截图.png"))).shape[:2])

    def test_cli_writes_evidence_heatmaps_and_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            evidence_dir = temp_path / "evidence"
            report_path = temp_path / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BASE_DIR / "main.py"),
                    "--image",
                    str(sample_image_path("截图.png")),
                    "--report",
                    str(report_path),
                    "--evidence-output-dir",
                    str(evidence_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(BASE_DIR),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(set(report["evidence_artifacts"]), set(EVIDENCE_ARTIFACT_NAMES))
            for artifact_key, file_name in EVIDENCE_ARTIFACT_NAMES.items():
                artifact_path = evidence_dir / file_name
                self.assertEqual(report["evidence_artifacts"][artifact_key], str(artifact_path))
                self.assertTrue(artifact_path.exists())
                self.assertIsNotNone(cv2.imread(str(artifact_path)))

    def test_data_samples_include_text_noise_candidates(self) -> None:
        data_dir = BASE_DIR / "data"
        sample_paths = [
            path
            for path in sorted(data_dir.glob("*.png"))
            if "检测结果" not in path.name
        ]
        if not sample_paths:
            self.skipTest("缺少 data 样例图")

        detector = UniversalTamperDetector()
        detected_sample_count = 0
        for image_path in sample_paths:
            result = detector.detect(str(image_path))
            if any(candidate.detection_type == "text_noise_anomaly" for candidate in result.candidate_regions):
                detected_sample_count += 1

        self.assertGreaterEqual(detected_sample_count, 1)

    def test_same_size_answer_images_extract_ground_truth_boxes(self) -> None:
        for image_name in ("截图.png", "票据.png"):
            image_path = sample_image_path(image_name)
            answer_path = image_path.with_name(f"{image_path.stem}检测结果{image_path.suffix}")
            boxes, alignment = extract_ground_truth_boxes(image_path, answer_path)
            self.assertTrue(boxes)
            self.assertEqual(alignment.mode, "same_size")
            self.assertTrue(all(box[2] > 0 and box[3] > 0 for box in boxes))

    def test_resized_answer_images_align_to_original(self) -> None:
        for image_name in ("300.png", "身份证.png"):
            image_path = sample_image_path(image_name)
            answer_path = image_path.with_name(f"{image_path.stem}检测结果{image_path.suffix}")
            image = cv2.imread(str(image_path))
            boxes, alignment = extract_ground_truth_boxes(image_path, answer_path)
            self.assertTrue(boxes)
            self.assertEqual(alignment.mode, "orb_affine")
            self.assertGreaterEqual(alignment.match_count, 50)
            self.assertGreaterEqual(alignment.inlier_ratio, 0.55)
            for x, y, w, h in boxes:
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + w, image.shape[1])
                self.assertLessEqual(y + h, image.shape[0])

    def test_dataset_evaluation_recall_threshold(self) -> None:
        report = evaluate_dataset(
            data_dir=BASE_DIR / "data",
            iou_threshold=0.3,
            max_detections=8,
        )
        self.assertGreaterEqual(report["recall"], 0.90)
        self.assertLessEqual(report["false_positive_count"], 3)
        for image_result in report["images"]:
            self.assertTrue(
                any(match["matched"] for match in image_result["matches"]),
                image_result["image"],
            )

    def test_reportable_outputs_suppress_correct_text_false_positives(self) -> None:
        report = evaluate_dataset(
            data_dir=BASE_DIR / "data",
            iou_threshold=0.3,
            max_detections=8,
        )
        by_image = {item["image"]: item for item in report["images"]}

        sample_300 = by_image["300.png"]
        self.assertEqual(sample_300["false_positive_count"], 0)
        self.assertTrue(
            any(
                detection["type"] == "text_noise_anomaly"
                and max(
                    bbox_iou(tuple(gt_box), tuple(detection["bbox"]))
                    for gt_box in sample_300["ground_truth_boxes"]
                )
                >= 0.3
                for detection in sample_300["detections"]
            )
        )
        self.assertFalse(any(detection["type"] == "time_group" for detection in sample_300["detections"]))
        self.assertFalse(any(detection["type"] == "text_region" for detection in sample_300["detections"]))

        sample_400 = by_image["400.png"]
        self.assertEqual(sample_400["false_positive_count"], 0)
        self.assertEqual(sample_400["recall"], 1.0)
        self.assertEqual(len(sample_400["ground_truth_boxes"]), 2)
        self.assertLessEqual(len(sample_400["detections"]), 3)
        self.assertTrue(all(match["matched"] for match in sample_400["matches"]))
        self.assertTrue(all(detection["type"] == "text_noise_anomaly" for detection in sample_400["detections"]))

        screenshot = by_image["截图.png"]
        self.assertEqual(screenshot["false_positive_count"], 0)
        self.assertLess(len(screenshot["detections"]), 8)
        self.assertEqual(screenshot["detections"][0]["type"], "time_group")

        receipt = by_image["票据.png"]
        self.assertEqual(receipt["false_positive_count"], 0)
        self.assertEqual(receipt["detections"][0]["type"], "digit_window")
        self.assertLessEqual(
            sum(1 for detection in receipt["detections"] if detection["type"] == "digit_window"),
            1,
        )

        invoice = by_image["发票.png"]
        self.assertEqual(invoice["false_positive_count"], 0)
        self.assertEqual(invoice["recall"], 1)
        self.assertEqual(len(invoice["detections"]), 4)
        self.assertTrue(any(detection["bbox"][0] > 1000 and detection["bbox"][1] < 100 for detection in invoice["detections"]))
        self.assertTrue(any(detection["bbox"][0] < 300 and 580 <= detection["bbox"][1] <= 700 for detection in invoice["detections"]))
        self.assertTrue(any(450 <= detection["bbox"][0] <= 550 and detection["bbox"][1] >= 950 for detection in invoice["detections"]))

        id_card = by_image["身份证.png"]
        self.assertEqual(id_card["false_positive_count"], 0)
        id_card_boxes = [tuple(detection["bbox"]) for detection in id_card["detections"]]
        self.assertFalse(any(overlaps(box, (356, 361, 34, 37)) for box in id_card_boxes))
        self.assertFalse(
            any(
                detection["type"] == "region_anomaly"
                and detection["bbox"][2] * detection["bbox"][3] <= 2500
                for detection in id_card["detections"]
            )
        )

    def test_evaluation_cli_writes_dataset_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "evaluation_report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BASE_DIR / "evaluation.py"),
                    "--data-dir",
                    str(BASE_DIR / "data"),
                    "--report",
                    str(report_path),
                    "--iou-threshold",
                    "0.3",
                    "--max-detections",
                    "8",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=str(BASE_DIR),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["recall"], 0.90)
            self.assertLessEqual(report["false_positive_count"], 3)
            self.assertIn("images", report)
            self.assertTrue(any(item["image"] == "300.png" for item in report["images"]))
            self.assertTrue(any(item["image"] == "400.png" for item in report["images"]))

    def test_missing_image_raises_clear_error(self) -> None:
        detector = UniversalTamperDetector()
        with self.assertRaises(FileNotFoundError):
            detector.detect(str(BASE_DIR / "不存在.png"))

    def test_blank_image_returns_legal_status(self) -> None:
        detector = UniversalTamperDetector()
        blank = np.full((320, 320, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "blank.png"
            self.assertTrue(cv2.imwrite(str(image_path), blank))
            result = detector.detect(str(image_path))
        self.assertIn(result.status, {"insufficient_context", "no_detection", "detected"})


if __name__ == "__main__":
    unittest.main()
