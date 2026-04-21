from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detector import TamperRegion, UniversalTamperDetector
from document_detector import TraditionalTamperDetector
from evaluation import bbox_iou, evaluate_dataset, extract_ground_truth_boxes


BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_ARTIFACT_NAMES = {
    "ela_heatmap": "ela_heatmap.png",
    "noise_heatmap": "noise_heatmap.png",
    "fused_heatmap": "fused_heatmap.png",
}
LEAK_FIX_TARGET_IMAGES = {"image.png", "image4.png", "image5.png"}


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

    def _report_threshold_candidates(self) -> list[TamperRegion]:
        candidates: list[TamperRegion] = []
        for index, score in enumerate((11.0, 9.0, 7.0, 5.0, 3.0)):
            candidates.append(
                TamperRegion(
                    detection_type="text_noise_anomaly",
                    label="噪声篡改",
                    bbox=(40 + index * 120, 60, 28, 24),
                    score=score,
                    detail={
                        "zone": "evidence_text_noise_refinement",
                        "support_count": 3,
                        "evidence_contrast": 3.4,
                        "evidence_window_score": 0.65,
                        "evidence": {"local_noise": 0.35},
                    },
                    char_indices=[index * 10, index * 10 + 1, index * 10 + 2],
                )
            )
        return candidates

    def _ground_truth_boxes(self, image_name: str) -> list[tuple[int, int, int, int]]:
        image_path = sample_image_path(image_name)
        answer_path = image_path.with_name(f"{image_path.stem}检测结果{image_path.suffix}")
        boxes, _ = extract_ground_truth_boxes(image_path, answer_path)
        return boxes

    def _report_by_image(self, report: dict[str, object]) -> dict[str, dict[str, object]]:
        images = report["images"]
        return {item["image"]: item for item in images}

    def _best_candidate_match(
        self,
        candidates: list[TamperRegion],
        ground_truth_box: tuple[int, int, int, int],
        allowed_types: set[str],
    ) -> tuple[TamperRegion | None, float]:
        best_candidate: TamperRegion | None = None
        best_iou = 0.0
        for candidate in candidates:
            if candidate.detection_type not in allowed_types:
                continue
            current_iou = bbox_iou(ground_truth_box, candidate.bbox)
            if current_iou > best_iou:
                best_candidate = candidate
                best_iou = current_iou
        return best_candidate, best_iou

    def _assert_detections_match_ground_truth(
        self,
        image_name: str,
        detections: list[TamperRegion],
        expected_types: list[str],
        min_iou: float = 0.3,
    ) -> None:
        ground_truth_boxes = self._ground_truth_boxes(image_name)
        self.assertEqual([item.detection_type for item in detections], expected_types)
        for detection in detections:
            best_iou = max(
                bbox_iou(ground_truth_box, detection.bbox)
                for ground_truth_box in ground_truth_boxes
            )
            self.assertGreaterEqual(
                best_iou,
                min_iou,
                f"{image_name} 的高阈值候选 {detection.detection_type} {detection.bbox} 未命中标准答案",
            )
        for ground_truth_box in ground_truth_boxes:
            best_iou = max(bbox_iou(ground_truth_box, detection.bbox) for detection in detections)
            self.assertGreaterEqual(
                best_iou,
                min_iou,
                f"{image_name} 的标准答案框 {ground_truth_box} 未被高阈值结果覆盖",
            )

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
        self.assertEqual(result.threshold_score_mode, "report_confidence")
        self.assertIn("threshold_score", result.detections[0].detail)
        self.assertIn("threshold_score_mode", result.detections[0].detail)

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
            self.assertIn("threshold_score_mode", report)
            self.assertIn("threshold_score_value", report)
            self.assertTrue(report["detections"])
            self.assertLessEqual(len(report["top_candidates"]), report["reportable_candidate_count"])
            self.assertIn("threshold_score", report["detections"][0]["detail"])

    def test_detector_accepts_tunable_overrides(self) -> None:
        detector = UniversalTamperDetector(
            max_output_detections=6,
            detector_overrides={
                "GLOBAL_EVIDENCE_WEIGHT": 1.1,
                "MAX_REPORT_CANDIDATES": 3,
                "REPORT_CONFIDENCE_THRESHOLD": 66.0,
            },
            document_detector_overrides={
                "TEXT_NOISE_THRESHOLD": 5.2,
                "METHOD_WEIGHT_STROKE": 1.6,
                "GLOBAL_SCAN_MAX_SIDE": 1200,
                "GLOBAL_SCAN_MAX_WINDOWS": 8000,
                "SPECIAL_RULE_TYPES": "digit_window,time_group",
            },
        )
        self.assertEqual(detector.GLOBAL_EVIDENCE_WEIGHT, 1.1)
        self.assertEqual(detector.MAX_REPORT_CANDIDATES, 3)
        self.assertEqual(detector.REPORT_CONFIDENCE_THRESHOLD, 66.0)
        self.assertEqual(detector.document_detector.TEXT_NOISE_THRESHOLD, 5.2)
        self.assertEqual(detector.document_detector.METHOD_WEIGHTS["stroke"], 1.6)
        self.assertEqual(detector.document_detector.GLOBAL_SCAN_MAX_SIDE, 1200)
        self.assertEqual(detector.document_detector.GLOBAL_SCAN_MAX_WINDOWS, 8000)
        self.assertEqual(detector.document_detector.SPECIAL_RULE_TYPES, {"digit_window", "time_group"})

    def test_large_image_dense_scan_is_downscaled_and_budgeted(self) -> None:
        detector = TraditionalTamperDetector()
        gray = np.zeros((4130, 5704), dtype=np.uint8)

        scan_gray, scale, scan_window_size, scan_stride, mapped_window_size = detector._prepare_dense_scan(
            gray=gray,
            window_size=24,
            stride=12,
        )

        self.assertLess(max(scan_gray.shape), max(gray.shape))
        self.assertLess(scale, 1.0)
        self.assertGreater(scan_window_size, 0)
        self.assertGreater(scan_stride, 0)
        self.assertGreater(mapped_window_size, scan_window_size)
        self.assertLessEqual(
            detector._dense_scan_window_count(scan_gray.shape, scan_window_size, scan_stride),
            detector.GLOBAL_SCAN_MAX_WINDOWS,
        )

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
                    "--detector-report-confidence-threshold",
                    "66",
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

    def test_single_report_threshold_can_reduce_output_boxes(self) -> None:
        image_path = str(sample_image_path("截图.png"))
        baseline = UniversalTamperDetector().detect(image_path, max_detections=8)
        tightened = UniversalTamperDetector(
            detector_overrides={"REPORT_CONFIDENCE_THRESHOLD": 999.0}
        ).detect(image_path, max_detections=8)
        self.assertGreaterEqual(len(baseline.detections), len(tightened.detections))
        self.assertFalse(tightened.detections)

    def test_report_threshold_lowering_increases_reportable_box_count(self) -> None:
        candidates = self._report_threshold_candidates()
        best_time_candidate = UniversalTamperDetector._best_candidate_of_type(candidates, "time_group")
        best_digit_candidate = UniversalTamperDetector._best_candidate_of_type(candidates, "digit_window")
        has_strong_refinement = any(
            UniversalTamperDetector._is_strong_evidence_refinement(candidate)
            for candidate in candidates
            if not UniversalTamperDetector._is_special_scene_conflicting_refinement(
                candidate,
                best_time_candidate,
                best_digit_candidate,
            )
        )
        confidences = sorted(
            [
                UniversalTamperDetector._single_threshold_score(
                    candidate,
                    has_strong_refinement,
                    best_time_candidate,
                    best_digit_candidate,
                )
                for candidate in candidates
            ],
            reverse=True,
        )
        threshold_between_second_and_third = (confidences[1] + confidences[2]) / 2

        tightened = UniversalTamperDetector._reportable_candidates(
            candidates,
            max_items=len(candidates),
            min_confidence=threshold_between_second_and_third,
            single_threshold_mode=True,
        )
        loosened = UniversalTamperDetector._reportable_candidates(
            candidates,
            max_items=len(candidates),
            min_confidence=0.0,
            single_threshold_mode=True,
        )

        self.assertEqual(len(tightened), 2)
        self.assertEqual(len(loosened), len(candidates))
        self.assertLess(len(tightened), len(loosened))

    def test_low_report_threshold_is_limited_by_max_items_not_type_caps(self) -> None:
        candidates = self._report_threshold_candidates()

        selected = UniversalTamperDetector._reportable_candidates(
            candidates,
            max_items=3,
            min_confidence=0.0,
            single_threshold_mode=True,
        )

        self.assertEqual(len(selected), 3)

    def test_high_report_threshold_keeps_ground_truth_candidates(self) -> None:
        high_threshold = 95.0

        screenshot = UniversalTamperDetector(
            detector_overrides={"REPORT_CONFIDENCE_THRESHOLD": high_threshold}
        ).detect(str(sample_image_path("截图.png")), max_detections=8)
        self._assert_detections_match_ground_truth("截图.png", screenshot.detections, ["time_group"])

        receipt = UniversalTamperDetector(
            detector_overrides={"REPORT_CONFIDENCE_THRESHOLD": high_threshold}
        ).detect(str(sample_image_path("票据.png")), max_detections=8)
        self._assert_detections_match_ground_truth("票据.png", receipt.detections, ["digit_window"])

        sample_300 = UniversalTamperDetector(
            detector_overrides={"REPORT_CONFIDENCE_THRESHOLD": high_threshold}
        ).detect(str(sample_image_path("300.png")), max_detections=8)
        self._assert_detections_match_ground_truth("300.png", sample_300.detections, ["text_noise_anomaly"])

        sample_400 = UniversalTamperDetector(
            detector_overrides={"REPORT_CONFIDENCE_THRESHOLD": high_threshold}
        ).detect(str(sample_image_path("400.png")), max_detections=8)
        self._assert_detections_match_ground_truth(
            "400.png",
            sample_400.detections,
            ["text_noise_anomaly", "text_noise_anomaly"],
        )

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
        by_image = self._report_by_image(report)
        self.assertEqual(report["image_count"], 9)
        self.assertEqual(report["ground_truth_count"], 27)
        self.assertGreaterEqual(report["hit_count"], 26)
        self.assertGreaterEqual(report["recall"], 0.95)
        self.assertLessEqual(report["false_positive_count"], 6)
        self.assertTrue(LEAK_FIX_TARGET_IMAGES.issubset(set(by_image)))
        self.assertEqual(by_image["300.png"]["recall"], 1.0)
        self.assertEqual(by_image["400.png"]["recall"], 1.0)
        self.assertEqual(by_image["image.png"]["recall"], 1.0)
        self.assertEqual(by_image["image4.png"]["recall"], 1.0)
        self.assertEqual(by_image["image5.png"]["recall"], 1.0)
        self.assertEqual(by_image["截图.png"]["recall"], 1.0)
        self.assertEqual(by_image["票据.png"]["recall"], 1.0)
        self.assertGreaterEqual(by_image["身份证.png"]["recall"], 0.8)

    def test_new_samples_are_explicitly_tracked_in_dataset_evaluation(self) -> None:
        report = evaluate_dataset(
            data_dir=BASE_DIR / "data",
            iou_threshold=0.3,
            max_detections=8,
        )
        by_image = self._report_by_image(report)

        image_sample = by_image["image.png"]
        self.assertEqual(len(image_sample["ground_truth_boxes"]), 7)
        self.assertGreater(image_sample["recall"], 0.0)
        self.assertGreaterEqual(
            sum(1 for match in image_sample["matches"] if match["matched"]),
            1,
        )
        self.assertTrue(
            any(
                detection["type"] in {"text_region", "text_noise_anomaly"}
                for detection in image_sample["detections"]
            )
        )
        self.assertLessEqual(image_sample["false_positive_count"], len(image_sample["detections"]))
        self.assertLessEqual(len(image_sample["detections"]), 8)

        image4_sample = by_image["image4.png"]
        self.assertEqual(len(image4_sample["ground_truth_boxes"]), 3)
        self.assertGreater(image4_sample["recall"], 0.0)
        self.assertGreaterEqual(
            sum(1 for match in image4_sample["matches"] if match["matched"]),
            1,
        )
        self.assertIn("digit_window", {detection["type"] for detection in image4_sample["detections"]})
        self.assertIn("text_noise_anomaly", {detection["type"] for detection in image4_sample["detections"]})
        self.assertLessEqual(image4_sample["false_positive_count"], len(image4_sample["detections"]))
        self.assertLessEqual(len(image4_sample["detections"]), 8)

        image5_sample = by_image["image5.png"]
        self.assertEqual(len(image5_sample["ground_truth_boxes"]), 3)
        self.assertGreater(image5_sample["recall"], 0.0)
        self.assertGreaterEqual(
            sum(1 for match in image5_sample["matches"] if match["matched"]),
            1,
        )
        self.assertIn(
            "text_noise_anomaly",
            {detection["type"] for detection in image5_sample["detections"]},
        )
        self.assertLessEqual(image5_sample["false_positive_count"], len(image5_sample["detections"]))
        self.assertLessEqual(len(image5_sample["detections"]), 8)

    def test_new_samples_candidate_pool_cover_repair_targets(self) -> None:
        expectations = {
            "image.png": [
                (0, 0.3, {"text_region", "text_noise_anomaly"}),
                (1, 0.3, {"text_region", "text_noise_anomaly"}),
                (2, 0.45, {"text_region", "text_noise_anomaly"}),
                (3, 0.3, {"text_noise_anomaly", "region_anomaly"}),
                (4, 0.4, {"text_region", "text_noise_anomaly"}),
                (5, 0.1, {"text_noise_anomaly", "region_anomaly"}),
                (6, 0.3, {"text_region", "text_noise_anomaly"}),
            ],
            "image4.png": [
                (0, 0.1, {"text_noise_anomaly", "text_region"}),
                (1, 0.12, {"text_noise_anomaly", "text_region"}),
                (2, 0.3, {"digit_window", "text_noise_anomaly"}),
            ],
            "image5.png": [
                (0, 0.01, {"text_noise_anomaly", "region_anomaly"}),
                (1, 0.02, {"text_noise_anomaly", "region_anomaly"}),
                (2, 0.3, {"text_noise_anomaly", "region_anomaly"}),
            ],
        }

        for image_name, image_expectations in expectations.items():
            result = self._detect(image_name)
            ground_truth_boxes = self._ground_truth_boxes(image_name)
            for ground_truth_index, min_iou, allowed_types in image_expectations:
                with self.subTest(image_name=image_name, ground_truth_index=ground_truth_index):
                    best_candidate, best_iou = self._best_candidate_match(
                        result.candidate_regions,
                        ground_truth_boxes[ground_truth_index],
                        allowed_types,
                    )
                    self.assertIsNotNone(best_candidate)
                    self.assertGreaterEqual(
                        best_iou,
                        min_iou,
                        (
                            f"{image_name} 的标准框 {ground_truth_index} 缺少候选池覆盖，"
                            f"允许类型 {sorted(allowed_types)}，当前最佳 IoU={best_iou:.4f}"
                        ),
                    )

    def test_reportable_outputs_suppress_correct_text_false_positives(self) -> None:
        report = evaluate_dataset(
            data_dir=BASE_DIR / "data",
            iou_threshold=0.3,
            max_detections=8,
        )
        by_image = {item["image"]: item for item in report["images"]}

        sample_300 = by_image["300.png"]
        self.assertEqual(sample_300["recall"], 1.0)
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
        self.assertEqual(sample_300["detections"][0]["type"], "text_noise_anomaly")

        sample_400 = by_image["400.png"]
        self.assertEqual(sample_400["recall"], 1.0)
        self.assertEqual(len(sample_400["ground_truth_boxes"]), 2)
        self.assertTrue(all(match["matched"] for match in sample_400["matches"]))
        self.assertEqual(sample_400["detections"][0]["type"], "text_noise_anomaly")

        image_sample = by_image["image.png"]
        self.assertGreater(image_sample["recall"], 0.0)
        self.assertLessEqual(image_sample["false_positive_count"], len(image_sample["detections"]))
        self.assertLessEqual(len(image_sample["detections"]), 8)

        image4_sample = by_image["image4.png"]
        self.assertGreater(image4_sample["recall"], 0.0)
        self.assertLessEqual(image4_sample["false_positive_count"], len(image4_sample["detections"]))
        self.assertLessEqual(len(image4_sample["detections"]), 8)

        image5_sample = by_image["image5.png"]
        self.assertGreater(image5_sample["recall"], 0.0)
        self.assertLessEqual(image5_sample["false_positive_count"], len(image5_sample["detections"]))
        self.assertLessEqual(len(image5_sample["detections"]), 8)
        self.assertIn("text_noise_anomaly", {detection["type"] for detection in image5_sample["detections"]})

        screenshot = by_image["截图.png"]
        self.assertEqual(screenshot["recall"], 1.0)
        self.assertEqual(screenshot["detections"][0]["type"], "time_group")

        receipt = by_image["票据.png"]
        self.assertEqual(receipt["recall"], 1.0)
        self.assertEqual(receipt["detections"][0]["type"], "digit_window")
        self.assertLessEqual(
            sum(1 for detection in receipt["detections"] if detection["type"] == "digit_window"),
            1,
        )

        invoice = by_image["发票.png"]
        self.assertGreaterEqual(invoice["recall"], 0.0)
        self.assertLessEqual(invoice["false_positive_count"], len(invoice["detections"]))
        self.assertTrue(invoice["detections"])
        self.assertTrue(
            all(
                detection["type"] in {"text_noise_anomaly", "text_region"}
                for detection in invoice["detections"]
            )
        )

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
            by_image = self._report_by_image(report)
            self.assertIn("images", report)
            self.assertEqual(report["image_count"], 9)
            self.assertEqual(report["ground_truth_count"], 27)
            self.assertGreaterEqual(report["hit_count"], 15)
            self.assertGreaterEqual(report["recall"], 0.55)
            self.assertTrue(any(item["image"] == "300.png" for item in report["images"]))
            self.assertTrue(any(item["image"] == "400.png" for item in report["images"]))
            self.assertTrue(any(item["image"] == "image.png" for item in report["images"]))
            self.assertTrue(any(item["image"] == "image4.png" for item in report["images"]))
            self.assertTrue(any(item["image"] == "image5.png" for item in report["images"]))
            self.assertEqual(by_image["截图.png"]["detections"][0]["type"], "time_group")
            self.assertEqual(by_image["票据.png"]["detections"][0]["type"], "digit_window")

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
