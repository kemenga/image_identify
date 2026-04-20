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
        self.assertTrue(any(overlaps(box, (356, 361, 34, 37)) for box in boxes))
        self.assertTrue(any(overlaps(box, (355, 626, 135, 30)) for box in boxes))
        self.assertTrue(any(overlaps(box, (818, 758, 150, 29)) for box in boxes))
        self.assertTrue(any(overlaps(box, (730, 320, 320, 400)) for box in boxes))

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
            self.assertTrue(report["detections"])

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
