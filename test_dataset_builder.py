from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from dataset_builder import (
    dataset_paths,
    download_originals,
    generate_labels,
    generate_tampered_dataset,
    search_sources,
    verify_dataset,
)
from evaluation import discover_image_pairs, evaluate_dataset, extract_ground_truth_boxes


def _encode_png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("编码测试图片失败")
    return buffer.tobytes()


def _synthetic_doc_image() -> np.ndarray:
    image = np.full((1400, 1100, 3), 248, dtype=np.uint8)
    rows = [
        "RECEIPT 2026-04-21",
        "ORDER NO 102438",
        "TABLE 07 TOTAL 88.00",
        "CASH 100.00 CHANGE 12.00",
        "TAX ID 310228",
        "THANK YOU",
    ]
    for index, text in enumerate(rows):
        baseline_y = 140 + index * 180
        cv2.putText(
            image,
            text,
            (70, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (30, 30, 30),
            4,
            lineType=cv2.LINE_AA,
        )
    return image


def _synthetic_natural_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height = 1280
    width = 1280
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x_gradient = np.tile(np.linspace(20, 210, width, dtype=np.uint8), (height, 1))
    y_gradient = np.tile(np.linspace(60, 230, height, dtype=np.uint8)[:, None], (1, width))
    image[..., 0] = x_gradient
    image[..., 1] = y_gradient
    image[..., 2] = ((x_gradient.astype(np.uint16) + y_gradient.astype(np.uint16)) // 2).astype(np.uint8)
    for _ in range(18):
        center = (int(rng.integers(80, width - 80)), int(rng.integers(80, height - 80)))
        radius = int(rng.integers(24, 120))
        color = tuple(int(value) for value in rng.integers(20, 245, size=3))
        cv2.circle(image, center, radius, color, thickness=-1)
    for _ in range(12):
        x1 = int(rng.integers(0, width - 180))
        y1 = int(rng.integers(0, height - 180))
        x2 = x1 + int(rng.integers(80, 220))
        y2 = y1 + int(rng.integers(80, 220))
        color = tuple(int(value) for value in rng.integers(20, 245, size=3))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)
    return image


class DatasetBuilderTest(unittest.TestCase):
    def test_search_writes_candidate_manifest_with_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "seed_dataset"

            def fake_openverse(keyword: str, page: int, page_size: int, source_type: str):
                if source_type == "docs":
                    return (
                        [
                            {
                                "provider": "openverse",
                                "source_type": "docs",
                                "keyword": keyword,
                                "title": "Receipt page",
                                "description": "Paper receipt with visible text",
                                "license": "CC BY 4.0",
                                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                                "source_page_url": "https://example.com/doc-page",
                                "download_url": "https://example.com/doc.png",
                                "creator": "tester",
                                "width": 1400,
                                "height": 1800,
                                "extension": "png",
                            }
                        ],
                        Counter(),
                    )
                return ([], Counter())

            def fake_wikimedia(keyword: str, page: int, page_size: int, source_type: str):
                if source_type == "natural":
                    return (
                        [
                            {
                                "provider": "wikimedia",
                                "source_type": "natural",
                                "keyword": keyword,
                                "title": "Street storefront",
                                "description": "Outdoor photo",
                                "license": "CC BY-SA 4.0",
                                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                                "source_page_url": "https://example.com/natural-page",
                                "download_url": "https://example.com/natural.png",
                                "creator": "tester",
                                "width": 1600,
                                "height": 1200,
                                "extension": "png",
                            }
                        ],
                        Counter(),
                    )
                return ([], Counter())

            with patch("dataset_builder._search_openverse", side_effect=fake_openverse), patch(
                "dataset_builder._search_wikimedia",
                side_effect=fake_wikimedia,
            ):
                manifest = search_sources(
                    dataset_root=root,
                    docs_target=1,
                    natural_target=1,
                    page_size=1,
                    max_pages=1,
                    oversample_factor=1,
                    doc_keywords=["receipt"],
                    natural_keywords=["street"],
                )

            self.assertEqual(len(manifest["candidates"]), 2)
            search_manifest = json.loads(dataset_paths(root).search_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(search_manifest["candidates"]), 2)
            for item in search_manifest["candidates"]:
                self.assertIn("title", item)
                self.assertIn("license", item)
                self.assertIn("source_page_url", item)
                self.assertIn("download_url", item)
                self.assertTrue(item["download_url"].startswith("https://"))

    def test_download_originals_filters_duplicates_and_writes_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "seed_dataset"
            paths = dataset_paths(root)
            paths.manifests_dir.mkdir(parents=True, exist_ok=True)
            search_manifest = {
                "candidates": [
                    {
                        "provider": "openverse",
                        "source_type": "docs",
                        "keyword": "receipt",
                        "title": "Doc A",
                        "description": "Receipt image",
                        "license": "CC BY 4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "source_page_url": "https://example.com/doc-a",
                        "download_url": "https://example.com/doc-a.png",
                        "creator": "tester",
                        "width": 1400,
                        "height": 1800,
                        "extension": "png",
                    },
                    {
                        "provider": "wikimedia",
                        "source_type": "docs",
                        "keyword": "receipt",
                        "title": "Doc B duplicate",
                        "description": "Duplicate receipt image",
                        "license": "CC BY 4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "source_page_url": "https://example.com/doc-b",
                        "download_url": "https://example.com/doc-b.png",
                        "creator": "tester",
                        "width": 1400,
                        "height": 1800,
                        "extension": "png",
                    },
                    {
                        "provider": "wikimedia",
                        "source_type": "natural",
                        "keyword": "street",
                        "title": "Natural A",
                        "description": "Street photo",
                        "license": "CC BY-SA 4.0",
                        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                        "source_page_url": "https://example.com/natural-a",
                        "download_url": "https://example.com/natural-a.png",
                        "creator": "tester",
                        "width": 1280,
                        "height": 1280,
                        "extension": "png",
                    },
                ]
            }
            paths.search_manifest_path.write_text(
                json.dumps(search_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            doc_bytes = _encode_png_bytes(_synthetic_doc_image())
            natural_bytes = _encode_png_bytes(_synthetic_natural_image(seed=7))

            def fake_fetch_binary(url: str) -> bytes:
                if "natural" in url:
                    return natural_bytes
                return doc_bytes

            with patch("dataset_builder._fetch_binary", side_effect=fake_fetch_binary):
                manifest = download_originals(
                    dataset_root=root,
                    docs_target=2,
                    natural_target=1,
                )

            self.assertEqual(manifest["actual_counts"]["docs"], 1)
            self.assertEqual(manifest["actual_counts"]["natural"], 1)
            self.assertIn("感知哈希重复", manifest["rejections"])
            for record in manifest["records"]:
                image_path = root / record["original_image"]
                self.assertTrue(image_path.exists())
                self.assertIsNotNone(cv2.imread(str(image_path)))

    def test_tamper_label_verify_pipeline_is_compatible_with_seed_dataset_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "seed_dataset"
            paths = dataset_paths(root)
            paths.originals_docs_dir.mkdir(parents=True, exist_ok=True)
            paths.originals_natural_dir.mkdir(parents=True, exist_ok=True)
            doc_path = paths.originals_docs_dir / "doc_0001_orig.png"
            natural_path = paths.originals_natural_dir / "natural_0001_orig.png"
            cv2.imwrite(str(doc_path), _synthetic_doc_image())
            cv2.imwrite(str(natural_path), _synthetic_natural_image(seed=11))
            original_manifest = {
                "records": [
                    {
                        "id": "doc_0001",
                        "source_type": "docs",
                        "provider": "openverse",
                        "keyword": "receipt",
                        "title": "Doc",
                        "description": "Synthetic document",
                        "license": "CC BY 4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "source_page_url": "https://example.com/doc",
                        "download_url": "https://example.com/doc.png",
                        "creator": "tester",
                        "width": 1100,
                        "height": 1400,
                        "dhash": "0",
                        "original_image": "originals/docs/doc_0001_orig.png",
                        "document_stats": {
                            "line_count": 6,
                            "character_count": 40,
                            "line_area_ratio_percent": 12,
                        },
                    },
                    {
                        "id": "natural_0001",
                        "source_type": "natural",
                        "provider": "wikimedia",
                        "keyword": "street",
                        "title": "Natural",
                        "description": "Synthetic natural image",
                        "license": "CC BY-SA 4.0",
                        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                        "source_page_url": "https://example.com/natural",
                        "download_url": "https://example.com/natural.png",
                        "creator": "tester",
                        "width": 1280,
                        "height": 1280,
                        "dhash": "1",
                        "original_image": "originals/natural/natural_0001_orig.png",
                        "document_stats": {
                            "line_count": 0,
                            "character_count": 0,
                            "line_area_ratio_percent": 0,
                        },
                    },
                ]
            }
            paths.manifests_dir.mkdir(parents=True, exist_ok=True)
            paths.original_manifest_path.write_text(
                json.dumps(original_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            tamper_manifest = generate_tampered_dataset(root, seed=123)
            self.assertEqual(tamper_manifest["record_count"], 6)

            labeled_manifest = generate_labels(root)
            self.assertEqual(len(labeled_manifest["records"]), 6)

            verify_report = verify_dataset(root)
            self.assertTrue(verify_report["passed"], verify_report)
            self.assertEqual(verify_report["pair_count"], 6)
            self.assertGreaterEqual(verify_report["extracted_ground_truth_box_count"], 6)

            pairs = discover_image_pairs(root)
            self.assertEqual(len(pairs), 6)
            boxes, _ = extract_ground_truth_boxes(pairs[0].image_path, pairs[0].answer_path)
            self.assertTrue(boxes)

            evaluation_report = evaluate_dataset(
                data_dir=root,
                iou_threshold=0.3,
                max_detections=1,
            )
            self.assertEqual(evaluation_report["image_count"], 6)
            self.assertGreaterEqual(evaluation_report["ground_truth_count"], 6)


if __name__ == "__main__":
    unittest.main()
