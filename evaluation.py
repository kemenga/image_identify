from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detector import UniversalTamperDetector


@dataclass(slots=True)
class ImagePair:
    image_path: Path
    answer_path: Path


@dataclass(slots=True)
class AlignmentResult:
    matrix: np.ndarray
    match_count: int
    inlier_count: int
    inlier_ratio: float
    scale: float
    mode: str


def discover_image_pairs(data_dir: str | Path) -> list[ImagePair]:
    root = Path(data_dir)
    pairs = _discover_flat_image_pairs(root)
    structured_pairs = _discover_seed_dataset_pairs(root)
    if not structured_pairs:
        return pairs

    known_images = {pair.image_path.resolve() for pair in pairs}
    for pair in structured_pairs:
        if pair.image_path.resolve() in known_images:
            continue
        pairs.append(pair)
    return sorted(pairs, key=lambda item: item.image_path.name)


def _discover_flat_image_pairs(root: Path) -> list[ImagePair]:
    pairs: list[ImagePair] = []
    for image_path in sorted(root.glob("*.png")):
        if "检测结果" in image_path.stem:
            continue
        answer_path = image_path.with_name(f"{image_path.stem}检测结果{image_path.suffix}")
        if answer_path.exists():
            pairs.append(ImagePair(image_path=image_path, answer_path=answer_path))
    return pairs


def _discover_seed_dataset_pairs(root: Path) -> list[ImagePair]:
    tampered_dir = root / "tampered"
    labels_dir = root / "labels_png"
    if not tampered_dir.exists() or not labels_dir.exists():
        return []

    pairs: list[ImagePair] = []
    for image_path in sorted(tampered_dir.glob("*.png")):
        answer_path = labels_dir / f"{image_path.stem}检测结果{image_path.suffix}"
        if answer_path.exists():
            pairs.append(ImagePair(image_path=image_path, answer_path=answer_path))
    return pairs


def bbox_iou(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = lhs
    bx, by, bw, bh = rhs
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    intersection = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    union = float(aw * ah + bw * bh - intersection)
    return intersection / max(union, 1.0)


def extract_ground_truth_boxes(
    image_path: str | Path,
    answer_path: str | Path,
) -> tuple[list[tuple[int, int, int, int]], AlignmentResult]:
    original = _read_image(Path(image_path))
    answer = _read_image(Path(answer_path))
    alignment = _estimate_alignment(original, answer)
    answer_boxes = _extract_answer_color_boxes(
        original=original,
        answer=answer,
        prefer_outer=alignment.mode != "same_size",
        use_difference=alignment.mode == "same_size",
    )
    if not answer_boxes:
        raise ValueError(f"未从标准答案图中抽取到彩色标注框: {answer_path}")
    mapped_boxes = [
        _clip_box(_map_box_to_original(box, alignment.matrix), original.shape)
        for box in answer_boxes
    ]
    valid_boxes = [box for box in mapped_boxes if _is_valid_ground_truth_box(box)]
    if not valid_boxes:
        raise ValueError(f"标准答案框映射后全部无效: {answer_path}")
    return _merge_overlapping_boxes(valid_boxes), alignment


def evaluate_dataset(
    data_dir: str | Path,
    iou_threshold: float = 0.3,
    max_detections: int = 8,
) -> dict[str, object]:
    pairs = discover_image_pairs(data_dir)
    detector = UniversalTamperDetector(max_output_detections=max_detections)
    image_results: list[dict[str, object]] = []
    total_gt = 0
    total_hits = 0
    total_false_positives = 0
    best_ious: list[float] = []

    for pair in pairs:
        gt_boxes, alignment = extract_ground_truth_boxes(pair.image_path, pair.answer_path)
        result = detector.detect(str(pair.image_path), max_detections=max_detections)
        detections = [
            {
                "type": item.detection_type,
                "label": item.label,
                "bbox": list(item.bbox),
                "score": round(float(item.score), 4),
            }
            for item in result.detections
        ]
        detection_boxes = [tuple(item["bbox"]) for item in detections]
        matches = _match_boxes(gt_boxes, detection_boxes, iou_threshold)
        hit_count = sum(1 for item in matches if item["matched"])
        false_positive_count = sum(
            1
            for det_index in range(len(detection_boxes))
            if all(match["detection_index"] != det_index for match in matches if match["matched"])
        )

        total_gt += len(gt_boxes)
        total_hits += hit_count
        total_false_positives += false_positive_count
        best_ious.extend(float(item["best_iou"]) for item in matches)
        image_results.append(
            {
                "image": pair.image_path.name,
                "answer": pair.answer_path.name,
                "alignment": {
                    "mode": alignment.mode,
                    "match_count": alignment.match_count,
                    "inlier_count": alignment.inlier_count,
                    "inlier_ratio": round(alignment.inlier_ratio, 4),
                    "scale": round(alignment.scale, 4),
                },
                "ground_truth_boxes": [list(box) for box in gt_boxes],
                "detections": detections,
                "matches": matches,
                "recall": round(hit_count / max(len(gt_boxes), 1), 4),
                "false_positive_count": false_positive_count,
                "average_best_iou": round(float(np.mean([item["best_iou"] for item in matches])), 4),
            }
        )

    recall = total_hits / max(total_gt, 1)
    return {
        "data_dir": str(Path(data_dir)),
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
        "image_count": len(pairs),
        "ground_truth_count": total_gt,
        "hit_count": total_hits,
        "recall": round(recall, 4),
        "false_positive_count": total_false_positives,
        "average_best_iou": round(float(np.mean(best_ious)) if best_ious else 0.0, 4),
        "images": image_results,
    }


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def _estimate_alignment(original: np.ndarray, answer: np.ndarray) -> AlignmentResult:
    if original.shape[:2] == answer.shape[:2]:
        return AlignmentResult(
            matrix=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            match_count=0,
            inlier_count=0,
            inlier_ratio=1.0,
            scale=1.0,
            mode="same_size",
        )

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    answer_gray = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    original_keypoints, original_descriptors = orb.detectAndCompute(original_gray, None)
    answer_keypoints, answer_descriptors = orb.detectAndCompute(answer_gray, None)
    if original_descriptors is None or answer_descriptors is None:
        raise ValueError("自动对齐失败: ORB 特征不足")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(original_descriptors, answer_descriptors),
        key=lambda item: item.distance,
    )[:500]
    if len(matches) < 50:
        raise ValueError(f"自动对齐失败: 匹配点不足 {len(matches)} < 50")

    original_points = np.float32([original_keypoints[item.queryIdx].pt for item in matches])
    answer_points = np.float32([answer_keypoints[item.trainIdx].pt for item in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        original_points,
        answer_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=3000,
    )
    if matrix is None or inliers is None:
        raise ValueError("自动对齐失败: 无法估计仿射变换")
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(len(matches), 1)
    scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    if inlier_ratio < 0.55:
        raise ValueError(f"自动对齐失败: 内点比例过低 {inlier_ratio:.4f} < 0.55")
    if not 0.5 <= scale <= 2.0:
        raise ValueError(f"自动对齐失败: 缩放比例异常 {scale:.4f}")

    return AlignmentResult(
        matrix=cv2.invertAffineTransform(matrix).astype(np.float32),
        match_count=len(matches),
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        scale=scale,
        mode="orb_affine",
    )


def _extract_answer_color_boxes(
    original: np.ndarray,
    answer: np.ndarray,
    prefer_outer: bool,
    use_difference: bool,
) -> list[tuple[int, int, int, int]]:
    difference_mask = _build_difference_mask(original, answer) if use_difference else None
    red_boxes = _extract_color_boxes(answer, "red", difference_mask)
    if red_boxes:
        if not prefer_outer:
            return _merge_overlapping_boxes(red_boxes)

        accent_boxes = _extract_color_boxes(answer, "yellow", difference_mask) + _extract_color_boxes(
            answer,
            "orange",
            difference_mask,
        )
        related_accents = [
            accent
            for accent in accent_boxes
            if any(_boxes_are_related(accent, red_box) for red_box in red_boxes)
        ]
        return _merge_overlapping_boxes(red_boxes + related_accents)

    for color_name in ("yellow", "orange"):
        boxes = _extract_color_boxes(answer, color_name, difference_mask)
        if boxes:
            return _merge_overlapping_boxes(boxes)
    return []


def _boxes_are_related(
    lhs: tuple[int, int, int, int],
    rhs: tuple[int, int, int, int],
) -> bool:
    return _box_overlap_ratio(lhs, rhs) > 0.08 or bbox_iou(lhs, rhs) > 0.02


def _is_valid_ground_truth_box(box: tuple[int, int, int, int]) -> bool:
    _, _, width, height = box
    area = width * height
    return width >= 8 and height >= 12 and area >= 120


def _build_difference_mask(original: np.ndarray, answer: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(answer, original)
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(difference_gray, 18, 255, cv2.THRESH_BINARY)
    return cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )


def _extract_color_boxes(
    answer: np.ndarray,
    color_name: str,
    difference_mask: np.ndarray | None,
) -> list[tuple[int, int, int, int]]:
    blue, green, red = cv2.split(answer)
    if color_name == "red":
        mask = (red > 150) & (green < 120) & (blue < 120)
    elif color_name == "yellow":
        mask = (red > 150) & (green > 150) & (blue < 150)
    elif color_name == "orange":
        mask = (red > 120) & (green > 60) & (green < 220) & (blue < 130)
    else:
        raise ValueError(f"未知标注颜色: {color_name}")

    mask_image = mask.astype(np.uint8) * 255
    if difference_mask is not None:
        mask_image = cv2.bitwise_and(mask_image, difference_mask)
    mask_image = cv2.morphologyEx(
        mask_image,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask_image, 8)
    boxes: list[tuple[int, int, int, int]] = []
    image_height, image_width = answer.shape[:2]
    image_area = float(image_height * image_width)
    for component_id in range(1, component_count):
        x, y, w, h, area = stats[component_id]
        if area < 50 or w < 5 or h < 5:
            continue
        if area > image_area * 0.25:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _map_box_to_original(
    box: tuple[int, int, int, int],
    inverse_matrix: np.ndarray,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    corners = np.float32(
        [
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h],
        ]
    ).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, inverse_matrix).reshape(-1, 2)
    x0, y0 = mapped.min(axis=0)
    x1, y1 = mapped.max(axis=0)
    return (
        int(round(x0)),
        int(round(y0)),
        int(round(x1 - x0)),
        int(round(y1 - y0)),
    )


def _clip_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    image_height, image_width = image_shape[:2]
    x0 = max(0, min(image_width - 1, x))
    y0 = max(0, min(image_height - 1, y))
    x1 = max(0, min(image_width, x + w))
    y1 = max(0, min(image_height, y + h))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        merged_index = None
        for index, existing in enumerate(merged):
            if _box_overlap_ratio(existing, box) > 0.55 or bbox_iou(existing, box) > 0.18:
                merged_index = index
                break
        if merged_index is None:
            merged.append(box)
        else:
            merged[merged_index] = _union_boxes(merged[merged_index], box)
    return sorted(merged, key=lambda item: (item[1], item[0]))


def _box_overlap_ratio(
    lhs: tuple[int, int, int, int],
    rhs: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = lhs
    bx, by, bw, bh = rhs
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    intersection = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    return intersection / max(min(aw * ah, bw * bh), 1)


def _union_boxes(
    lhs: tuple[int, int, int, int],
    rhs: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = lhs
    bx, by, bw, bh = rhs
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0


def _match_boxes(
    gt_boxes: list[tuple[int, int, int, int]],
    detection_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[dict[str, object]]:
    used_detections: set[int] = set()
    matches: list[dict[str, object]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        best_detection_index = -1
        best_iou = 0.0
        for detection_index, detection_box in enumerate(detection_boxes):
            if detection_index in used_detections:
                continue
            current_iou = bbox_iou(gt_box, detection_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_detection_index = detection_index
        matched = best_iou >= iou_threshold and best_detection_index >= 0
        if matched:
            used_detections.add(best_detection_index)
        matches.append(
            {
                "ground_truth_index": gt_index,
                "ground_truth_box": list(gt_box),
                "detection_index": best_detection_index if matched else None,
                "best_iou": round(best_iou, 4),
                "matched": matched,
            }
        )
    return matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="标准答案驱动的篡改检测评测")
    parser.add_argument("--data-dir", default="./data", help="样例数据目录")
    parser.add_argument("--report", default="./evaluation_report.json", help="评测报告输出路径")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="命中判定 IoU 阈值")
    parser.add_argument("--max-detections", type=int, default=8, help="每张图最多检测框数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_dataset(
        data_dir=args.data_dir,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )
    report_path = Path(args.report)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("评测完成")
    print(f"图片数量: {report['image_count']}")
    print(f"标准框数量: {report['ground_truth_count']}")
    print(f"召回率: {report['recall']:.4f}")
    print(f"平均最佳 IoU: {report['average_best_iou']:.4f}")
    print(f"评测报告: {report_path}")


if __name__ == "__main__":
    main()
