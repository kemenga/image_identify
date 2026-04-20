from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from document_detector import DetectionResult as DocumentDetectionResult
from document_detector import TamperRegion as DocumentTamperRegion
from document_detector import TraditionalTamperDetector
from preprocessing import load_image


@dataclass(slots=True)
class TamperRegion:
    detection_type: str
    label: str
    bbox: tuple[int, int, int, int]
    score: float
    detail: dict[str, object]
    line_index: int = -1
    char_indices: list[int] | None = None


@dataclass(slots=True)
class DetectionResult:
    status: str
    reason: str | None
    detections: list[TamperRegion]
    candidate_regions: list[TamperRegion]
    evidence: dict[str, float]
    evidence_artifacts: dict[str, str]


@dataclass(slots=True)
class EvidenceBundle:
    ela_map: np.ndarray
    noise_map: np.ndarray
    ela_score: float
    noise_score: float
    document_score: float


class UniversalTamperDetector:
    """文档优先的通用图像篡改检测器。"""

    GLOBAL_EVIDENCE_WEIGHT = 0.85
    LOCAL_ELA_WEIGHT = 1.6
    LOCAL_NOISE_WEIGHT = 1.2
    MAX_REPORT_CANDIDATES = 8

    def __init__(self, max_output_detections: int = 5):
        self.max_output_detections = max_output_detections
        self.document_detector = TraditionalTamperDetector()

    def detect(
        self,
        image_path: str,
        output_path: str | None = None,
        report_path: str | None = None,
        max_detections: int = 5,
        evidence_output_dir: str | None = None,
    ) -> DetectionResult:
        image, gray = load_image(image_path)
        document_result = self.document_detector.detect(gray, image=image)
        evidence_bundle = self._build_evidence_bundle(
            image_path=image_path,
            image=image,
            gray=gray,
            document_result=document_result,
        )

        enriched_candidates = [
            self._convert_region(region, evidence_bundle)
            for region in document_result.candidate_regions
        ]
        enriched_detections = self._build_final_detections(
            document_result=document_result,
            candidates=enriched_candidates,
            image_shape=gray.shape,
            max_detections=max_detections,
        )
        status, reason = self._resolve_status(
            document_result=document_result,
            detections=enriched_detections,
            candidates=enriched_candidates,
        )
        evidence_artifacts = self._export_evidence_artifacts(
            evidence_bundle=evidence_bundle,
            output_dir=evidence_output_dir,
        )

        result = DetectionResult(
            status=status,
            reason=reason,
            detections=enriched_detections,
            candidate_regions=enriched_candidates,
            evidence={
                "ela_score": round(evidence_bundle.ela_score, 4),
                "noise_score": round(evidence_bundle.noise_score, 4),
                "document_score": round(evidence_bundle.document_score, 4),
            },
            evidence_artifacts=evidence_artifacts,
        )

        if output_path:
            cv2.imwrite(output_path, visualize_detection(image, result))
        if report_path:
            Path(report_path).write_text(
                json.dumps(build_report(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return result

    def _export_evidence_artifacts(
        self,
        evidence_bundle: EvidenceBundle,
        output_dir: str | None,
    ) -> dict[str, str]:
        if output_dir is None:
            return {}

        artifact_dir = Path(output_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        fused_map = (
            self.LOCAL_ELA_WEIGHT * evidence_bundle.ela_map
            + self.LOCAL_NOISE_WEIGHT * evidence_bundle.noise_map
        ) / (self.LOCAL_ELA_WEIGHT + self.LOCAL_NOISE_WEIGHT)
        maps = {
            "ela_heatmap": evidence_bundle.ela_map,
            "noise_heatmap": evidence_bundle.noise_map,
            "fused_heatmap": fused_map,
        }

        artifacts: dict[str, str] = {}
        for name, score_map in maps.items():
            output_path = artifact_dir / f"{name}.png"
            heatmap = self._evidence_map_to_heatmap(score_map)
            if not cv2.imwrite(str(output_path), heatmap):
                raise OSError(f"无法写出证据热力图: {output_path}")
            artifacts[name] = str(output_path)
        return artifacts

    @staticmethod
    def _evidence_map_to_heatmap(score_map: np.ndarray) -> np.ndarray:
        # 统一清理异常值并重新归一化，让不同证据图都能直观看到相对热点。
        clean_map = np.nan_to_num(
            score_map.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        clean_map = np.clip(clean_map, 0.0, None)
        min_value = float(clean_map.min()) if clean_map.size else 0.0
        max_value = float(clean_map.max()) if clean_map.size else 0.0
        if max_value > min_value:
            normalized = (clean_map - min_value) / (max_value - min_value)
        else:
            normalized = np.zeros_like(clean_map, dtype=np.float32)
        heatmap_gray = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

    def _build_evidence_bundle(
        self,
        image_path: str,
        image: np.ndarray,
        gray: np.ndarray,
        document_result: DocumentDetectionResult,
    ) -> EvidenceBundle:
        ela_map, ela_score = self._ela_evidence(image_path)
        noise_map, noise_score = self._noise_evidence(gray)
        top_document_score = 0.0
        if document_result.candidate_regions:
            top_document_score = max(region.score for region in document_result.candidate_regions[:8])
        document_score = top_document_score / (top_document_score + 8.0) if top_document_score > 0 else 0.0
        if ela_map.shape != gray.shape:
            ela_map = cv2.resize(ela_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        if noise_map.shape != gray.shape:
            noise_map = cv2.resize(noise_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        return EvidenceBundle(
            ela_map=ela_map,
            noise_map=noise_map,
            ela_score=float(ela_score),
            noise_score=float(noise_score),
            document_score=float(document_score),
        )

    def _ela_evidence(self, image_path: str) -> tuple[np.ndarray, float]:
        pil_image = Image.open(image_path).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_file:
            pil_image.save(temp_file.name, "JPEG", quality=90)
            compressed = Image.open(temp_file.name).convert("RGB")
            ela_image = ImageChops.difference(pil_image, compressed)

        extrema = ela_image.getextrema()
        max_diff = max(channel[1] for channel in extrema)
        max_diff = max(max_diff, 1)
        ela_image = ImageEnhance.Brightness(ela_image).enhance((255.0 / max_diff) * (15.0 / 255.0))
        ela_gray = cv2.cvtColor(np.array(ela_image), cv2.COLOR_RGB2GRAY).astype(np.float32)
        ela_norm = cv2.normalize(ela_gray, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return ela_norm, float(np.mean(ela_norm))

    def _noise_evidence(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred).astype(np.float32)
        local_mean = cv2.blur(noise, (15, 15))
        local_sq_mean = cv2.blur(noise ** 2, (15, 15))
        local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
        variance_norm = cv2.normalize(local_variance, None, 0.0, 1.0, cv2.NORM_MINMAX)
        noise_score = np.std(local_variance) / (np.mean(local_variance) + 1e-6)
        return variance_norm, float(min(noise_score / 10.0, 1.0))

    def _convert_region(
        self,
        region: DocumentTamperRegion,
        evidence_bundle: EvidenceBundle,
    ) -> TamperRegion:
        local_ela = self._bbox_score(evidence_bundle.ela_map, region.bbox)
        local_noise = self._bbox_score(evidence_bundle.noise_map, region.bbox)
        fused_score = (
            float(region.score)
            + self.LOCAL_ELA_WEIGHT * local_ela
            + self.LOCAL_NOISE_WEIGHT * local_noise
            + self.GLOBAL_EVIDENCE_WEIGHT * (evidence_bundle.ela_score + evidence_bundle.noise_score)
        )
        detail = dict(region.detail)
        detail["raw_document_score"] = round(float(region.score), 4)
        detail["fused_score"] = round(float(fused_score), 4)
        detail["evidence"] = {
            "local_ela": round(local_ela, 4),
            "local_noise": round(local_noise, 4),
            "global_ela": round(evidence_bundle.ela_score, 4),
            "global_noise": round(evidence_bundle.noise_score, 4),
            "document_score": round(evidence_bundle.document_score, 4),
        }
        return TamperRegion(
            detection_type=region.detection_type,
            label=region.label,
            bbox=tuple(int(value) for value in region.bbox),
            score=float(fused_score),
            detail=detail,
            line_index=region.line_index,
            char_indices=list(region.char_indices),
        )

    def _build_final_detections(
        self,
        document_result: DocumentDetectionResult,
        candidates: list[TamperRegion],
        image_shape: tuple[int, int],
        max_detections: int,
    ) -> list[TamperRegion]:
        enriched_by_key = {
            self._region_key(candidate): candidate
            for candidate in candidates
        }
        ordered: list[TamperRegion] = []
        for region in document_result.detections:
            candidate = enriched_by_key.get(self._region_key(region))
            if candidate is not None and not self._is_duplicate(candidate, ordered):
                ordered.append(candidate)

        if not ordered:
            for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
                if self._is_duplicate(candidate, ordered):
                    continue
                ordered.append(candidate)
                if len(ordered) >= max_detections:
                    break

        photo_candidate = self._select_photo_candidate(candidates, image_shape)
        if photo_candidate is not None and not self._is_duplicate(photo_candidate, ordered):
            if len(ordered) < max_detections:
                ordered.append(photo_candidate)
            else:
                replace_index = self._find_replaceable_index(ordered)
                ordered[replace_index] = photo_candidate

        return ordered[:max_detections]

    def _resolve_status(
        self,
        document_result: DocumentDetectionResult,
        detections: list[TamperRegion],
        candidates: list[TamperRegion],
    ) -> tuple[str, str | None]:
        if detections:
            return "detected", None
        if document_result.status == "insufficient_context":
            return "insufficient_context", document_result.reason
        if candidates:
            return "no_detection", "未发现稳定篡改证据"
        return "no_detection", document_result.reason or "未发现可用候选区域"

    def _select_photo_candidate(
        self,
        candidates: list[TamperRegion],
        image_shape: tuple[int, int],
    ) -> TamperRegion | None:
        image_height, image_width = image_shape
        image_area = float(image_height * image_width)
        best: TamperRegion | None = None
        best_rank = -1.0
        for candidate in candidates:
            x, y, w, h = candidate.bbox
            area = w * h
            aspect = w / max(h, 1)
            if candidate.detection_type != "region_anomaly":
                continue
            if area < image_area * 0.045:
                continue
            if x < image_width * 0.45:
                continue
            if y > image_height * 0.45:
                continue
            if not 0.55 <= aspect <= 1.15:
                continue
            if "zone" in candidate.detail:
                continue
            rank = candidate.score + area / max(image_area, 1.0) * 20.0
            if rank > best_rank:
                best = candidate
                best_rank = rank
        return best

    def _find_replaceable_index(self, detections: list[TamperRegion]) -> int:
        for index in range(len(detections) - 1, -1, -1):
            if detections[index].detection_type == "region_anomaly":
                return index
        return len(detections) - 1

    @staticmethod
    def _region_key(region: DocumentTamperRegion | TamperRegion) -> tuple[str, tuple[int, int, int, int], str]:
        return region.detection_type, tuple(int(value) for value in region.bbox), region.label

    @staticmethod
    def _bbox_score(score_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        x, y, w, h = bbox
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(score_map.shape[1], x + w)
        y1 = min(score_map.shape[0], y + h)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        patch = score_map[y0:y1, x0:x1]
        return float(np.mean(patch))

    def _is_duplicate(self, candidate: TamperRegion, existing: list[TamperRegion]) -> bool:
        return any(self._bbox_iou(candidate.bbox, item.bbox) >= 0.35 for item in existing)

    @staticmethod
    def _bbox_iou(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> float:
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


def visualize_detection(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()
    palette = [
        (0, 255, 255),
        (0, 165, 255),
        (255, 128, 0),
        (0, 220, 0),
        (255, 0, 0),
    ]

    for index, region in enumerate(result.detections):
        x, y, w, h = region.bbox
        color = palette[index % len(palette)]
        cv2.rectangle(overlay, (max(0, x - 6), max(0, y - 6)), (x + w + 6, y + h + 6), color, -1)
        cv2.rectangle(output, (max(0, x - 6), max(0, y - 6)), (x + w + 6, y + h + 6), color, 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{region.label} {region.score:.2f}"
        text_origin = (x, max(18, y - 10))
        cv2.putText(output, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.18, output, 0.82, 0)


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_report(result: DetectionResult) -> dict[str, object]:
    return {
        "status": result.status,
        "reason": result.reason,
        "detections": [
            {
                "type": detection.detection_type,
                "label": detection.label,
                "line_index": detection.line_index,
                "bbox": list(detection.bbox),
                "char_indices": detection.char_indices or [],
                "score": round(detection.score, 4),
                "detail": _to_builtin(detection.detail),
            }
            for detection in result.detections
        ],
        "candidate_count": len(result.candidate_regions),
        "top_candidates": [
            {
                "type": candidate.detection_type,
                "label": candidate.label,
                "line_index": candidate.line_index,
                "bbox": list(candidate.bbox),
                "score": round(candidate.score, 4),
            }
            for candidate in sorted(
                result.candidate_regions,
                key=lambda item: item.score,
                reverse=True,
            )[: UniversalTamperDetector.MAX_REPORT_CANDIDATES]
        ],
        "evidence": result.evidence,
        "evidence_artifacts": result.evidence_artifacts,
    }


def detect_image_tamper(
    image_path: str,
    output_path: str | None = None,
    report_path: str | None = None,
    max_detections: int = 5,
    evidence_output_dir: str | None = None,
) -> DetectionResult:
    detector = UniversalTamperDetector(max_output_detections=max_detections)
    return detector.detect(
        image_path=image_path,
        output_path=output_path,
        report_path=report_path,
        max_detections=max_detections,
        evidence_output_dir=evidence_output_dir,
    )
