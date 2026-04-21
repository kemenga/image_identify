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
    threshold_score_mode: str
    threshold_score_value: float


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
    LOCAL_NOISE_WEIGHT = 12
    MAX_REPORT_CANDIDATES = 8
    REPORT_CONFIDENCE_THRESHOLD = 0.0

    def __init__(
        self,
        max_output_detections: int = 5,
        detector_overrides: dict[str, object] | None = None,
        document_detector_overrides: dict[str, object] | None = None,
    ):
        self.max_output_detections = max_output_detections
        self.single_threshold_mode = bool(
            detector_overrides
            and "REPORT_CONFIDENCE_THRESHOLD" in detector_overrides
        )
        if detector_overrides:
            for key, value in detector_overrides.items():
                setattr(self, key, value)
        self.document_detector = TraditionalTamperDetector(overrides=document_detector_overrides)

    @classmethod
    def tunable_defaults(cls) -> dict[str, object]:
        return {
            "GLOBAL_EVIDENCE_WEIGHT": cls.GLOBAL_EVIDENCE_WEIGHT,
            "LOCAL_ELA_WEIGHT": cls.LOCAL_ELA_WEIGHT,
            "LOCAL_NOISE_WEIGHT": cls.LOCAL_NOISE_WEIGHT,
            "MAX_REPORT_CANDIDATES": cls.MAX_REPORT_CANDIDATES,
            "REPORT_CONFIDENCE_THRESHOLD": cls.REPORT_CONFIDENCE_THRESHOLD,
        }

    @classmethod
    def document_tunable_defaults(cls) -> dict[str, object]:
        return TraditionalTamperDetector.tunable_defaults()

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
        enriched_candidates.extend(
            self._build_evidence_text_noise_candidates(
                gray=gray,
                evidence_bundle=evidence_bundle,
                candidates=enriched_candidates,
            )
        )
        enriched_candidates.extend(
            self._build_invoice_field_candidates(
                gray=gray,
                evidence_bundle=evidence_bundle,
                candidates=enriched_candidates,
            )
        )
        enriched_detections = self._build_final_detections(
            document_result=document_result,
            candidates=enriched_candidates,
            image_shape=gray.shape,
            max_detections=max_detections,
        )
        threshold_score_mode = "single_threshold" if self.single_threshold_mode else "report_confidence"
        for detection in enriched_detections:
            detection.detail["threshold_score"] = round(
                self._threshold_score_for_detection(detection, enriched_candidates),
                4,
            )
            detection.detail["threshold_score_mode"] = threshold_score_mode
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
            threshold_score_mode=threshold_score_mode,
            threshold_score_value=round(float(self.REPORT_CONFIDENCE_THRESHOLD), 4),
        )

        if output_path:
            cv2.imwrite(output_path, visualize_detection(image, result))
        if report_path:
            Path(report_path).write_text(
                json.dumps(
                    build_report(
                        result,
                        max_report_candidates=int(self.MAX_REPORT_CANDIDATES),
                        min_report_confidence=float(self.REPORT_CONFIDENCE_THRESHOLD),
                        single_threshold_mode=self.single_threshold_mode,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
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
        fused_map = self._build_fused_evidence_map(evidence_bundle)
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

    def _build_fused_evidence_map(self, evidence_bundle: EvidenceBundle) -> np.ndarray:
        return (
            self.LOCAL_ELA_WEIGHT * evidence_bundle.ela_map
            + self.LOCAL_NOISE_WEIGHT * evidence_bundle.noise_map
        ) / (self.LOCAL_ELA_WEIGHT + self.LOCAL_NOISE_WEIGHT)

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

    def _build_evidence_text_noise_candidates(
        self,
        gray: np.ndarray,
        evidence_bundle: EvidenceBundle,
        candidates: list[TamperRegion],
    ) -> list[TamperRegion]:
        # 只在既有文本块内部做细化，避免把全图噪声热点误当成文字篡改。
        fused_map = self._build_fused_evidence_map(evidence_bundle)
        component_boxes = self.document_detector._text_component_boxes(gray)
        refined_candidates: list[TamperRegion] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if candidate.detection_type != "text_region":
                continue
            zone = str(candidate.detail.get("zone", ""))
            if zone and zone != "generic_text_block":
                continue

            for refined_box, support_count, evidence_score, contrast in self._find_noisy_component_clusters_in_text_box(
                source_bbox=candidate.bbox,
                component_boxes=component_boxes,
                ela_map=evidence_bundle.ela_map,
                noise_map=evidence_bundle.noise_map,
                fused_map=fused_map,
            ):
                if any(self._is_duplicate_box(refined_box, item.bbox) for item in refined_candidates):
                    continue

                local_ela = self._bbox_score(evidence_bundle.ela_map, refined_box)
                local_noise = self._bbox_score(evidence_bundle.noise_map, refined_box)
                score = (
                    5.1
                    + 1.6 * evidence_score
                    + 0.55 * min(contrast, 6.0)
                    + 1.25 * local_noise
                    + 0.75 * local_ela
                    + 0.12 * min(support_count, 6)
                    + self.GLOBAL_EVIDENCE_WEIGHT * (evidence_bundle.ela_score + evidence_bundle.noise_score)
                )
                refined_candidates.append(
                    TamperRegion(
                        detection_type="text_noise_anomaly",
                        label="噪声篡改",
                        bbox=refined_box,
                        score=float(score),
                        detail={
                            "raw_document_score": round(float(candidate.score), 4),
                            "fused_score": round(float(score), 4),
                            "source_bbox": list(candidate.bbox),
                            "source_key": self._bbox_key(candidate.bbox),
                            "zone": "evidence_text_noise_refinement",
                            "support_count": support_count,
                            "evidence_window_score": round(float(evidence_score), 4),
                            "evidence_contrast": round(float(contrast), 4),
                            "evidence": {
                                "local_ela": round(local_ela, 4),
                                "local_noise": round(local_noise, 4),
                                "global_ela": round(evidence_bundle.ela_score, 4),
                                "global_noise": round(evidence_bundle.noise_score, 4),
                                "document_score": round(evidence_bundle.document_score, 4),
                            },
                        },
                        line_index=candidate.line_index,
                        char_indices=[],
                    )
                )
                break
            if len(refined_candidates) >= 8:
                break

        return sorted(refined_candidates, key=lambda item: item.score, reverse=True)

    def _threshold_score_for_detection(
        self,
        detection: TamperRegion,
        candidates: list[TamperRegion],
    ) -> float:
        best_time_candidate = self._best_candidate_of_type(candidates, "time_group")
        best_digit_candidate = self._best_candidate_of_type(candidates, "digit_window")
        has_strong_refinement = any(
            self._is_strong_evidence_refinement(candidate)
            for candidate in candidates
            if not self._is_special_scene_conflicting_refinement(
                candidate,
                best_time_candidate,
                best_digit_candidate,
            )
        )
        if self.single_threshold_mode:
            return self._single_threshold_score(
                detection,
                has_strong_refinement,
                best_time_candidate,
                best_digit_candidate,
            )
        return self._report_confidence(detection)

    def _find_noisy_component_clusters_in_text_box(
        self,
        source_bbox: tuple[int, int, int, int],
        component_boxes: list[tuple[int, int, int, int]],
        ela_map: np.ndarray,
        noise_map: np.ndarray,
        fused_map: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], int, float, float]]:
        x, y, w, h = source_bbox
        if w < 80 or h < 14 or h > 90 or w < h * 2.0:
            return []

        inside_components = [
            box
            for box in component_boxes
            if x <= self._center_x(box) <= x + w
            and y <= self._center_y(box) <= y + h
        ]
        if len(inside_components) < 2:
            return []

        component_metrics: list[dict[str, object]] = []
        for box in inside_components:
            local_ela = self._bbox_score(ela_map, box)
            local_noise = self._bbox_score(noise_map, box)
            local_fused = self._bbox_score(fused_map, box)
            component_metrics.append(
                {
                    "bbox": box,
                    "local_ela": local_ela,
                    "local_noise": local_noise,
                    "local_fused": local_fused,
                }
            )

        noise_scores = self._robust_1d_scores([float(item["local_noise"]) for item in component_metrics])
        ela_scores = self._robust_1d_scores([float(item["local_ela"]) for item in component_metrics])
        fused_values = [float(item["local_fused"]) for item in component_metrics]
        fused_median = float(np.median(fused_values))
        ela_median = float(np.median([float(item["local_ela"]) for item in component_metrics]))
        noise_median = float(np.median([float(item["local_noise"]) for item in component_metrics]))

        ranked_items: list[dict[str, object]] = []
        for item, noise_score, ela_score in zip(component_metrics, noise_scores, ela_scores):
            mix_score = 0.75 * noise_score + 0.25 * ela_score
            ranked_items.append(
                {
                    **item,
                    "noise_score": noise_score,
                    "ela_score": ela_score,
                    "mix_score": mix_score,
                }
            )
        ranked_items.sort(key=lambda item: item["bbox"][0])

        hot_items = [
            item
            for item in ranked_items
            if item["mix_score"] >= 1.4
            or item["noise_score"] >= 1.6
            or item["ela_score"] >= 1.8
        ]
        if not hot_items:
            return []

        seed_clusters: list[list[dict[str, object]]] = []
        for item in hot_items:
            if not seed_clusters:
                seed_clusters.append([item])
                continue
            prev_box = seed_clusters[-1][-1]["bbox"]
            current_box = item["bbox"]
            horizontal_gap = current_box[0] - (prev_box[0] + prev_box[2])
            y_overlap = min(prev_box[1] + prev_box[3], current_box[1] + current_box[3]) - max(prev_box[1], current_box[1])
            if horizontal_gap <= 18 and y_overlap >= min(prev_box[3], current_box[3]) * 0.2:
                seed_clusters[-1].append(item)
            else:
                seed_clusters.append([item])

        results: list[tuple[tuple[int, int, int, int], int, float, float]] = []
        for cluster in seed_clusters:
            peak_mix_score = max(float(item["mix_score"]) for item in cluster)
            core_cluster = [
                item
                for item in cluster
                if float(item["mix_score"]) >= peak_mix_score * 0.70
            ]
            if len(core_cluster) < 2:
                continue

            expanded_cluster = list(core_cluster)
            cluster_boxes = [item["bbox"] for item in expanded_cluster]
            cluster_bbox = self._union_boxes_many(cluster_boxes)
            changed = True
            while changed:
                changed = False
                for item in ranked_items:
                    if item in expanded_cluster:
                        continue
                    box = item["bbox"]
                    if box[0] + box[2] < cluster_bbox[0]:
                        continue
                    horizontal_gap = max(
                        box[0] - (cluster_bbox[0] + cluster_bbox[2]),
                        cluster_bbox[0] - (box[0] + box[2]),
                        0,
                    )
                    y_overlap = min(cluster_bbox[1] + cluster_bbox[3], box[1] + box[3]) - max(cluster_bbox[1], box[1])
                    if horizontal_gap > 18:
                        continue
                    if y_overlap < min(cluster_bbox[3], box[3]) * 0.2:
                        continue
                    if (
                        item["local_fused"] >= fused_median
                        or item["local_ela"] >= ela_median
                        or item["local_noise"] >= noise_median
                    ):
                        expanded_cluster.append(item)
                        cluster_bbox = self._union_boxes_many([cluster_bbox, box])
                        changed = True

            if len(expanded_cluster) < 2:
                continue

            cluster_bbox = self._union_boxes_many([item["bbox"] for item in expanded_cluster])
            padded_bbox = self._pad_bbox_within(cluster_bbox, source_bbox, pad_x=6, pad_y=6)
            if padded_bbox[2] < max(24, int(h * 0.7)):
                continue
            if padded_bbox[2] > max(int(w * 0.45), 90):
                continue

            max_mix_score = max(float(item["mix_score"]) for item in expanded_cluster)
            max_noise = max(float(item["local_noise"]) for item in expanded_cluster)
            evidence_score = min(
                1.0,
                0.12 * max_mix_score + 0.45 * max_noise + 0.03 * min(len(expanded_cluster), 6),
            )
            results.append(
                (
                    padded_bbox,
                    len(expanded_cluster),
                    evidence_score,
                    max_mix_score,
                )
            )

        return sorted(results, key=lambda item: (item[3], item[2], item[1]), reverse=True)[:2]

    def _build_invoice_field_candidates(
        self,
        gray: np.ndarray,
        evidence_bundle: EvidenceBundle,
        candidates: list[TamperRegion],
    ) -> list[TamperRegion]:
        if not self._looks_like_invoice(gray):
            return []

        component_boxes = [
            box
            for box in self.document_detector._text_component_boxes(gray)
            if box[3] >= 35 or box[2] * box[3] >= 700
        ]
        results: list[TamperRegion] = []
        for row_boxes in self._group_boxes_by_row(component_boxes, row_tolerance=42.0):
            for cluster in self._cluster_boxes_by_gap(row_boxes, max_gap=28):
                if len(cluster) < 2:
                    continue
                cluster_bbox = self._union_boxes_many(cluster)
                if (
                    cluster_bbox[2] < 80
                    or cluster_bbox[2] > 380
                    or cluster_bbox[3] < 40
                    or cluster_bbox[3] > 120
                ):
                    continue
                if cluster_bbox[1] < gray.shape[0] * 0.22:
                    continue
                results.append(
                    self._build_invoice_text_candidate(
                        bbox=cluster_bbox,
                        evidence_bundle=evidence_bundle,
                        support_count=len(cluster),
                        zone="invoice_large_text_field",
                    )
                )

        header_boxes = [
            candidate.bbox
            for candidate in candidates
            if candidate.detection_type == "text_region"
            and str(candidate.detail.get("zone", "")) == "generic_text_block"
            and int(candidate.detail.get("support_count", 0)) >= 8
            and candidate.bbox[1] < gray.shape[0] * 0.2
            and gray.shape[1] * 0.35 <= candidate.bbox[0] <= gray.shape[1] * 0.62
        ]
        if header_boxes:
            results.append(
                self._build_invoice_text_candidate(
                    bbox=self._union_boxes_many(header_boxes),
                    evidence_bundle=evidence_bundle,
                    support_count=len(header_boxes),
                    zone="invoice_header_stamp",
                )
            )

        results = self._merge_invoice_field_candidates(results)
        deduped: list[TamperRegion] = []
        for candidate in sorted(results, key=lambda item: item.score, reverse=True):
            if self._is_duplicate(candidate, deduped):
                continue
            deduped.append(candidate)
        return deduped

    def _build_invoice_text_candidate(
        self,
        bbox: tuple[int, int, int, int],
        evidence_bundle: EvidenceBundle,
        support_count: int,
        zone: str,
    ) -> TamperRegion:
        local_ela = self._bbox_score(evidence_bundle.ela_map, bbox)
        local_noise = self._bbox_score(evidence_bundle.noise_map, bbox)
        score = (
            7.2
            + 0.9 * local_noise
            + 0.45 * local_ela
            + 0.15 * min(support_count, 10)
            + self.GLOBAL_EVIDENCE_WEIGHT * (evidence_bundle.ela_score + evidence_bundle.noise_score)
        )
        return TamperRegion(
            detection_type="text_region",
            label="文字篡改",
            bbox=bbox,
            score=float(score),
            detail={
                "raw_document_score": round(float(score), 4),
                "fused_score": round(float(score), 4),
                "support_count": support_count,
                "zone": zone,
                "evidence": {
                    "local_ela": round(local_ela, 4),
                    "local_noise": round(local_noise, 4),
                    "global_ela": round(evidence_bundle.ela_score, 4),
                    "global_noise": round(evidence_bundle.noise_score, 4),
                    "document_score": round(evidence_bundle.document_score, 4),
                },
            },
            line_index=-1,
            char_indices=[],
        )

    def _merge_invoice_field_candidates(
        self,
        candidates: list[TamperRegion],
    ) -> list[TamperRegion]:
        merged: list[TamperRegion] = []
        for candidate in sorted(candidates, key=lambda item: item.bbox[0]):
            if str(candidate.detail.get("zone", "")) != "invoice_large_text_field":
                merged.append(candidate)
                continue
            merged_index = None
            for index, existing in enumerate(merged):
                if str(existing.detail.get("zone", "")) != "invoice_large_text_field":
                    continue
                same_row = abs((existing.bbox[1] + existing.bbox[3] / 2.0) - (candidate.bbox[1] + candidate.bbox[3] / 2.0)) <= 48
                horizontal_gap = max(
                    candidate.bbox[0] - (existing.bbox[0] + existing.bbox[2]),
                    existing.bbox[0] - (candidate.bbox[0] + candidate.bbox[2]),
                    0,
                )
                if same_row and horizontal_gap <= 90:
                    merged_index = index
                    break
            if merged_index is None:
                merged.append(candidate)
                continue

            existing = merged[merged_index]
            merged[merged_index] = TamperRegion(
                detection_type="text_region",
                label="文字篡改",
                bbox=self._union_boxes(existing.bbox, candidate.bbox),
                score=max(existing.score, candidate.score) + 0.05,
                detail={
                    **existing.detail,
                    "support_count": int(existing.detail.get("support_count", 1))
                    + int(candidate.detail.get("support_count", 1)),
                    "fused_score": round(max(existing.score, candidate.score) + 0.05, 4),
                },
                line_index=-1,
                char_indices=[],
            )
        return merged

    @staticmethod
    def _looks_like_invoice(gray: np.ndarray) -> bool:
        return gray.shape[1] >= 2000 and gray.shape[0] >= 1200

    @staticmethod
    def _group_boxes_by_row(
        boxes: list[tuple[int, int, int, int]],
        row_tolerance: float,
    ) -> list[list[tuple[int, int, int, int]]]:
        rows: list[list[tuple[int, int, int, int]]] = []
        row_centers: list[float] = []
        for box in sorted(boxes, key=lambda item: item[1] + item[3] / 2.0):
            center_y = box[1] + box[3] / 2.0
            matched_index = None
            for row_index, row_center in enumerate(row_centers):
                if abs(center_y - row_center) <= row_tolerance:
                    matched_index = row_index
                    break
            if matched_index is None:
                rows.append([box])
                row_centers.append(center_y)
                continue
            rows[matched_index].append(box)
            row_centers[matched_index] = float(
                np.mean([item[1] + item[3] / 2.0 for item in rows[matched_index]])
            )
        return [sorted(row, key=lambda item: item[0]) for row in rows]

    @staticmethod
    def _cluster_boxes_by_gap(
        boxes: list[tuple[int, int, int, int]],
        max_gap: int,
    ) -> list[list[tuple[int, int, int, int]]]:
        if not boxes:
            return []
        clusters: list[list[tuple[int, int, int, int]]] = [[boxes[0]]]
        for box in boxes[1:]:
            previous = clusters[-1][-1]
            gap = box[0] - (previous[0] + previous[2])
            y_overlap = min(previous[1] + previous[3], box[1] + box[3]) - max(previous[1], box[1])
            if gap <= max_gap and y_overlap >= min(previous[3], box[3]) * 0.25:
                clusters[-1].append(box)
            else:
                clusters.append([box])
        return clusters

    @staticmethod
    def _union_boxes_many(
        boxes: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[0] + box[2] for box in boxes)
        y1 = max(box[1] + box[3] for box in boxes)
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def _union_boxes(
        lhs: tuple[int, int, int, int],
        rhs: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x0 = min(lhs[0], rhs[0])
        y0 = min(lhs[1], rhs[1])
        x1 = max(lhs[0] + lhs[2], rhs[0] + rhs[2])
        y1 = max(lhs[1] + lhs[3], rhs[1] + rhs[3])
        return x0, y0, x1 - x0, y1 - y0

    def _build_final_detections(
        self,
        document_result: DocumentDetectionResult,
        candidates: list[TamperRegion],
        image_shape: tuple[int, int],
        max_detections: int,
    ) -> list[TamperRegion]:
        return self._reportable_candidates(
            candidates,
            max_items=max_detections,
            min_confidence=float(self.REPORT_CONFIDENCE_THRESHOLD),
            single_threshold_mode=self.single_threshold_mode,
        )

    @classmethod
    def _reportable_candidates(
        cls,
        candidates: list[TamperRegion],
        max_items: int | None = None,
        min_confidence: float = 0.0,
        single_threshold_mode: bool = False,
    ) -> list[TamperRegion]:
        best_time_candidate = cls._best_candidate_of_type(candidates, "time_group")
        best_digit_candidate = cls._best_candidate_of_type(candidates, "digit_window")
        has_strong_refinement = any(
            cls._is_strong_evidence_refinement(candidate)
            for candidate in candidates
            if not cls._is_special_scene_conflicting_refinement(
                candidate,
                best_time_candidate,
                best_digit_candidate,
            )
        )
        if single_threshold_mode:
            filtered = [
                candidate
                for candidate in candidates
                if cls._is_relaxed_report_candidate(candidate)
            ]
        else:
            filtered = [
                candidate
                for candidate in candidates
                if cls._is_reportable_candidate(candidate, has_strong_refinement)
                and not cls._is_special_scene_conflicting_refinement(
                    candidate,
                    best_time_candidate,
                    best_digit_candidate,
                )
            ]
        selected: list[TamperRegion] = []
        limited_counts: dict[str, int] = {}
        for candidate in sorted(
            filtered,
            key=lambda item: cls._single_threshold_score(
                item,
                has_strong_refinement,
                best_time_candidate,
                best_digit_candidate,
            ) if single_threshold_mode else cls._report_confidence(item),
            reverse=True,
        ):
            if max_items is not None and len(selected) >= max_items:
                break
            confidence = cls._single_threshold_score(
                candidate,
                has_strong_refinement,
                best_time_candidate,
                best_digit_candidate,
            ) if single_threshold_mode else cls._report_confidence(candidate)
            if confidence < min_confidence:
                continue
            limited_key, limited_max = ("", 0) if single_threshold_mode else cls._report_limit(candidate)
            if limited_key and limited_counts.get(limited_key, 0) >= limited_max:
                continue
            if cls._is_duplicate_against(candidate, selected):
                continue

            selected.append(candidate)
            if limited_key:
                limited_counts[limited_key] = limited_counts.get(limited_key, 0) + 1
        return selected

    @classmethod
    def _report_priority(
        cls,
        candidate: TamperRegion,
        has_strong_refinement: bool,
        best_time_candidate: TamperRegion | None,
        best_digit_candidate: TamperRegion | None,
    ) -> int:
        # 旧报告规则只影响排序，不再硬性排除候选，最终数量由阈值和去重共同决定。
        if not cls._is_reportable_candidate(candidate, has_strong_refinement):
            return 0
        if cls._is_special_scene_conflicting_refinement(
            candidate,
            best_time_candidate,
            best_digit_candidate,
        ):
            return 0
        return 1

    @classmethod
    def _single_threshold_score(
        cls,
        candidate: TamperRegion,
        has_strong_refinement: bool,
        best_time_candidate: TamperRegion | None,
        best_digit_candidate: TamperRegion | None,
    ) -> float:
        score = cls._report_confidence(candidate)
        zone = str(candidate.detail.get("zone", ""))
        conflict = cls._is_special_scene_conflicting_refinement(
            candidate,
            best_time_candidate,
            best_digit_candidate,
        )
        strict_ok = cls._is_reportable_candidate(candidate, has_strong_refinement) and not conflict
        is_best_time = (
            best_time_candidate is not None
            and candidate.detection_type == "time_group"
            and candidate.bbox == best_time_candidate.bbox
        )
        is_best_digit = (
            best_digit_candidate is not None
            and candidate.detection_type == "digit_window"
            and candidate.bbox == best_digit_candidate.bbox
        )

        if candidate.detection_type == "time_group":
            score += 35.0 if strict_ok and is_best_time else 0.0 if strict_ok else -35.0
        elif candidate.detection_type == "digit_window":
            score += 35.0 if strict_ok and is_best_digit else 0.0 if strict_ok else -20.0
        elif candidate.detection_type == "text_region":
            if zone.endswith("_precise") or zone in {"invoice_large_text_field", "invoice_header_stamp"}:
                score += 45.0 if strict_ok else -15.0
            else:
                score -= 20.0
        elif candidate.detection_type == "region_anomaly":
            if zone == "embedded_id_photo":
                score += 40.0 if strict_ok else -10.0
            else:
                score -= 25.0
        elif zone == "evidence_text_noise_refinement":
            if strict_ok:
                score += 18.0
            else:
                score -= 10.0
        elif zone == "text_noise_consistency":
            score -= 15.0
        elif strict_ok:
            score += 5.0
        else:
            score -= 10.0

        if conflict:
            score -= 30.0
        return score

    @staticmethod
    def _is_relaxed_report_candidate(candidate: TamperRegion) -> bool:
        x, y, w, h = candidate.bbox
        if w <= 0 or h <= 0:
            return False
        if not np.isfinite(candidate.score):
            return False
        area = w * h
        if candidate.detection_type == "region_anomaly":
            return area >= 300
        return area >= 80

    @classmethod
    def _is_reportable_candidate(
        cls,
        candidate: TamperRegion,
        has_strong_refinement: bool,
    ) -> bool:
        zone = str(candidate.detail.get("zone", ""))
        support_count = int(candidate.detail.get("support_count", len(candidate.char_indices or [])))
        local_noise = cls._detail_float(candidate, "local_noise")
        area = candidate.bbox[2] * candidate.bbox[3]

        if zone == "evidence_text_noise_refinement":
            return cls._evidence_refinement_level(candidate) in {"strong", "medium"}

        if candidate.detection_type == "time_group":
            return (
                not has_strong_refinement
                and support_count >= 5
                and candidate.score >= 8.0
            )

        if candidate.detection_type == "digit_window":
            return candidate.score >= 10.0 and support_count >= 4

        if candidate.detection_type == "text_region":
            return (
                (
                    zone.endswith("_precise")
                    and support_count >= 2
                    and candidate.score >= 5.8
                )
                or (
                    zone == "invoice_large_text_field"
                    and support_count >= 2
                    and candidate.score >= 7.0
                )
                or (
                    zone == "invoice_header_stamp"
                    and support_count >= 1
                    and candidate.score >= 7.0
                )
            )

        if candidate.detection_type == "region_anomaly":
            if zone == "embedded_id_photo":
                return True
            return area >= 50000 and support_count >= 4 and candidate.score >= 10.0

        if candidate.detection_type == "text_noise_anomaly":
            return (
                zone == "text_noise_consistency"
                and support_count >= 2
                and local_noise >= 0.25
                and candidate.score >= 8.5
            )

        return False

    @classmethod
    def _report_confidence(cls, candidate: TamperRegion) -> float:
        zone = str(candidate.detail.get("zone", ""))
        support_count = int(candidate.detail.get("support_count", len(candidate.char_indices or [])))
        local_noise = cls._detail_float(candidate, "local_noise")
        confidence = float(candidate.score)
        area = max(candidate.bbox[2] * candidate.bbox[3], 0)

        if zone == "evidence_text_noise_refinement":
            # 单阈值模式下更希望优先保留时间框、数字框、证件精确文本等主框，
            # 因此这里不再给文字噪声细化框过高的基础报告分，避免高阈值时被它反压。
            confidence += 52.0
            confidence += 2.0 * cls._plain_detail_float(candidate, "evidence_contrast")
            confidence += 7.0 * cls._plain_detail_float(candidate, "evidence_window_score")
            confidence += min(support_count, 4) * 0.8
            confidence -= min((candidate.bbox[2] * candidate.bbox[3]) / 1800.0, 8.0)
        elif candidate.detection_type == "digit_window":
            confidence += 78.0
        elif candidate.detection_type == "time_group":
            confidence += 77.0
        elif candidate.detection_type == "region_anomaly":
            if zone == "embedded_id_photo":
                confidence += 74.0
            else:
                confidence += 32.0
                confidence += min(support_count, 8) * 0.2
                if area <= 4000:
                    confidence -= 18.0
                elif area <= 16000:
                    confidence -= 12.0
                elif area >= 80000:
                    confidence -= 8.0
        elif candidate.detection_type == "text_region" and zone.endswith("_precise"):
            confidence += 70.0
            confidence += min(support_count, 8) * 0.35
        elif zone == "invoice_header_stamp":
            confidence += 57.0 + min(support_count, 4) * 0.4
        elif zone == "invoice_large_text_field":
            confidence += 53.0 + min(support_count, 8) * 0.25
        elif candidate.detection_type == "text_noise_anomaly":
            confidence += 32.0 + 4.0 * local_noise

        return confidence

    @staticmethod
    def _report_limit(candidate: TamperRegion) -> tuple[str, int]:
        zone = str(candidate.detail.get("zone", ""))
        if zone == "evidence_text_noise_refinement":
            return zone, 2
        if candidate.detection_type in {"digit_window", "time_group", "region_anomaly"}:
            return candidate.detection_type, 1
        if candidate.detection_type == "text_noise_anomaly" and zone == "text_noise_consistency":
            return zone, 1
        return "", 0

    @classmethod
    def _is_strong_evidence_refinement(cls, candidate: TamperRegion) -> bool:
        return cls._evidence_refinement_level(candidate) == "strong"

    @classmethod
    def _evidence_refinement_level(cls, candidate: TamperRegion) -> str:
        if str(candidate.detail.get("zone", "")) != "evidence_text_noise_refinement":
            return ""
        contrast = cls._plain_detail_float(candidate, "evidence_contrast")
        evidence_score = cls._plain_detail_float(candidate, "evidence_window_score")
        local_noise = cls._detail_float(candidate, "local_noise")
        support_count = int(candidate.detail.get("support_count", 1))
        if (
            contrast >= 3.0
            and evidence_score >= 0.55
            and support_count >= 3
            and local_noise >= 0.28
        ):
            return "strong"
        if (
            contrast >= 2.6
            and evidence_score >= 0.52
            and support_count >= 6
            and local_noise >= 0.30
        ):
            return "medium"
        return ""

    @classmethod
    def _is_duplicate_against(
        cls,
        candidate: TamperRegion,
        existing: list[TamperRegion],
    ) -> bool:
        return any(
            cls._bbox_iou(candidate.bbox, item.bbox) >= 0.35
            or cls._bbox_overlap_ratio(candidate.bbox, item.bbox) >= 0.72
            for item in existing
        )

    @staticmethod
    def _detail_float(candidate: TamperRegion, key: str) -> float:
        evidence = candidate.detail.get("evidence", {})
        if isinstance(evidence, dict):
            try:
                return float(evidence.get(key, 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _plain_detail_float(candidate: TamperRegion, key: str) -> float:
        try:
            return float(candidate.detail.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _best_candidate_of_type(
        cls,
        candidates: list[TamperRegion],
        detection_type: str,
    ) -> TamperRegion | None:
        matches = [candidate for candidate in candidates if candidate.detection_type == detection_type]
        if not matches:
            return None
        return max(matches, key=cls._report_confidence)

    @classmethod
    def _is_special_scene_conflicting_refinement(
        cls,
        candidate: TamperRegion,
        best_time_candidate: TamperRegion | None,
        best_digit_candidate: TamperRegion | None,
    ) -> bool:
        if str(candidate.detail.get("zone", "")) != "evidence_text_noise_refinement":
            return False
        source_bbox = cls._detail_bbox(candidate, "source_bbox") or candidate.bbox

        if best_time_candidate is not None:
            vertical_overlap = cls._vertical_overlap_ratio(source_bbox, best_time_candidate.bbox)
            if vertical_overlap <= 0.08 and source_bbox[1] + source_bbox[3] <= best_time_candidate.bbox[1] + 24:
                return True
            if cls._bbox_iou(candidate.bbox, best_time_candidate.bbox) > 0.0:
                return True

        if best_digit_candidate is not None:
            vertical_overlap = cls._vertical_overlap_ratio(source_bbox, best_digit_candidate.bbox)
            source_center_y = source_bbox[1] + source_bbox[3] / 2.0
            digit_center_y = best_digit_candidate.bbox[1] + best_digit_candidate.bbox[3] / 2.0
            if vertical_overlap <= 0.08 and abs(source_center_y - digit_center_y) > max(42.0, source_bbox[3] * 1.2):
                return True

        return False

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

    def _is_duplicate_box(self, bbox: tuple[int, int, int, int], existing: tuple[int, int, int, int]) -> bool:
        return self._bbox_iou(bbox, existing) >= 0.35 or self._bbox_overlap_ratio(bbox, existing) >= 0.72

    @staticmethod
    def _bbox_key(bbox: tuple[int, int, int, int]) -> str:
        return ",".join(str(int(value)) for value in bbox)

    @staticmethod
    def _center_x(box: tuple[int, int, int, int]) -> float:
        return box[0] + box[2] / 2.0

    @staticmethod
    def _center_y(box: tuple[int, int, int, int]) -> float:
        return box[1] + box[3] / 2.0

    @staticmethod
    def _robust_1d_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        array = np.asarray(values, dtype=np.float32)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median))) + 1e-6
        return [float((value - median) / mad) for value in array]

    @staticmethod
    def _pad_bbox_within(
        bbox: tuple[int, int, int, int],
        outer_bbox: tuple[int, int, int, int],
        pad_x: int,
        pad_y: int,
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        ox, oy, ow, oh = outer_bbox
        left = max(ox, x - pad_x)
        top = max(oy, y - pad_y)
        right = min(ox + ow, x + w + pad_x)
        bottom = min(oy + oh, y + h + pad_y)
        return int(left), int(top), int(right - left), int(bottom - top)

    @staticmethod
    def _detail_bbox(candidate: TamperRegion, key: str) -> tuple[int, int, int, int] | None:
        value = candidate.detail.get(key)
        if not isinstance(value, list | tuple) or len(value) != 4:
            return None
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _vertical_overlap_ratio(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> float:
        top = max(lhs[1], rhs[1])
        bottom = min(lhs[1] + lhs[3], rhs[1] + rhs[3])
        if top >= bottom:
            return 0.0
        return float(bottom - top) / max(min(lhs[3], rhs[3]), 1.0)

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

    @staticmethod
    def _bbox_overlap_ratio(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = lhs
        bx, by, bw, bh = rhs
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax + aw, bx + bw)
        inter_y2 = min(ay + ah, by + bh)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0
        intersection = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        return intersection / max(min(aw * ah, bw * bh), 1.0)


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
        threshold_score = float(region.detail.get("threshold_score", region.score))
        label = f"{region.label} T={threshold_score:.2f}"
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


def build_report(
    result: DetectionResult,
    max_report_candidates: int | None = None,
    min_report_confidence: float = 0.0,
    single_threshold_mode: bool = False,
) -> dict[str, object]:
    if max_report_candidates is None:
        max_report_candidates = UniversalTamperDetector.MAX_REPORT_CANDIDATES
    reportable_candidates = UniversalTamperDetector._reportable_candidates(
        result.candidate_regions,
        max_items=max_report_candidates,
        min_confidence=min_report_confidence,
        single_threshold_mode=single_threshold_mode,
    )
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
                "report_confidence": round(UniversalTamperDetector._report_confidence(detection), 4),
                "detail": _to_builtin(detection.detail),
            }
            for detection in result.detections
        ],
        "candidate_count": len(result.candidate_regions),
        "reportable_candidate_count": len(reportable_candidates),
        "top_candidates": [
            {
                "type": candidate.detection_type,
                "label": candidate.label,
                "line_index": candidate.line_index,
                "bbox": list(candidate.bbox),
                "score": round(candidate.score, 4),
                "report_confidence": round(UniversalTamperDetector._report_confidence(candidate), 4),
            }
            for candidate in reportable_candidates
        ],
        "evidence": result.evidence,
        "evidence_artifacts": result.evidence_artifacts,
        "threshold_score_mode": result.threshold_score_mode,
        "threshold_score_value": result.threshold_score_value,
    }


def detect_image_tamper(
    image_path: str,
    output_path: str | None = None,
    report_path: str | None = None,
    max_detections: int = 5,
    evidence_output_dir: str | None = None,
    detector_overrides: dict[str, object] | None = None,
    document_detector_overrides: dict[str, object] | None = None,
) -> DetectionResult:
    detector = UniversalTamperDetector(
        max_output_detections=max_detections,
        detector_overrides=detector_overrides,
        document_detector_overrides=document_detector_overrides,
    )
    return detector.detect(
        image_path=image_path,
        output_path=output_path,
        report_path=report_path,
        max_detections=max_detections,
        evidence_output_dir=evidence_output_dir,
    )
