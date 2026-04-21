from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from preprocessing import CharacterCandidate, detect_text_lines, segment_line_characters


def _uniform_lbp(image: np.ndarray) -> np.ndarray:
    """使用 8 邻域 uniform LBP，避免依赖 skimage 的二进制扩展。"""
    image = image.astype(np.uint8, copy=False)
    padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    center = padded[1:-1, 1:-1]
    neighbors = np.stack(
        [
            padded[:-2, :-2],
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, 2:],
            padded[2:, 2:],
            padded[2:, 1:-1],
            padded[2:, :-2],
            padded[1:-1, :-2],
        ],
        axis=0,
    )
    bits = neighbors >= center
    transitions = np.count_nonzero(bits != np.roll(bits, -1, axis=0), axis=0)
    ones = bits.sum(axis=0)
    return np.where(transitions <= 2, ones, 9).astype(np.uint8)


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """对小尺寸二值块做形态学细化，返回近似骨架布尔图。"""
    working = np.where(mask > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(working)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(working, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(working, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        working = eroded
        if cv2.countNonZero(working) == 0:
            break

    return skeleton > 0


# 数字窗口检测阶段的中间结果。
# 这里会保留每种特征方法的离群分数，便于最终汇总成一个统一候选框。
@dataclass(slots=True)
class WindowResult:
    line_index: int
    line_box: tuple[int, int, int, int]
    character_indices: list[int]
    outlier_index_in_window: int
    outlier_global_index: int
    selected_bbox: tuple[int, int, int, int]
    method_scores: dict[str, list[float]]
    method_outliers: dict[str, int]
    window_score: float


# 输出给上层的统一候选区域结构。
# 无论来源是数字规则、时间规则、文字热点还是局部区域异常，最终都落到同一数据模型里。
@dataclass(slots=True)
class TamperRegion:
    detection_type: str
    line_index: int
    bbox: tuple[int, int, int, int]
    char_indices: list[int]
    score: float
    label: str
    detail: dict[str, object]


# 检测器最终返回值。
# `candidate_regions` 保留全部候选，`detections` 仅保留筛选后的输出框。
@dataclass(slots=True)
class DetectionResult:
    detections: list[TamperRegion]
    candidate_regions: list[TamperRegion]
    digit_windows: list[WindowResult]
    status: str
    reason: str | None


# 当前图像的候选分布统计，仅用于图内自适应阈值和候选裁决。
@dataclass(slots=True)
class CandidateStats:
    score_median: float
    score_mad: float
    type_scores: dict[str, list[float]]
    line_width_median: float
    line_height_median: float
    char_width_median: float
    char_height_median: float
    gap_median: float


class TraditionalTamperDetector:
    # 多特征融合的总体思路：
    # 1. 先构建文本行和字符上下文；
    # 2. 对数字、分隔符、中文分别做类型化异常打分；
    # 3. 生成多种规则候选（数字窗口、时间串、短中文、证件区域、区域滑窗）；
    # 4. 统一做阈值过滤、候选合并和输出框裁决。
    METHOD_WEIGHTS = {
        "texture": 0.55,
        "edge": 0.55,
        "jpeg": 1.0,
        "stroke": 1.2,
        "clahe": 1.05,
    }

    PRIMARY_METHOD = "stroke"
    DIGIT_MIN_WIDTH = 12
    DIGIT_MAX_WIDTH = 28
    MAX_DIGIT_GAP = 6
    PREFIX_BONUS = 1.4
    DECIMAL_SUFFIX_BONUS = 2.4
    RUN_OFFSET_PENALTY = 1.1
    LONG_RUN_PENALTY = 0.3
    NO_CONTEXT_PENALTY = 3.0

    DIGIT_WINDOW_THRESHOLD = 12.0
    TIME_GROUP_THRESHOLD = 4.8
    SHORT_TEXT_THRESHOLD = 6.0
    TEXT_NOISE_THRESHOLD = 4.6
    TIME_RUN_MIN_LENGTH = 5
    TIME_RUN_MAX_LENGTH = 6
    TIME_RUN_MAX_GAP = 18
    SHORT_SEPARATOR_HEIGHT_RATIO = 0.86
    SHORT_TEXT_MIN_WIDTH_RATIO = 2.4
    SHORT_TEXT_MIN_SPARE_WIDTH = 120
    MAX_OUTPUT_DETECTIONS = 5
    MIN_CONTEXT_TOTAL_CHARS = 6
    MIN_CONTEXT_LINE_CHARS = 4
    SPECIAL_RULE_TYPES = {"digit_window", "time_group", "short_text"}
    GLOBAL_SCAN_MAX_SIDE = 1600
    GLOBAL_SCAN_MAX_WINDOWS = 12000
    GLOBAL_SCAN_MIN_WINDOW = 12
    GLOBAL_SCAN_MIN_STRIDE = 4

    def __init__(self, overrides: dict[str, object] | None = None):
        self.METHOD_WEIGHTS = dict(self.__class__.METHOD_WEIGHTS)
        self.SPECIAL_RULE_TYPES = set(self.__class__.SPECIAL_RULE_TYPES)
        if not overrides:
            return

        for key, value in overrides.items():
            if key.startswith("METHOD_WEIGHT_"):
                method_name = key.removeprefix("METHOD_WEIGHT_").lower()
                self.METHOD_WEIGHTS[method_name] = float(value)
                continue
            if key == "SPECIAL_RULE_TYPES":
                self.SPECIAL_RULE_TYPES = {
                    item.strip()
                    for item in str(value).split(",")
                    if item.strip()
                }
                continue
            setattr(self, key, value)

    @classmethod
    def tunable_defaults(cls) -> dict[str, object]:
        return {
            "METHOD_WEIGHT_TEXTURE": float(cls.METHOD_WEIGHTS["texture"]),
            "METHOD_WEIGHT_EDGE": float(cls.METHOD_WEIGHTS["edge"]),
            "METHOD_WEIGHT_JPEG": float(cls.METHOD_WEIGHTS["jpeg"]),
            "METHOD_WEIGHT_STROKE": float(cls.METHOD_WEIGHTS["stroke"]),
            "METHOD_WEIGHT_CLAHE": float(cls.METHOD_WEIGHTS["clahe"]),
            "PRIMARY_METHOD": cls.PRIMARY_METHOD,
            "DIGIT_MIN_WIDTH": cls.DIGIT_MIN_WIDTH,
            "DIGIT_MAX_WIDTH": cls.DIGIT_MAX_WIDTH,
            "MAX_DIGIT_GAP": cls.MAX_DIGIT_GAP,
            "PREFIX_BONUS": cls.PREFIX_BONUS,
            "DECIMAL_SUFFIX_BONUS": cls.DECIMAL_SUFFIX_BONUS,
            "RUN_OFFSET_PENALTY": cls.RUN_OFFSET_PENALTY,
            "LONG_RUN_PENALTY": cls.LONG_RUN_PENALTY,
            "NO_CONTEXT_PENALTY": cls.NO_CONTEXT_PENALTY,
            "DIGIT_WINDOW_THRESHOLD": cls.DIGIT_WINDOW_THRESHOLD,
            "TIME_GROUP_THRESHOLD": cls.TIME_GROUP_THRESHOLD,
            "SHORT_TEXT_THRESHOLD": cls.SHORT_TEXT_THRESHOLD,
            "TEXT_NOISE_THRESHOLD": cls.TEXT_NOISE_THRESHOLD,
            "TIME_RUN_MIN_LENGTH": cls.TIME_RUN_MIN_LENGTH,
            "TIME_RUN_MAX_LENGTH": cls.TIME_RUN_MAX_LENGTH,
            "TIME_RUN_MAX_GAP": cls.TIME_RUN_MAX_GAP,
            "SHORT_SEPARATOR_HEIGHT_RATIO": cls.SHORT_SEPARATOR_HEIGHT_RATIO,
            "SHORT_TEXT_MIN_WIDTH_RATIO": cls.SHORT_TEXT_MIN_WIDTH_RATIO,
            "SHORT_TEXT_MIN_SPARE_WIDTH": cls.SHORT_TEXT_MIN_SPARE_WIDTH,
            "MAX_OUTPUT_DETECTIONS": cls.MAX_OUTPUT_DETECTIONS,
            "MIN_CONTEXT_TOTAL_CHARS": cls.MIN_CONTEXT_TOTAL_CHARS,
            "MIN_CONTEXT_LINE_CHARS": cls.MIN_CONTEXT_LINE_CHARS,
            "GLOBAL_SCAN_MAX_SIDE": cls.GLOBAL_SCAN_MAX_SIDE,
            "GLOBAL_SCAN_MAX_WINDOWS": cls.GLOBAL_SCAN_MAX_WINDOWS,
            "GLOBAL_SCAN_MIN_WINDOW": cls.GLOBAL_SCAN_MIN_WINDOW,
            "GLOBAL_SCAN_MIN_STRIDE": cls.GLOBAL_SCAN_MIN_STRIDE,
            "SPECIAL_RULE_TYPES": ",".join(sorted(cls.SPECIAL_RULE_TYPES)),
        }

    def detect(
        self,
        gray: np.ndarray,
        image: np.ndarray | None = None,
    ) -> DetectionResult:
        # 先缓存当前图像上下文，供后续大区域收缩、证件定位等辅助函数复用。
        self._last_gray = gray
        self._last_gray_shape = gray.shape
        self._document_bbox = self._detect_document_bbox(gray, image)

        # 第一阶段：构建文本上下文。
        line_boxes = detect_text_lines(gray)
        if not line_boxes:
            return self._fallback_region_result(gray, reason="文本行不足")

        line_characters = [
            segment_line_characters(gray, line_box, line_index)
            for line_index, line_box in enumerate(line_boxes)
        ]
        all_characters = [char for characters in line_characters for char in characters]

        # 第二阶段：按字符类别构建可比较的风格向量和离群分数。
        type_cache = {id(char): self._classify_character(char) for char in all_characters}
        style_cache = {id(char): self._style_vector(char) for char in all_characters}
        digit_scores: dict[int, float] = {}
        cjk_scores: dict[int, float] = {}
        if all_characters:
            digit_scores = self._build_type_scores(
                line_characters=line_characters,
                type_cache=type_cache,
                style_cache=style_cache,
                allowed_types={"digit", "separator"},
            )
            cjk_scores = self._build_type_scores(
                line_characters=line_characters,
                type_cache=type_cache,
                style_cache=style_cache,
                allowed_types={"cjk"},
            )

        # 第三阶段：并行生成不同来源的候选区域。
        digit_windows = self._enumerate_digit_windows(gray, line_boxes, line_characters)
        candidates: list[TamperRegion] = []
        candidates.extend(self._enumerate_text_block_regions(gray, line_boxes))
        candidates.extend(self._enumerate_text_noise_anomalies(gray))
        candidates.extend(self._enumerate_text_window_anomalies(gray))
        candidates.extend(self._enumerate_region_anomalies(gray))
        if all_characters:
            candidates.extend(self._digit_window_regions(digit_windows))
            candidates.extend(
                self._enumerate_time_group_regions(
                    line_boxes=line_boxes,
                    line_characters=line_characters,
                    type_cache=type_cache,
                    digit_scores=digit_scores,
                )
            )
            candidates.extend(
                self._enumerate_short_text_regions(
                    line_boxes=line_boxes,
                    line_characters=line_characters,
                    type_cache=type_cache,
                    cjk_scores=cjk_scores,
                )
            )
            candidates.extend(self._enumerate_embedded_id_card_regions(gray))
        if not candidates:
            return self._fallback_region_result(gray, reason="文本上下文不足")

        # 第四阶段：统一排序、阈值筛选，并压缩成最终输出框。
        ordered_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        self._candidate_stats = self._build_candidate_stats(
            line_boxes=line_boxes,
            line_characters=line_characters,
            candidates=ordered_candidates,
        )
        scored_candidates = [
            candidate
            for candidate in ordered_candidates
            if candidate.score >= self._candidate_threshold(candidate)
        ]
        detections = self._select_regions(scored_candidates)

        if detections:
            status = "detected"
            reason = None
        else:
            status, reason = self._summarize_empty_result(
                line_boxes=line_boxes,
                line_characters=line_characters,
                ordered_candidates=ordered_candidates,
            )

        return DetectionResult(
            detections=detections,
            candidate_regions=ordered_candidates,
            digit_windows=sorted(
                digit_windows,
                key=lambda item: item.window_score,
                reverse=True,
            ),
            status=status,
            reason=reason,
        )

    def _fallback_region_result(
        self,
        gray: np.ndarray,
        reason: str,
    ) -> DetectionResult:
        # 如果文本上下文完全不可用，仍尝试用纯区域异常方法给出一个兜底结果，
        # 避免在大块贴图、模糊裁切或极少文本场景下完全无输出。
        candidates = sorted(
            self._enumerate_region_anomalies(gray),
            key=lambda item: item.score,
            reverse=True,
        )
        self._candidate_stats = self._build_candidate_stats(
            line_boxes=[],
            line_characters=[],
            candidates=candidates,
        )
        scored_candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= self._candidate_threshold(candidate)
        ]
        detections = self._select_regions(scored_candidates)
        if detections:
            return DetectionResult(
                detections=detections,
                candidate_regions=candidates,
                digit_windows=[],
                status="detected",
                reason=None,
            )

        return DetectionResult(
            detections=[],
            candidate_regions=candidates,
            digit_windows=[],
            status="insufficient_context",
            reason=reason,
        )

    def _digit_window_regions(self, digit_windows: list[WindowResult]) -> list[TamperRegion]:
        # 将“4 字符窗口中的异常字符”映射成统一候选结构。
        results: list[TamperRegion] = []
        for window in digit_windows:
            results.append(
                TamperRegion(
                    detection_type="digit_window",
                    line_index=window.line_index,
                    bbox=window.selected_bbox,
                    char_indices=window.character_indices,
                    score=window.window_score,
                    label="数字篡改",
                    detail={
                        "line_box": list(window.line_box),
                        "selected_index_in_window": window.outlier_index_in_window,
                        "selected_index_in_line": window.outlier_global_index,
                        "method_outliers": window.method_outliers,
                        "support_count": len(window.character_indices),
                    },
                )
            )
        return results

    def _enumerate_time_group_regions(
        self,
        line_boxes: list[tuple[int, int, int, int]],
        line_characters: list[list[CharacterCandidate]],
        type_cache: dict[int, str],
        digit_scores: dict[int, float],
    ) -> list[TamperRegion]:
        # 时间串检测面向类似“21:16”“12-30”这类数字+分隔符组合。
        # 它不是只找一个异常字符，而是把整组数字视作一个候选区域，
        # 再根据组内离群程度、分隔符形态和前后中文上下文综合评分。
        results: list[TamperRegion] = []

        for line_index, (line_box, characters) in enumerate(zip(line_boxes, line_characters)):
            for run_start, run_end in self._enumerate_numeric_runs(characters, type_cache):
                group = characters[run_start:run_end]
                if not (self.TIME_RUN_MIN_LENGTH <= len(group) <= self.TIME_RUN_MAX_LENGTH):
                    continue

                gaps = self._character_gaps(group)
                if gaps and max(gaps) > self.TIME_RUN_MAX_GAP:
                    continue

                median_height = float(np.median([char.height for char in group]))
                delimiter_positions = [
                    idx
                    for idx, char in enumerate(group)
                    if 0 < idx < len(group) - 1
                    and char.width <= 14
                    and char.height <= median_height * self.SHORT_SEPARATOR_HEIGHT_RATIO
                ]
                if not delimiter_positions:
                    continue

                anomaly_scores = [
                    float(digit_scores.get(id(char), 0.0))
                    for char in group
                ]
                internal_scores = self._internal_group_scores(group)
                top_scores = sorted(anomaly_scores, reverse=True)
                mean_top2 = float(np.mean(top_scores[:2]))

                context_bonus = 0.0
                if run_start > 0 and type_cache[id(characters[run_start - 1])] == "cjk":
                    context_bonus += 0.35
                if run_end < len(characters) and type_cache[id(characters[run_end])] == "cjk":
                    context_bonus += 0.35

                # 分数既看组内最异常字符，也看整体结构是否像一个独立时间字段。
                group_score = (
                    top_scores[0]
                    + 0.65 * mean_top2
                    + 0.45 * len(delimiter_positions)
                    + context_bonus
                    + 0.55 * max(internal_scores)
                    + 0.15 * min(len(group), 6)
                )

                results.append(
                    TamperRegion(
                        detection_type="time_group",
                        line_index=line_index,
                        bbox=self._union_bbox(group),
                        char_indices=[char.index_in_line for char in group],
                        score=group_score,
                        label="时间篡改",
                        detail={
                            "line_box": list(line_box),
                            "delimiter_positions": delimiter_positions,
                            "char_scores": [round(value, 4) for value in anomaly_scores],
                            "internal_scores": [round(value, 4) for value in internal_scores],
                            "context_bonus": round(context_bonus, 4),
                            "support_count": len(group),
                        },
                    )
                )

        return results

    def _enumerate_short_text_regions(
        self,
        line_boxes: list[tuple[int, int, int, int]],
        line_characters: list[list[CharacterCandidate]],
        type_cache: dict[int, str],
        cjk_scores: dict[int, float],
    ) -> list[TamperRegion]:
        # 短中文检测更关注“局部插入/替换的小字段”，
        # 因此除了字符风格异常外，还会额外利用左右留白、局部间距和行宽占比。
        results: list[TamperRegion] = []

        for line_index, (line_box, characters) in enumerate(zip(line_boxes, line_characters)):
            for group_start, group_end in self._enumerate_short_cjk_groups(characters, type_cache):
                group = characters[group_start:group_end]
                group_bbox = self._union_bbox(group)
                leading_gap = group_bbox[0] - line_box[0]
                trailing_gap = (line_box[0] + line_box[2]) - (group_bbox[0] + group_bbox[2])
                prev_gap = (
                    leading_gap
                    if group_start == 0
                    else group_bbox[0]
                    - (characters[group_start - 1].bbox[0] + characters[group_start - 1].bbox[2])
                )
                next_gap = (
                    trailing_gap
                    if group_end == len(characters)
                    else characters[group_end].bbox[0] - (group_bbox[0] + group_bbox[2])
                )
                local_gap = max(prev_gap, next_gap)
                edge_spare = max(leading_gap, trailing_gap)
                spare_width = max(edge_spare, local_gap)
                width_ratio = line_box[2] / max(group_bbox[2], 1)
                spare_ratio = spare_width / max(group_bbox[2], 1)
                local_gap_ratio = local_gap / max(group_bbox[2], 1)
                edge_spare_ratio = edge_spare / max(group_bbox[2], 1)
                min_width_ratio = 1.55 if line_box[2] < 260 else 2.1
                min_spare_width = min(
                    self.SHORT_TEXT_MIN_SPARE_WIDTH,
                    max(36, int(line_box[2] * 0.18)),
                )
                if len(characters) > len(group) + 2 and local_gap < max(26, int(group_bbox[2] * 0.55)):
                    continue

                if (
                    width_ratio < min_width_ratio
                    and spare_width < min_spare_width
                    and edge_spare_ratio < 1.0
                    and local_gap_ratio < 0.5
                ):
                    continue

                anomaly_scores = [
                    float(cjk_scores.get(id(char), 0.0))
                    for char in group
                ]
                capped_scores = [min(score, 5.0) for score in anomaly_scores]
                line_count_bonus = 0.85 if len(line_boxes) <= 2 else 0.0
                short_group_bonus = 0.5 if len(group) == 2 else 0.25
                # 这里特意把间距类特征权重拉高，
                # 因为短文本篡改常常表现为“词组本身不长，但两侧空白或邻接关系不自然”。
                group_score = (
                    max(capped_scores)
                    + 0.7 * float(np.mean(capped_scores))
                    + 0.45 * min(width_ratio, 4.0)
                    + 1.85 * min(local_gap_ratio, 2.6)
                    + 0.8 * min(edge_spare_ratio, 1.8)
                    + line_count_bonus
                    + short_group_bonus
                )

                results.append(
                    TamperRegion(
                        detection_type="short_text",
                        line_index=line_index,
                        bbox=group_bbox,
                        char_indices=[char.index_in_line for char in group],
                        score=group_score,
                        label="中文篡改",
                        detail={
                            "line_box": list(line_box),
                            "char_scores": [round(value, 4) for value in anomaly_scores],
                            "capped_char_scores": [round(value, 4) for value in capped_scores],
                            "width_ratio": round(float(width_ratio), 4),
                            "spare_width": int(spare_width),
                            "spare_ratio": round(float(spare_ratio), 4),
                            "edge_spare_ratio": round(float(edge_spare_ratio), 4),
                            "local_gap": int(local_gap),
                            "local_gap_ratio": round(float(local_gap_ratio), 4),
                            "line_count": len(line_boxes),
                            "support_count": len(group),
                        },
                    )
                )

        return results

    def _enumerate_short_cjk_groups(
        self,
        characters: list[CharacterCandidate],
        type_cache: dict[int, str],
    ) -> list[tuple[int, int]]:
        # 先提取连续中文 run，再从中枚举适合做“短文本异常检测”的窗口。
        groups: set[tuple[int, int]] = set()
        run_start: int | None = None

        for index, char in enumerate(characters):
            if type_cache[id(char)] == "cjk":
                if run_start is None:
                    run_start = index
                continue

            if run_start is not None:
                self._append_short_cjk_windows(groups, run_start, index)
                run_start = None

        if run_start is not None:
            self._append_short_cjk_windows(groups, run_start, len(characters))

        return sorted(groups)

    def _append_short_cjk_windows(
        self,
        groups: set[tuple[int, int]],
        run_start: int,
        run_end: int,
    ) -> None:
        # 超过 4 个字符的连续中文不直接整段判断，
        # 而是切成首尾若干个短窗口，避免大段正常文本稀释掉局部异常。
        run_length = run_end - run_start
        if run_length < 2:
            return
        if run_length <= 4:
            groups.add((run_start, run_end))
            return

        for size in range(2, 5):
            groups.add((run_start, run_start + size))
            groups.add((run_end - size, run_end))

    def _enumerate_region_anomalies(
        self,
        gray: np.ndarray,
    ) -> list[TamperRegion]:
        # 这是最通用的局部异常回退逻辑。
        # 它不依赖字符切分，直接在整张图上做多尺度滑窗，
        # 用窗口特征与全局/局部统计的偏离程度来找可疑区域。
        image_height, image_width = gray.shape
        min_side = min(image_height, image_width)
        window_sizes = sorted(
            {
                24,
                max(48, min_side // 10),
                max(96, min_side // 5),
            }
        )

        results: list[TamperRegion] = []
        for window_size in window_sizes:
            stride = max(12, window_size // 2)
            scan_gray, scan_scale, scan_window_size, scan_stride, mapped_window_size = self._prepare_dense_scan(
                gray=gray,
                window_size=window_size,
                stride=stride,
            )
            if scan_window_size <= 0:
                continue
            windows: list[tuple[int, int, int, int]] = []
            feature_rows: list[np.ndarray] = []

            for y in range(0, scan_gray.shape[0] - scan_window_size + 1, scan_stride):
                for x in range(0, scan_gray.shape[1] - scan_window_size + 1, scan_stride):
                    patch = scan_gray[y : y + scan_window_size, x : x + scan_window_size]
                    feature_rows.append(self._region_feature_vector(patch))
                    windows.append((x, y, scan_window_size, scan_window_size))

            if len(windows) < 8:
                continue

            feature_matrix = np.asarray(feature_rows, dtype=np.float32)
            raw_scores = self._robust_scores(feature_matrix)
            min_score = 8.6 if window_size <= 28 else 3.9 if window_size <= 80 else 3.1
            ranked_indices = np.argsort(raw_scores)[::-1]
            top_indices: list[int] = []
            for index in ranked_indices:
                raw_score = float(raw_scores[index])
                if raw_score < min_score:
                    break
                bbox = windows[index]
                # 同尺度窗口如果高度重叠，只保留更强的那个热点，减少重复候选。
                if any(self._bbox_iou(bbox, windows[selected]) > 0.08 for selected in top_indices):
                    continue
                top_indices.append(int(index))
                if len(top_indices) >= 8:
                    break

            for index in top_indices:
                raw_score = float(raw_scores[index])
                scan_bbox = windows[index]
                bbox = self._map_scan_bbox_to_original(scan_bbox, scan_scale, gray.shape)
                context_score = self._region_context_score(scan_gray, scan_bbox, feature_matrix[index])
                final_score = raw_score + 0.7 * context_score
                # 只有“本身异常”且“与周围上下文明显不一样”的窗口才保留下来。
                if final_score < min_score + 0.4:
                    continue
                normalized_score = final_score / max(min_score, 1e-6)

                results.append(
                    TamperRegion(
                        detection_type="region_anomaly",
                        line_index=-1,
                        bbox=bbox,
                        char_indices=[],
                        score=normalized_score,
                        label="局部篡改",
                        detail={
                            "window_size": mapped_window_size,
                            "raw_score": round(raw_score, 4),
                            "context_score": round(context_score, 4),
                            "final_score": round(final_score, 4),
                            "support_count": 1,
                        },
                    )
                )

        return self._merge_region_anomaly_candidates(results)

    def _enumerate_text_window_anomalies(
        self,
        gray: np.ndarray,
    ) -> list[TamperRegion]:
        # 对小尺度热点窗口做一次“贴文字组件”的精细化校正，
        # 以便输出更贴近篡改文字而不是原始滑窗的矩形框。
        window_size = 24
        stride = 12
        scan_gray, scan_scale, scan_window_size, scan_stride, mapped_window_size = self._prepare_dense_scan(
            gray=gray,
            window_size=window_size,
            stride=stride,
        )
        if scan_window_size <= 0:
            return []
        windows: list[tuple[int, int, int, int]] = []
        feature_rows: list[np.ndarray] = []
        for y in range(0, scan_gray.shape[0] - scan_window_size + 1, scan_stride):
            for x in range(0, scan_gray.shape[1] - scan_window_size + 1, scan_stride):
                feature_rows.append(
                    self._region_feature_vector(
                        scan_gray[y : y + scan_window_size, x : x + scan_window_size]
                    )
                )
                windows.append((x, y, scan_window_size, scan_window_size))

        if len(windows) < 8:
            return []

        scores = self._robust_scores(np.asarray(feature_rows, dtype=np.float32))
        hot_windows = [
            (
                float(scores[index]),
                self._map_scan_bbox_to_original(windows[index], scan_scale, gray.shape),
            )
            for index in range(len(windows))
            if float(scores[index]) >= 4.2
        ]
        hot_windows.sort(key=lambda item: item[0], reverse=True)
        seed_windows: list[tuple[float, tuple[int, int, int, int]]] = []
        for score, bbox in hot_windows:
            if any(
                self._bbox_overlap_ratio(seed_bbox, bbox) > 0.1
                or self._center_distance(seed_bbox, bbox) <= 20.0
                for _, seed_bbox in seed_windows
            ):
                continue
            seed_windows.append((score, bbox))
            if len(seed_windows) >= 40:
                break

        component_boxes = self._text_component_boxes(gray)
        results: list[TamperRegion] = []
        snapped_seen: list[tuple[int, int, int, int]] = []
        for score, bbox in seed_windows:
            if bbox[2] > 180 or bbox[3] > 90:
                continue

            snapped_bbox, support_count = self._snap_text_cluster_to_components(bbox, component_boxes)
            if support_count == 0:
                continue
            if any(self._bbox_overlap_ratio(seen, snapped_bbox) > 0.35 for seen in snapped_seen):
                continue
            snapped_seen.append(snapped_bbox)

            base_score = float(np.log1p(max(score, 0.0)))
            box_area_ratio = (snapped_bbox[2] * snapped_bbox[3]) / float(
                mapped_window_size * mapped_window_size
            )
            # 贴合后的文字框如果放大太多，往往意味着热点抓到的是大片纹理或背景，
            # 因此这里对过大的文字框加一个面积惩罚。
            size_penalty = 0.22 * max(box_area_ratio - 2.5, 0.0)
            final_score = 2.9 + 1.55 * base_score + 0.08 * support_count - size_penalty
            results.append(
                TamperRegion(
                    detection_type="text_region",
                    line_index=-1,
                    bbox=snapped_bbox,
                    char_indices=[],
                    score=final_score,
                    label="文字篡改",
                    detail={
                        "window_size": mapped_window_size,
                        "raw_score": round(base_score, 4),
                        "support_count": support_count,
                        "size_penalty": round(size_penalty, 4),
                    },
                )
            )

        results.extend(
            self._enumerate_document_zone_text_regions(
                gray=gray,
                hot_windows=hot_windows,
                component_boxes=component_boxes,
                window_size=mapped_window_size,
            )
        )
        return results

    def _prepare_dense_scan(
        self,
        gray: np.ndarray,
        window_size: int,
        stride: int,
    ) -> tuple[np.ndarray, float, int, int, int]:
        image_height, image_width = gray.shape
        max_side = max(image_height, image_width)
        max_scan_side = max(64, int(self.GLOBAL_SCAN_MAX_SIDE))
        scale = 1.0
        scan_gray = gray
        if max_side > max_scan_side:
            scale = max_scan_side / float(max_side)
            scan_width = max(1, int(round(image_width * scale)))
            scan_height = max(1, int(round(image_height * scale)))
            scan_gray = cv2.resize(
                gray,
                (scan_width, scan_height),
                interpolation=cv2.INTER_AREA,
            )

        min_window = max(4, int(self.GLOBAL_SCAN_MIN_WINDOW))
        min_stride = max(1, int(self.GLOBAL_SCAN_MIN_STRIDE))
        scan_window_size = max(min_window, int(round(window_size * scale)))
        scan_window_size = min(scan_window_size, scan_gray.shape[0], scan_gray.shape[1])
        if scan_window_size < 4:
            return scan_gray, scale, 0, 0, 0

        scan_stride = max(min_stride, int(round(stride * scale)))
        scan_stride = min(scan_stride, scan_window_size)
        scan_stride = self._fit_dense_scan_stride(
            image_shape=scan_gray.shape,
            window_size=scan_window_size,
            stride=scan_stride,
        )
        mapped_window_size = max(1, int(round(scan_window_size / max(scale, 1e-6))))
        return scan_gray, scale, scan_window_size, scan_stride, mapped_window_size

    def _fit_dense_scan_stride(
        self,
        image_shape: tuple[int, int],
        window_size: int,
        stride: int,
    ) -> int:
        max_windows = int(self.GLOBAL_SCAN_MAX_WINDOWS)
        if max_windows <= 0:
            return stride
        current_stride = max(1, stride)
        while (
            self._dense_scan_window_count(image_shape, window_size, current_stride)
            > max_windows
        ):
            current_stride += max(1, current_stride // 4)
        return current_stride

    @staticmethod
    def _dense_scan_window_count(
        image_shape: tuple[int, int],
        window_size: int,
        stride: int,
    ) -> int:
        image_height, image_width = image_shape
        if window_size <= 0 or stride <= 0:
            return 0
        if image_height < window_size or image_width < window_size:
            return 0
        row_count = (image_height - window_size) // stride + 1
        col_count = (image_width - window_size) // stride + 1
        return int(row_count * col_count)

    @staticmethod
    def _map_scan_bbox_to_original(
        bbox: tuple[int, int, int, int],
        scale: float,
        original_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        if scale >= 0.999:
            return bbox
        x, y, w, h = bbox
        inverse_scale = 1.0 / max(scale, 1e-6)
        mapped = (
            int(round(x * inverse_scale)),
            int(round(y * inverse_scale)),
            int(round(w * inverse_scale)),
            int(round(h * inverse_scale)),
        )
        return TraditionalTamperDetector._clip_bbox_to_image(mapped, original_shape)

    @staticmethod
    def _clip_bbox_to_image(
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        image_height, image_width = image_shape
        x0 = max(0, min(image_width - 1, x))
        y0 = max(0, min(image_height - 1, y))
        x1 = max(1, min(image_width, x + w))
        y1 = max(1, min(image_height, y + h))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def _enumerate_text_block_regions(
        self,
        gray: np.ndarray,
        line_boxes: list[tuple[int, int, int, int]],
    ) -> list[TamperRegion]:
        # 这一层不依赖单字符切分，而是直接把黑帽连通域聚成文本块。
        # 这样即使字符粘连、局部模糊或版式不规则，也能给出较稳定的文本候选。
        if not line_boxes:
            return []

        component_boxes = self._text_component_boxes(gray)
        if len(component_boxes) < 4:
            return []

        results: list[TamperRegion] = []
        image_height, image_width = gray.shape
        for line_index, line_box in enumerate(line_boxes):
            line_x, line_y, line_w, line_h = line_box
            expanded_box = (
                max(0, line_x - 8),
                max(0, line_y - 4),
                min(image_width - max(0, line_x - 8), line_w + 16),
                min(image_height - max(0, line_y - 4), line_h + 8),
            )
            line_components = [
                box
                for box in component_boxes
                if expanded_box[0] <= self._center_x(box) <= expanded_box[0] + expanded_box[2]
                and expanded_box[1] <= self._center_y(box) <= expanded_box[1] + expanded_box[3]
            ]
            if len(line_components) < 2:
                continue

            median_component_gap = np.median([box[2] for box in line_components]) if line_components else 8.0
            clusters = self._cluster_component_boxes(
                line_components,
                max_gap=max(8, int(median_component_gap * 1.5)),
                min_overlap_ratio=0.25,
            )
            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                cluster_bbox = self._union_boxes_many(cluster)
                if cluster_bbox[2] < 24 or cluster_bbox[3] < 10:
                    continue

                patch = gray[
                    cluster_bbox[1] : cluster_bbox[1] + cluster_bbox[3],
                    cluster_bbox[0] : cluster_bbox[0] + cluster_bbox[2],
                ]
                if patch.size == 0:
                    continue

                patch_feature = self._region_feature_vector(patch)
                context_score = self._region_context_score(gray, cluster_bbox, patch_feature)
                line_fill = cluster_bbox[2] / max(line_w, 1)
                compactness_bonus = 0.16 * min(len(cluster), 6)
                span_bonus = 0.32 * min(line_fill, 3.0)
                final_score = 3.9 + 0.78 * context_score + compactness_bonus + span_bonus
                if final_score < 4.2 and len(cluster) < 3:
                    continue

                snapped_bbox, support_count = self._snap_text_cluster_to_components(cluster_bbox, component_boxes)
                final_bbox = snapped_bbox if support_count > 0 else cluster_bbox
                results.append(
                    TamperRegion(
                        detection_type="text_region",
                        line_index=line_index,
                        bbox=final_bbox,
                        char_indices=[],
                        score=final_score + 0.04 * support_count,
                        label="文字篡改",
                        detail={
                            "window_size": max(final_bbox[2], final_bbox[3]),
                            "raw_score": round(float(context_score), 4),
                            "support_count": max(support_count, len(cluster)),
                            "size_penalty": 0.0,
                            "zone": "generic_text_block",
                            "component_count": len(cluster),
                        },
                    )
                )

        return self._merge_text_region_candidates(results)

    def _enumerate_text_noise_anomalies(
        self,
        gray: np.ndarray,
    ) -> list[TamperRegion]:
        # 用户指出的核心证据是“少量文字的噪声和其余文字不一致”。
        # 这里以黑帽文字组件为比较单元，同时结合全图文字基线和同行文字基线，
        # 避免只拿局部背景做比较导致表格线、边缘纹理误报。
        component_boxes = self._filter_noise_component_boxes(gray, self._text_component_boxes(gray))
        if len(component_boxes) < 8:
            return []

        feature_matrix = np.asarray(
            [self._text_noise_feature(gray, box) for box in component_boxes],
            dtype=np.float32,
        )
        global_scores = self._robust_scores(feature_matrix)
        row_groups = self._group_text_components_by_row(component_boxes)
        row_score_by_index = {index: 0.0 for index in range(len(component_boxes))}
        row_size_by_index = {index: 0 for index in range(len(component_boxes))}
        for row_indices in row_groups:
            if len(row_indices) < 4:
                continue
            row_features = feature_matrix[row_indices]
            row_scores = self._robust_scores(row_features)
            for offset, component_index in enumerate(row_indices):
                row_score_by_index[component_index] = float(row_scores[offset])
                row_size_by_index[component_index] = len(row_indices)

        combined_scores: list[float] = []
        for index, global_score in enumerate(global_scores):
            row_score = row_score_by_index[index]
            if row_size_by_index[index] >= 4:
                combined_scores.append(0.62 * row_score + 0.38 * float(global_score))
            else:
                combined_scores.append(float(global_score))

        if not combined_scores:
            return []

        score_values = np.asarray(combined_scores, dtype=np.float32)
        adaptive_floor = max(
            self.TEXT_NOISE_THRESHOLD,
            float(np.median(score_values) + 1.8 * np.median(np.abs(score_values - np.median(score_values)))),
        )
        hot_indices = [
            int(index)
            for index in np.argsort(score_values)[::-1]
            if float(score_values[index]) >= adaptive_floor
        ][:24]
        if not hot_indices:
            return []

        row_lookup: dict[int, list[int]] = {}
        for row_indices in row_groups:
            for component_index in row_indices:
                row_lookup[component_index] = row_indices

        results: list[TamperRegion] = []
        used_boxes: list[tuple[int, int, int, int]] = []
        for index in hot_indices:
            seed_box = component_boxes[index]
            if any(self._bbox_overlap_ratio(seed_box, used_box) > 0.35 for used_box in used_boxes):
                continue

            seed_score = float(combined_scores[index])
            row_indices = row_lookup.get(index, [index])
            cluster_indices = self._collect_noise_cluster_indices(
                seed_index=index,
                row_indices=row_indices,
                component_boxes=component_boxes,
                scores=combined_scores,
            )
            cluster_boxes = [component_boxes[item] for item in cluster_indices]
            cluster_bbox = self._union_boxes_many(cluster_boxes)
            if cluster_bbox[2] * cluster_bbox[3] < 180 and len(cluster_indices) < 2:
                continue
            if cluster_bbox[2] * cluster_bbox[3] > gray.shape[0] * gray.shape[1] * 0.035:
                continue

            support_count = len(cluster_indices)
            final_score = 3.15 + 0.72 * seed_score + 0.10 * min(support_count, 5)
            results.append(
                TamperRegion(
                    detection_type="text_noise_anomaly",
                    line_index=-1,
                    bbox=cluster_bbox,
                    char_indices=[],
                    score=final_score,
                    label="噪声篡改",
                    detail={
                        "raw_score": round(seed_score, 4),
                        "global_score": round(float(global_scores[index]), 4),
                        "row_score": round(float(row_score_by_index[index]), 4),
                        "support_count": support_count,
                        "component_count": support_count,
                        "zone": "text_noise_consistency",
                    },
                )
            )
            used_boxes.append(cluster_bbox)
            if len(results) >= 12:
                break

        return self._merge_text_noise_candidates(results)

    def _filter_noise_component_boxes(
        self,
        gray: np.ndarray,
        component_boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        image_height, image_width = gray.shape
        filtered: list[tuple[int, int, int, int]] = []
        for x, y, w, h in component_boxes:
            area = w * h
            if area < 45 or area > 9000:
                continue
            if w < 5 or h < 8 or w > 150 or h > 120:
                continue
            if x <= 1 or y <= 1 or x + w >= image_width - 1 or y + h >= image_height - 1:
                continue
            aspect = w / max(h, 1)
            if not 0.08 <= aspect <= 4.8:
                continue
            filtered.append((x, y, w, h))
        return filtered

    def _text_noise_feature(
        self,
        gray: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        x, y, w, h = bbox
        margin = max(2, min(w, h) // 4)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(gray.shape[1], x + w + margin)
        y1 = min(gray.shape[0], y + h + margin)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return np.zeros(8, dtype=np.float32)

        residual3 = cv2.absdiff(patch, cv2.GaussianBlur(patch, (3, 3), 0)).astype(np.float32)
        residual5 = cv2.absdiff(patch, cv2.GaussianBlur(patch, (5, 5), 0)).astype(np.float32)
        laplacian = np.abs(cv2.Laplacian(patch, cv2.CV_32F))
        mask = self._patch_mask(patch) > 0
        if np.count_nonzero(mask) < 6:
            mask = np.ones_like(patch, dtype=bool)
        background_mask = ~mask
        foreground_residual = residual3[mask]
        background_residual = residual3[background_mask] if np.any(background_mask) else residual3.ravel()
        foreground_laplacian = laplacian[mask]
        foreground_residual5 = residual5[mask]
        background_mean = float(background_residual.mean()) + 1e-6

        return np.asarray(
            [
                float(foreground_residual.mean() / 255.0),
                float(foreground_residual.std() / 255.0),
                float(foreground_residual5.mean() / 255.0),
                float(foreground_residual5.std() / 255.0),
                float(foreground_laplacian.mean() / 255.0),
                float(foreground_laplacian.std() / 255.0),
                float(background_residual.mean() / 255.0),
                float(min(foreground_residual.mean() / background_mean, 8.0) / 8.0),
            ],
            dtype=np.float32,
        )

    def _group_text_components_by_row(
        self,
        component_boxes: list[tuple[int, int, int, int]],
    ) -> list[list[int]]:
        if not component_boxes:
            return []

        heights = [box[3] for box in component_boxes]
        row_tolerance = max(10.0, float(np.median(heights)) * 0.72)
        rows: list[list[int]] = []
        row_centers: list[float] = []
        for index, box in sorted(enumerate(component_boxes), key=lambda item: self._center_y(item[1])):
            center_y = self._center_y(box)
            matched_index = None
            for row_index, row_center in enumerate(row_centers):
                if abs(center_y - row_center) <= row_tolerance:
                    matched_index = row_index
                    break
            if matched_index is None:
                rows.append([index])
                row_centers.append(center_y)
                continue

            rows[matched_index].append(index)
            row_centers[matched_index] = float(
                np.mean([self._center_y(component_boxes[item]) for item in rows[matched_index]])
            )

        return [sorted(row, key=lambda item: component_boxes[item][0]) for row in rows]

    def _collect_noise_cluster_indices(
        self,
        seed_index: int,
        row_indices: list[int],
        component_boxes: list[tuple[int, int, int, int]],
        scores: list[float],
    ) -> list[int]:
        seed_box = component_boxes[seed_index]
        seed_score = float(scores[seed_index])
        cluster = [seed_index]
        for component_index in row_indices:
            if component_index == seed_index:
                continue
            box = component_boxes[component_index]
            horizontal_gap = max(
                box[0] - (seed_box[0] + seed_box[2]),
                seed_box[0] - (box[0] + box[2]),
                0,
            )
            y_overlap = min(seed_box[1] + seed_box[3], box[1] + box[3]) - max(seed_box[1], box[1])
            if horizontal_gap > max(14, seed_box[2], box[2]):
                continue
            if y_overlap < min(seed_box[3], box[3]) * 0.25:
                continue
            if float(scores[component_index]) < max(self.TEXT_NOISE_THRESHOLD - 0.4, seed_score * 0.58):
                continue
            cluster.append(component_index)
        return sorted(cluster, key=lambda item: component_boxes[item][0])

    def _merge_text_noise_candidates(
        self,
        candidates: list[TamperRegion],
    ) -> list[TamperRegion]:
        merged: list[TamperRegion] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            merged_index = None
            for index, existing in enumerate(merged):
                if self._should_merge_small_windows(existing.bbox, candidate.bbox):
                    merged_index = index
                    break
            if merged_index is None:
                merged.append(candidate)
                continue

            existing = merged[merged_index]
            merged[merged_index] = TamperRegion(
                detection_type="text_noise_anomaly",
                line_index=-1,
                bbox=self._union_boxes(existing.bbox, candidate.bbox),
                char_indices=[],
                score=max(existing.score, candidate.score) + 0.05,
                label="噪声篡改",
                detail={
                    **existing.detail,
                    "support_count": int(existing.detail.get("support_count", 1))
                    + int(candidate.detail.get("support_count", 1)),
                    "component_count": int(existing.detail.get("component_count", 1))
                    + int(candidate.detail.get("component_count", 1)),
                },
            )
        return merged

    def _merge_text_region_candidates(
        self,
        candidates: list[TamperRegion],
    ) -> list[TamperRegion]:
        # 文本候选经常会在同一行里分裂成多个相邻热点，
        # 这里把相近的文本框合并成更稳定的区域输出。
        clusters: list[TamperRegion] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            merged = False
            for index, cluster in enumerate(clusters):
                if not self._should_merge_text_candidate(cluster, candidate):
                    continue
                clusters[index] = TamperRegion(
                    detection_type="text_region",
                    line_index=-1,
                    bbox=self._union_boxes(cluster.bbox, candidate.bbox),
                    char_indices=[],
                    score=max(cluster.score, candidate.score) + 0.04,
                    label="文字篡改",
                    detail={
                        "window_size": max(
                            int(cluster.detail.get("window_size", cluster.bbox[2])),
                            int(candidate.detail.get("window_size", candidate.bbox[2])),
                        ),
                        "raw_score": max(
                            float(cluster.detail.get("raw_score", cluster.score)),
                            float(candidate.detail.get("raw_score", candidate.score)),
                        ),
                        "support_count": int(cluster.detail.get("support_count", 1))
                        + int(candidate.detail.get("support_count", 1)),
                        "size_penalty": min(
                            float(cluster.detail.get("size_penalty", 0.0)),
                            float(candidate.detail.get("size_penalty", 0.0)),
                        ),
                        "zone": str(cluster.detail.get("zone", candidate.detail.get("zone", ""))),
                        "component_count": int(cluster.detail.get("component_count", 0))
                        + int(candidate.detail.get("component_count", 0)),
                    },
                )
                merged = True
                break
            if not merged:
                clusters.append(candidate)

        refined: list[TamperRegion] = []
        for cluster in clusters:
            refined_bbox = self._snap_large_region_to_rectangle(cluster.bbox, cluster.detail.get("window_size", 0))
            refined.append(
                TamperRegion(
                    detection_type=cluster.detection_type,
                    line_index=cluster.line_index,
                    bbox=refined_bbox,
                    char_indices=cluster.char_indices,
                    score=cluster.score,
                    label=cluster.label,
                    detail=cluster.detail,
                )
            )
        return refined

    def _should_merge_text_candidate(
        self,
        cluster: TamperRegion,
        candidate: TamperRegion,
    ) -> bool:
        if self._bbox_iou(cluster.bbox, candidate.bbox) > 0.0:
            return True
        if self._bbox_overlap_ratio(cluster.bbox, candidate.bbox) > 0.0:
            return True

        cx0 = cluster.bbox[0] + cluster.bbox[2] / 2.0
        cy0 = cluster.bbox[1] + cluster.bbox[3] / 2.0
        cx1 = candidate.bbox[0] + candidate.bbox[2] / 2.0
        cy1 = candidate.bbox[1] + candidate.bbox[3] / 2.0
        max_w = max(cluster.bbox[2], candidate.bbox[2])
        max_h = max(cluster.bbox[3], candidate.bbox[3])
        return (
            abs(cx0 - cx1) <= max_w * 1.4
            and abs(cy0 - cy1) <= max_h * 1.1
        )

    def _enumerate_document_zone_text_regions(
        self,
        gray: np.ndarray,
        hot_windows: list[tuple[float, tuple[int, int, int, int]]],
        component_boxes: list[tuple[int, int, int, int]],
        window_size: int,
    ) -> list[TamperRegion]:
        # 证件图像里，姓名/住址/号码的大致版式是相对稳定的。
        # 当整图看起来像身份证时，优先在这些区域内做更有针对性的文字候选聚合。
        if not self._is_id_card_like(gray):
            return []

        precise_results = self._enumerate_id_card_precise_text_regions(
            gray=gray,
            hot_windows=hot_windows,
            component_boxes=component_boxes,
            window_size=window_size,
        )
        if precise_results:
            return precise_results

        height, width = gray.shape
        zone_specs = [
            ("id_name_zone", (0.02, 0.34, 0.04, 0.24), 72, 32, "score"),
            ("id_address_zone", (0.10, 0.62, 0.48, 0.72), 120, 42, "score"),
            ("id_number_zone", (0.35, 0.96, 0.78, 0.98), 180, 36, "right"),
        ]
        results: list[TamperRegion] = []
        for zone_name, (x0_ratio, x1_ratio, y0_ratio, y1_ratio), x_radius, y_radius, mode in zone_specs:
            zone_items = [
                (score, bbox)
                for score, bbox in hot_windows
                if self._center_in_ratio_box(
                    bbox=bbox,
                    width=width,
                    height=height,
                    x0_ratio=x0_ratio,
                    x1_ratio=x1_ratio,
                    y0_ratio=y0_ratio,
                    y1_ratio=y1_ratio,
                )
            ]
            if not zone_items:
                continue

            if mode == "right":
                # 身份证号通常右侧更稳定，因此在号码区域优先选择更靠右的热点作为锚点。
                seed_score, seed_bbox = max(zone_items, key=lambda item: (item[1][0], item[0]))
                cluster = [
                    (score, bbox)
                    for score, bbox in zone_items
                    if 0 <= seed_bbox[0] - bbox[0] <= x_radius
                    and abs(self._center_y(bbox) - self._center_y(seed_bbox)) <= y_radius
                ]
            else:
                seed_score, seed_bbox = max(zone_items, key=lambda item: item[0])
                cluster = [
                    (score, bbox)
                    for score, bbox in zone_items
                    if abs(self._center_x(bbox) - self._center_x(seed_bbox)) <= x_radius
                    and abs(self._center_y(bbox) - self._center_y(seed_bbox)) <= y_radius
                ]

            merged_bbox = self._union_boxes_many([bbox for _, bbox in cluster])
            snapped_bbox, support_count = self._snap_text_cluster_to_components(merged_bbox, component_boxes)
            final_bbox = snapped_bbox if support_count > 0 else merged_bbox
            final_support = max(support_count, len(cluster))
            score = max(score for score, _ in cluster) + 0.05 * final_support + 0.02 * len(cluster)
            results.append(
                TamperRegion(
                    detection_type="text_region",
                    line_index=-1,
                    bbox=final_bbox,
                    char_indices=[],
                    score=score,
                    label="文字篡改",
                    detail={
                        "window_size": window_size,
                        "raw_score": round(float(seed_score), 4),
                        "support_count": final_support,
                        "size_penalty": 0.0,
                        "zone": zone_name,
                    },
                )
            )
        return results

    def _enumerate_id_card_precise_text_regions(
        self,
        gray: np.ndarray,
        hot_windows: list[tuple[float, tuple[int, int, int, int]]],
        component_boxes: list[tuple[int, int, int, int]],
        window_size: int,
    ) -> list[TamperRegion]:
        height, width = gray.shape
        results: list[TamperRegion] = []

        name_components = [
            box
            for box in component_boxes
            if width * 0.22 <= box[0] <= width * 0.31
            and height * 0.10 <= box[1] <= height * 0.20
        ]
        if name_components:
            name_bbox = self._union_boxes_many(name_components)
            name_score = max(
                (
                    score
                    for score, bbox in hot_windows
                    if self._bbox_overlap_ratio(name_bbox, bbox) > 0.0
                    or self._center_distance(name_bbox, bbox) <= 28.0
                ),
                default=6.8,
            )
            results.append(
                self._build_precise_text_region(
                    bbox=name_bbox,
                    score=name_score,
                    window_size=window_size,
                    support_count=len(name_components),
                    zone="id_name_precise",
                )
            )

        address_components = [
            box
            for box in component_boxes
            if width * 0.20 <= box[0] <= width * 0.46
            and height * 0.60 <= box[1] <= height * 0.69
            and box[2] <= 38
            and 18 <= box[3] <= 40
        ]
        address_clusters = self._cluster_component_boxes(address_components, max_gap=10, min_overlap_ratio=0.45)
        for cluster in address_clusters:
            cluster_bbox = self._union_boxes_many(cluster)
            if cluster_bbox[2] < 70:
                continue
            cluster_score = max(
                (
                    score
                    for score, bbox in hot_windows
                    if self._bbox_overlap_ratio(cluster_bbox, bbox) > 0.0
                    or self._center_distance(cluster_bbox, bbox) <= 32.0
                ),
                default=6.2,
            )
            results.append(
                self._build_precise_text_region(
                    bbox=cluster_bbox,
                    score=cluster_score,
                    window_size=window_size,
                    support_count=len(cluster),
                    zone="id_address_precise",
                )
            )

        id_components = [
            box
            for box in component_boxes
            if box[0] >= width * 0.72
            and height * 0.86 <= box[1] <= height * 0.96
            and box[3] >= 30
        ]
        if id_components:
            id_bbox = self._union_boxes_many(id_components)
            id_score = max(
                (
                    score
                    for score, bbox in hot_windows
                    if self._bbox_overlap_ratio(id_bbox, bbox) > 0.0
                    or self._center_distance(id_bbox, bbox) <= 28.0
                ),
                default=7.0,
            )
            results.append(
                self._build_precise_text_region(
                    bbox=id_bbox,
                    score=id_score,
                    window_size=window_size,
                    support_count=len(id_components),
                    zone="id_number_precise",
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)

    def _enumerate_embedded_id_card_regions(
        self,
        gray: np.ndarray,
    ) -> list[TamperRegion]:
        document_bbox = getattr(self, "_document_bbox", None)
        if document_bbox is None or not self._bbox_looks_like_id_card(document_bbox):
            return []

        doc_x, doc_y, doc_w, doc_h = document_bbox
        card = gray[doc_y : doc_y + doc_h, doc_x : doc_x + doc_w]
        if card.size == 0:
            return []

        hot_windows = self._collect_hot_windows(card, window_size=24, stride=12, min_score=4.0)
        component_boxes = self._text_component_boxes(card)
        if not hot_windows and not component_boxes:
            return []

        zone_specs = [
            ("embedded_id_name_precise", (0.18, 0.34, 0.08, 0.22), 1, 10, 0.2, "score"),
            ("embedded_id_address_precise", (0.15, 0.50, 0.58, 0.70), 2, 12, 0.25, "score"),
            ("embedded_id_number_precise", (0.70, 0.97, 0.82, 0.95), 1, 14, 0.25, "right"),
        ]

        results: list[TamperRegion] = []
        for zone_name, zone_ratios, max_regions, cluster_gap, overlap_ratio, mode in zone_specs:
            zone_regions = self._collect_embedded_id_text_regions(
                component_boxes=component_boxes,
                hot_windows=hot_windows,
                document_size=(doc_w, doc_h),
                document_offset=(doc_x, doc_y),
                zone_name=zone_name,
                zone_ratios=zone_ratios,
                max_regions=max_regions,
                cluster_gap=cluster_gap,
                min_overlap_ratio=overlap_ratio,
                mode=mode,
            )
            results.extend(zone_regions)

        photo_bbox = self._detect_embedded_id_photo_bbox(gray, document_bbox)
        if photo_bbox is not None:
            photo_score = max(
                (
                    score
                    for score, bbox in hot_windows
                    if self._center_in_ratio_box(
                        bbox=bbox,
                        width=doc_w,
                        height=doc_h,
                        x0_ratio=0.60,
                        x1_ratio=0.96,
                        y0_ratio=0.06,
                        y1_ratio=0.78,
                    )
                ),
                default=6.2,
            )
            results.append(
                TamperRegion(
                    detection_type="region_anomaly",
                    line_index=-1,
                    bbox=photo_bbox,
                    char_indices=[],
                    score=4.6 + 0.28 * float(photo_score),
                    label="局部篡改",
                    detail={
                        "window_size": max(photo_bbox[2], photo_bbox[3]),
                        "raw_score": round(float(photo_score), 4),
                        "context_score": 0.0,
                        "final_score": round(4.6 + 0.28 * float(photo_score), 4),
                        "support_count": 1,
                        "zone": "embedded_id_photo",
                    },
                )
            )

        return results

    def _collect_embedded_id_text_regions(
        self,
        component_boxes: list[tuple[int, int, int, int]],
        hot_windows: list[tuple[float, tuple[int, int, int, int]]],
        document_size: tuple[int, int],
        document_offset: tuple[int, int],
        zone_name: str,
        zone_ratios: tuple[float, float, float, float],
        max_regions: int,
        cluster_gap: int,
        min_overlap_ratio: float,
        mode: str,
    ) -> list[TamperRegion]:
        doc_w, doc_h = document_size
        doc_x, doc_y = document_offset
        x0_ratio, x1_ratio, y0_ratio, y1_ratio = zone_ratios

        zone_components = [
            box
            for box in component_boxes
            if self._center_in_ratio_box(
                bbox=box,
                width=doc_w,
                height=doc_h,
                x0_ratio=x0_ratio,
                x1_ratio=x1_ratio,
                y0_ratio=y0_ratio,
                y1_ratio=y1_ratio,
            )
        ]
        if not zone_components:
            return []

        clusters = self._cluster_component_boxes(
            zone_components,
            max_gap=cluster_gap,
            min_overlap_ratio=min_overlap_ratio,
        )
        if not clusters:
            return []

        ranked: list[tuple[float, tuple[int, int, int, int], int]] = []
        for cluster in clusters:
            cluster_bbox = self._union_boxes_many(cluster)
            hotspot = max(
                (
                    score
                    for score, bbox in hot_windows
                    if self._bbox_overlap_ratio(cluster_bbox, bbox) > 0.0
                    or self._center_distance(cluster_bbox, bbox) <= max(20.0, cluster_bbox[2] * 0.35)
                ),
                default=0.0,
            )
            if hotspot <= 0.0:
                continue
            support_count = len(cluster)
            ranked.append(
                (
                    float(hotspot) + 0.35 + 0.10 * min(support_count, 4),
                    cluster_bbox,
                    support_count,
                )
            )

        if not ranked:
            return []

        if mode == "right":
            ranked.sort(key=lambda item: (item[0], item[1][0] + item[1][2]), reverse=True)
        else:
            ranked.sort(key=lambda item: item[0], reverse=True)

        selected: list[TamperRegion] = []
        for score, local_bbox, support_count in ranked:
            global_bbox = (
                doc_x + local_bbox[0],
                doc_y + local_bbox[1],
                local_bbox[2],
                local_bbox[3],
            )
            if any(self._bbox_overlap_ratio(global_bbox, item.bbox) > 0.55 for item in selected):
                continue
            selected.append(
                self._build_precise_text_region(
                    bbox=global_bbox,
                    score=score,
                    window_size=24,
                    support_count=support_count,
                    zone=zone_name,
                )
            )
            if len(selected) >= max_regions:
                break
        return selected

    def _collect_hot_windows(
        self,
        gray: np.ndarray,
        window_size: int,
        stride: int,
        min_score: float,
    ) -> list[tuple[float, tuple[int, int, int, int]]]:
        windows: list[tuple[int, int, int, int]] = []
        feature_rows: list[np.ndarray] = []
        for y in range(0, gray.shape[0] - window_size + 1, stride):
            for x in range(0, gray.shape[1] - window_size + 1, stride):
                windows.append((x, y, window_size, window_size))
                feature_rows.append(self._region_feature_vector(gray[y : y + window_size, x : x + window_size]))

        if len(windows) < 8:
            return []

        scores = self._robust_scores(np.asarray(feature_rows, dtype=np.float32))
        return [
            (float(scores[index]), windows[index])
            for index in range(len(windows))
            if float(scores[index]) >= min_score
        ]

    def _detect_embedded_id_photo_bbox(
        self,
        gray: np.ndarray,
        document_bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        doc_x, doc_y, doc_w, doc_h = document_bbox
        px0 = int(doc_w * 0.60)
        py0 = int(doc_h * 0.06)
        px1 = int(doc_w * 0.96)
        py1 = int(doc_h * 0.78)
        if px1 <= px0 or py1 <= py0:
            return None

        roi = gray[doc_y + py0 : doc_y + py1, doc_x + px0 : doc_x + px1]
        if roi.size == 0:
            return None

        edges = cv2.Canny(cv2.GaussianBlur(roi, (5, 5), 0), 60, 160)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox: tuple[int, int, int, int] | None = None
        best_area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / max(h, 1)
            if area < 30000 or not (0.65 <= ratio <= 0.92):
                continue
            if area > best_area:
                best_area = area
                best_bbox = (doc_x + px0 + x, doc_y + py0 + y, w, h)
        return best_bbox

    def _build_precise_text_region(
        self,
        bbox: tuple[int, int, int, int],
        score: float,
        window_size: int,
        support_count: int,
        zone: str,
    ) -> TamperRegion:
        return TamperRegion(
            detection_type="text_region",
            line_index=-1,
            bbox=bbox,
            char_indices=[],
            score=float(score) + 0.06 * support_count,
            label="文字篡改",
            detail={
                "window_size": window_size,
                "raw_score": round(float(score), 4),
                "support_count": support_count,
                "size_penalty": 0.0,
                "zone": zone,
            },
        )

    def _cluster_component_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        max_gap: int,
        min_overlap_ratio: float,
    ) -> list[list[tuple[int, int, int, int]]]:
        if not boxes:
            return []

        ordered = sorted(boxes, key=lambda item: (item[0], item[1]))
        clusters: list[list[tuple[int, int, int, int]]] = []
        current: list[tuple[int, int, int, int]] = [ordered[0]]
        for box in ordered[1:]:
            prev = current[-1]
            gap = box[0] - (prev[0] + prev[2])
            y_overlap = min(prev[1] + prev[3], box[1] + box[3]) - max(prev[1], box[1])
            overlap_ratio = y_overlap / max(min(prev[3], box[3]), 1)
            if gap <= max_gap and overlap_ratio >= min_overlap_ratio:
                current.append(box)
            else:
                clusters.append(current)
                current = [box]
        clusters.append(current)
        return clusters

    def _merge_region_anomaly_candidates(
        self,
        candidates: list[TamperRegion],
    ) -> list[TamperRegion]:
        clusters: list[TamperRegion] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            merged = False
            for index, cluster in enumerate(clusters):
                if not self._should_merge_region_candidate(cluster, candidate):
                    continue
                clusters[index] = TamperRegion(
                    detection_type="region_anomaly",
                    line_index=-1,
                    bbox=self._union_boxes(cluster.bbox, candidate.bbox),
                    char_indices=[],
                    score=max(cluster.score, candidate.score) + 0.03,
                    label="局部篡改",
                    detail={
                        "window_size": max(
                            int(cluster.detail.get("window_size", cluster.bbox[2])),
                            int(candidate.detail.get("window_size", candidate.bbox[2])),
                        ),
                        "raw_score": max(
                            float(cluster.detail.get("raw_score", cluster.score)),
                            float(candidate.detail.get("raw_score", candidate.score)),
                        ),
                        "context_score": max(
                            float(cluster.detail.get("context_score", 0.0)),
                            float(candidate.detail.get("context_score", 0.0)),
                        ),
                        "final_score": max(
                            float(cluster.detail.get("final_score", cluster.score)),
                            float(candidate.detail.get("final_score", candidate.score)),
                        ),
                        "support_count": int(cluster.detail.get("support_count", 1))
                        + int(candidate.detail.get("support_count", 1)),
                    },
                )
                merged = True
                break
            if not merged:
                clusters.append(candidate)

        refined: list[TamperRegion] = []
        for cluster in clusters:
            refined_bbox = self._snap_large_region_to_rectangle(cluster.bbox, cluster.detail.get("window_size", 0))
            refined.append(
                TamperRegion(
                    detection_type=cluster.detection_type,
                    line_index=cluster.line_index,
                    bbox=refined_bbox,
                    char_indices=cluster.char_indices,
                    score=cluster.score,
                    label=cluster.label,
                    detail=cluster.detail,
                )
            )
        return refined

    def _should_merge_region_candidate(
        self,
        cluster: TamperRegion,
        candidate: TamperRegion,
    ) -> bool:
        if self._bbox_iou(cluster.bbox, candidate.bbox) > 0.0:
            return True
        if self._bbox_overlap_ratio(cluster.bbox, candidate.bbox) > 0.0:
            return True

        cx0 = cluster.bbox[0] + cluster.bbox[2] / 2.0
        cy0 = cluster.bbox[1] + cluster.bbox[3] / 2.0
        cx1 = candidate.bbox[0] + candidate.bbox[2] / 2.0
        cy1 = candidate.bbox[1] + candidate.bbox[3] / 2.0
        max_w = max(cluster.bbox[2], candidate.bbox[2])
        max_h = max(cluster.bbox[3], candidate.bbox[3])
        return (
            abs(cx0 - cx1) <= max_w * 1.8
            and abs(cy0 - cy1) <= max_h * 1.4
        )

    @staticmethod
    def _should_merge_small_windows(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> bool:
        ax0, ay0, aw, ah = box_a
        bx0, by0, bw, bh = box_b
        return not (
            bx0 > ax0 + aw + 18
            or ax0 > bx0 + bw + 18
            or by0 > ay0 + ah + 18
            or ay0 > by0 + bh + 18
        )

    def _build_type_scores(
        self,
        line_characters: list[list[CharacterCandidate]],
        type_cache: dict[int, str],
        style_cache: dict[int, np.ndarray],
        allowed_types: set[str],
    ) -> dict[int, float]:
        global_chars = [
            char
            for characters in line_characters
            for char in characters
            if type_cache[id(char)] in allowed_types
        ]
        global_scores = self._build_global_scores(global_chars, style_cache)
        global_count = len(global_chars)

        combined_scores: dict[int, float] = {}
        for characters in line_characters:
            scoped_chars = [char for char in characters if type_cache[id(char)] in allowed_types]
            line_scores = self._build_global_scores(scoped_chars, style_cache)
            line_count = len(scoped_chars)
            for char in scoped_chars:
                global_score = float(global_scores.get(id(char), 0.0))
                line_score = float(line_scores.get(id(char), 0.0))
                if line_count >= 4 and global_count >= 6:
                    combined_scores[id(char)] = 0.65 * line_score + 0.35 * global_score
                elif line_count >= 3:
                    combined_scores[id(char)] = max(line_score, 0.6 * global_score)
                elif global_count >= 4:
                    combined_scores[id(char)] = global_score
                else:
                    combined_scores[id(char)] = max(line_score, 0.5 * global_score)
        return combined_scores

    def _build_candidate_stats(
        self,
        line_boxes: list[tuple[int, int, int, int]],
        line_characters: list[list[CharacterCandidate]],
        candidates: list[TamperRegion],
    ) -> CandidateStats:
        # 阈值尽量跟着当前图像的分布走，而不是只依赖写死的常量。
        scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float32)
        if scores.size == 0:
            score_median = 0.0
            score_mad = 0.0
        else:
            score_median = float(np.median(scores))
            score_mad = float(np.median(np.abs(scores - score_median)))

        type_scores: dict[str, list[float]] = {}
        for candidate in candidates:
            type_scores.setdefault(candidate.detection_type, []).append(float(candidate.score))

        line_widths = [box[2] for box in line_boxes]
        line_heights = [box[3] for box in line_boxes]
        char_widths = [char.width for characters in line_characters for char in characters]
        char_heights = [char.height for characters in line_characters for char in characters]
        gaps = [
            gap
            for characters in line_characters
            for gap in self._character_gaps(characters)
            if gap >= 0
        ]

        return CandidateStats(
            score_median=score_median,
            score_mad=score_mad,
            type_scores=type_scores,
            line_width_median=float(np.median(line_widths)) if line_widths else 0.0,
            line_height_median=float(np.median(line_heights)) if line_heights else 0.0,
            char_width_median=float(np.median(char_widths)) if char_widths else 0.0,
            char_height_median=float(np.median(char_heights)) if char_heights else 0.0,
            gap_median=float(np.median(gaps)) if gaps else 0.0,
        )

    @staticmethod
    def _adaptive_threshold_from_scores(
        scores: list[float],
        floor: float,
        percentile: float = 60.0,
        spread_weight: float = 0.25,
        slack: float = 0.0,
    ) -> float:
        if not scores:
            return floor

        values = np.asarray(scores, dtype=np.float32)
        if values.size == 1:
            return max(floor, float(values[0]) * 0.72)

        median = float(np.median(values))
        spread = float(np.median(np.abs(values - median)))
        adaptive = float(np.percentile(values, percentile)) - spread_weight * spread - slack
        return max(floor, adaptive)

    def _internal_group_scores(
        self,
        characters: list[CharacterCandidate],
    ) -> list[float]:
        if len(characters) < 3:
            return [0.0 for _ in characters]

        feature_matrix = np.asarray(
            [
                [
                    self._character_aspect_ratio(char),
                    self._character_density(char),
                    float(char.hole_count),
                    float(char.width),
                    float(char.height),
                ]
                for char in characters
            ],
            dtype=np.float32,
        )
        return self._robust_scores(feature_matrix)

    def _candidate_threshold(self, candidate: TamperRegion) -> float:
        support_count = int(candidate.detail.get("support_count", len(candidate.char_indices)))
        stats = getattr(self, "_candidate_stats", None)
        type_scores = stats.type_scores if stats is not None else {}
        line_width = int(stats.line_width_median) if stats is not None and stats.line_width_median > 0 else candidate.bbox[2]
        zone = str(candidate.detail.get("zone", ""))

        if candidate.detection_type == "digit_window":
            threshold = self._adaptive_threshold_from_scores(
                type_scores.get("digit_window", []),
                floor=self.DIGIT_WINDOW_THRESHOLD - 0.8,
                percentile=58.0,
                spread_weight=0.22,
            )
            if line_width < 240:
                threshold -= 0.6
            return max(9.8, threshold)

        if candidate.detection_type == "time_group":
            threshold = self._adaptive_threshold_from_scores(
                type_scores.get("time_group", []),
                floor=self.TIME_GROUP_THRESHOLD - 0.5,
                percentile=55.0,
                spread_weight=0.18,
            )
            if line_width < 360:
                threshold -= 0.55
            if support_count < 6:
                threshold -= 0.25
            if float(candidate.detail.get("context_bonus", 0.0)) <= 0.0:
                threshold += 0.1
            return max(4.1, threshold)

        if candidate.detection_type == "region_anomaly":
            threshold = self._adaptive_threshold_from_scores(
                type_scores.get("region_anomaly", []),
                floor=3.0,
                percentile=58.0,
                spread_weight=0.2,
            )
            window_size = int(candidate.detail.get("window_size", max(candidate.bbox[2], candidate.bbox[3])))
            if window_size <= 28:
                threshold -= 0.12
            elif window_size <= 80:
                threshold -= 0.06
            else:
                threshold -= 0.02
            if zone == "embedded_id_photo":
                threshold -= 1.2
            if hasattr(self, "_last_gray") and self._is_id_card_like(self._last_gray):
                threshold += 0.18
            return max(3.0, threshold)

        if candidate.detection_type == "text_region":
            if min(candidate.bbox[2], candidate.bbox[3]) < 16:
                return 100.0
            if candidate.bbox[2] * candidate.bbox[3] < 260 and support_count < 2:
                return 9.0
            threshold = self._adaptive_threshold_from_scores(
                type_scores.get("text_region", []),
                floor=4.0,
                percentile=54.0,
                spread_weight=0.22,
            )
            if zone.endswith("_precise"):
                threshold = min(threshold, 5.2)
            if stats is not None and stats.line_width_median > 0 and stats.line_width_median < 320:
                threshold -= 0.35
            return max(3.8, threshold)

        if candidate.detection_type == "text_noise_anomaly":
            if min(candidate.bbox[2], candidate.bbox[3]) < 8:
                return 100.0
            threshold = self._adaptive_threshold_from_scores(
                type_scores.get("text_noise_anomaly", []),
                floor=self.TEXT_NOISE_THRESHOLD,
                percentile=55.0,
                spread_weight=0.18,
            )
            if support_count >= 2:
                threshold -= 0.25
            return max(4.2, threshold)

        threshold = self._adaptive_threshold_from_scores(
            type_scores.get(candidate.detection_type, []),
            floor=4.2,
            percentile=56.0,
            spread_weight=0.22,
        )
        if line_width < 260:
            threshold -= 1.2
        if int(candidate.detail.get("line_count", 9)) <= 2:
            threshold -= 0.6
        if support_count <= 2:
            threshold -= 0.35
        if float(candidate.detail.get("width_ratio", 9.0)) < 1.8:
            threshold += 1.8
        if (
            float(candidate.detail.get("local_gap_ratio", 9.0)) < 0.4
            and float(candidate.detail.get("edge_spare_ratio", 9.0)) < 0.9
        ):
            threshold += 2.2
        return max(4.2, threshold)

    def _summarize_empty_result(
        self,
        line_boxes: list[tuple[int, int, int, int]],
        line_characters: list[list[CharacterCandidate]],
        ordered_candidates: list[TamperRegion],
    ) -> tuple[str, str]:
        total_chars = sum(len(characters) for characters in line_characters)
        max_line_chars = max((len(characters) for characters in line_characters), default=0)
        has_context = (
            len(line_boxes) >= 2
            or total_chars >= self.MIN_CONTEXT_TOTAL_CHARS
            or max_line_chars >= self.MIN_CONTEXT_LINE_CHARS
        )
        if not has_context:
            return "insufficient_context", "文本上下文不足"
        if ordered_candidates:
            return "no_detection", "候选存在但置信度不足"
        return "no_detection", "未发现满足规则的候选区域"

    def _selection_priority(self, candidate: TamperRegion) -> float:
        # 统一候选进入最终裁决时，不直接用原始分数做排序。
        # 原因是不同来源的分数尺度并不完全一致：
        # 通用滑窗候选容易出现数值偏大，而时间串、短中文、证件精确文本虽然更有语义价值，
        # 原始分数未必更高。因此这里额外加入“场景价值”与“小噪声惩罚”。
        priority = float(candidate.score)
        zone = str(candidate.detail.get("zone", ""))
        area = candidate.bbox[2] * candidate.bbox[3]
        support_count = int(candidate.detail.get("support_count", len(candidate.char_indices)))

        if candidate.detection_type in self.SPECIAL_RULE_TYPES:
            priority += 12.0

        if zone.endswith("_precise"):
            priority += 18.0
        elif zone == "embedded_id_photo":
            priority += 14.0
        elif zone == "generic_text_block":
            priority += 2.5
        elif zone == "text_noise_consistency":
            priority += 4.5

        # 很小的通用滑窗热点更容易是噪声或边缘碎片，
        # 在已经存在高质量专用候选时，应当明显后移。
        if candidate.detection_type == "region_anomaly":
            if area <= 1600:
                priority -= 18.0
            elif area <= 4000:
                priority -= 8.0
            elif zone != "embedded_id_photo" and area >= 80000:
                # 对特别大的通用局部异常框做额外惩罚，
                # 避免它覆盖整块版面并压住更贴近文本内容的候选。
                priority -= 12.0

        if candidate.detection_type == "text_region" and area < 1200 and support_count <= 1 and not zone.endswith("_precise"):
            priority -= 6.0

        if candidate.detection_type == "text_noise_anomaly":
            priority += 4.0
            if support_count <= 1:
                priority -= 1.5
            if area < 180:
                priority -= 2.0

        return priority

    def _select_regions(self, candidates: list[TamperRegion]) -> list[TamperRegion]:
        selected: list[TamperRegion] = []
        document_bbox = getattr(self, "_document_bbox", None)
        has_id_card_context = (
            hasattr(self, "_last_gray")
            and (
                self._is_id_card_like(self._last_gray)
                or (document_bbox is not None and self._bbox_looks_like_id_card(document_bbox))
            )
        )
        id_card_precise_text = (
            has_id_card_context
            and any(
                candidate.detection_type == "text_region"
                and str(candidate.detail.get("zone", "")).endswith("_precise")
                for candidate in candidates
            )
        )
        for candidate in sorted(candidates, key=self._selection_priority, reverse=True):
            if len(selected) >= self.MAX_OUTPUT_DETECTIONS:
                break

            candidate_zone = str(candidate.detail.get("zone", ""))
            if (
                id_card_precise_text
                and candidate.detection_type == "text_region"
                and not candidate_zone.endswith("_precise")
            ):
                continue

            if candidate.detection_type == "text_region":
                same_text_row_index = None
                for index, item in enumerate(selected):
                    if item.detection_type != "text_region":
                        continue
                    if abs((item.bbox[1] + item.bbox[3] / 2.0) - (candidate.bbox[1] + candidate.bbox[3] / 2.0)) <= 40:
                        same_text_row_index = index
                        break
                if same_text_row_index is not None:
                    existing = selected[same_text_row_index]
                    merged_box = self._union_boxes(existing.bbox, candidate.bbox)
                    horizontal_gap = max(
                        candidate.bbox[0] - (existing.bbox[0] + existing.bbox[2]),
                        existing.bbox[0] - (candidate.bbox[0] + candidate.bbox[2]),
                        0,
                    )
                    if horizontal_gap <= 28 and merged_box[2] <= 200:
                        selected[same_text_row_index] = TamperRegion(
                            detection_type="text_region",
                            line_index=-1,
                            bbox=merged_box,
                            char_indices=[],
                            score=max(existing.score, candidate.score) + 0.05,
                            label="文字篡改",
                            detail={
                                **existing.detail,
                                "support_count": int(existing.detail.get("support_count", 1))
                                + int(candidate.detail.get("support_count", 1)),
                            },
                        )
                        continue
                    existing_precise = str(existing.detail.get("zone", "")).endswith("_precise")
                    candidate_precise = candidate_zone.endswith("_precise")
                    if (
                        not existing_precise
                        and not candidate_precise
                        and candidate.bbox[0] > existing.bbox[0]
                        and candidate.score >= existing.score * 0.72
                    ):
                        selected[same_text_row_index] = candidate
                        continue
                    if not (existing_precise and candidate_precise):
                        continue

            if any(
                self._bbox_iou(candidate.bbox, item.bbox) > 0.2
                or self._bbox_overlap_ratio(candidate.bbox, item.bbox) > 0.72
                for item in selected
            ):
                continue

            selected.append(candidate)

        if not any(item.detection_type == "region_anomaly" for item in selected):
            if id_card_precise_text and sum(1 for item in selected if item.detection_type == "text_region") >= 4:
                return selected
            for candidate in candidates:
                if candidate.detection_type != "region_anomaly":
                    continue
                if candidate.score < self._candidate_threshold(candidate):
                    continue
                if any(
                    self._bbox_iou(candidate.bbox, item.bbox) > 0.2
                    or self._bbox_overlap_ratio(candidate.bbox, item.bbox) > 0.72
                    for item in selected
                ):
                    continue
                if len(selected) >= self.MAX_OUTPUT_DETECTIONS:
                    selected[-1] = candidate
                else:
                    selected.append(candidate)
                break
        return selected

    def _enumerate_digit_windows(
        self,
        gray: np.ndarray,
        line_boxes: list[tuple[int, int, int, int]],
        line_characters: list[list[CharacterCandidate]],
    ) -> list[WindowResult]:
        image_height = gray.shape[0]
        window_results: list[WindowResult] = []

        for line_index, (line_box, characters) in enumerate(zip(line_boxes, line_characters)):
            if len(characters) < 4:
                continue

            for run_start, run_end in self._enumerate_digit_runs(characters):
                run_length = run_end - run_start
                if run_length < 4:
                    continue

                for offset in range(run_length - 3):
                    window_start = run_start + offset
                    window_end = window_start + 4
                    window = characters[window_start:window_end]

                    gaps = self._character_gaps(window)
                    if gaps and max(gaps) > self.MAX_DIGIT_GAP:
                        continue

                    prev_char = characters[window_start - 1] if window_start > 0 else None
                    separator_gap = 0
                    if prev_char is not None:
                        separator_gap = window[0].bbox[0] - (prev_char.bbox[0] + prev_char.bbox[2])

                    method_scores = self._evaluate_methods(window)
                    method_outliers = {
                        method_name: int(np.argmax(scores))
                        for method_name, scores in method_scores.items()
                    }
                    outlier_index = self._choose_outlier(method_scores, method_outliers)
                    window_score = self._score_window(
                        method_scores=method_scores,
                        method_outliers=method_outliers,
                        outlier_index=outlier_index,
                        window=window,
                        separator_gap=separator_gap,
                        image_height=image_height,
                    )
                    window_score += self._context_adjustment(
                        characters=characters,
                        run_start=run_start,
                        run_end=run_end,
                        window_start=window_start,
                        window_end=window_end,
                    )

                    selected_char = window[outlier_index]
                    window_results.append(
                        WindowResult(
                            line_index=line_index,
                            line_box=line_box,
                            character_indices=[char.index_in_line for char in window],
                            outlier_index_in_window=outlier_index,
                            outlier_global_index=selected_char.index_in_line,
                            selected_bbox=selected_char.bbox,
                            method_scores=method_scores,
                            method_outliers=method_outliers,
                            window_score=window_score,
                        )
                    )
        return window_results

    def _enumerate_digit_runs(
        self,
        characters: list[CharacterCandidate],
    ) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        run_start: int | None = None

        for index, char in enumerate(characters):
            if self._is_digit_like(char):
                if run_start is None:
                    run_start = index
                continue

            if run_start is not None:
                runs.append((run_start, index))
                run_start = None

        if run_start is not None:
            runs.append((run_start, len(characters)))
        return runs

    def _enumerate_numeric_runs(
        self,
        characters: list[CharacterCandidate],
        type_cache: dict[int, str],
    ) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        run_start: int | None = None

        for index, char in enumerate(characters):
            if type_cache[id(char)] in {"digit", "separator"}:
                if run_start is None:
                    run_start = index
                continue

            if run_start is not None:
                runs.append((run_start, index))
                run_start = None

        if run_start is not None:
            runs.append((run_start, len(characters)))
        return runs

    def _context_adjustment(
        self,
        characters: list[CharacterCandidate],
        run_start: int,
        run_end: int,
        window_start: int,
        window_end: int,
    ) -> float:
        prefix_char = characters[run_start - 1] if run_start > 0 else None
        has_prefix_separator = prefix_char is not None and self._is_separator_like(prefix_char)
        has_decimal_suffix = self._has_decimal_suffix(characters, window_end)

        score = 0.0
        if has_prefix_separator:
            score += self.PREFIX_BONUS
        if has_decimal_suffix:
            score += self.DECIMAL_SUFFIX_BONUS

        score -= self.RUN_OFFSET_PENALTY * (window_start - run_start)
        score -= self.LONG_RUN_PENALTY * max((run_end - run_start) - 4, 0)

        if not has_prefix_separator and not has_decimal_suffix:
            score -= self.NO_CONTEXT_PENALTY
        return score

    def _has_decimal_suffix(
        self,
        characters: list[CharacterCandidate],
        window_end: int,
    ) -> bool:
        if window_end + 2 >= len(characters):
            return False

        dot_like = characters[window_end]
        suffix_digits = characters[window_end + 1 : window_end + 3]
        return self._is_separator_like(dot_like) and all(
            self._is_digit_like(char) for char in suffix_digits
        )

    def _classify_character(self, char: CharacterCandidate) -> str:
        if self._is_separator_like(char):
            return "separator"
        if self._is_context_digit_like(char):
            return "digit"
        if self._is_cjk_like(char):
            return "cjk"
        return "other"

    def _is_digit_like(self, char: CharacterCandidate) -> bool:
        aspect_ratio = self._character_aspect_ratio(char)
        foreground_density = self._character_density(char)
        return (
            self.DIGIT_MIN_WIDTH <= char.width <= self.DIGIT_MAX_WIDTH
            and 0.22 <= aspect_ratio <= 0.58
            and 0.18 <= foreground_density <= 0.42
            and char.hole_count <= 1
            and not self._is_separator_like(char)
        )

    def _is_context_digit_like(self, char: CharacterCandidate) -> bool:
        aspect_ratio = self._character_aspect_ratio(char)
        foreground_density = self._character_density(char)
        return (
            10 <= char.width <= 40
            and 0.16 <= aspect_ratio <= 0.75
            and 0.05 <= foreground_density <= 0.45
            and char.hole_count <= 2
            and not self._is_separator_like(char)
        )

    def _is_separator_like(self, char: CharacterCandidate) -> bool:
        aspect_ratio = self._character_aspect_ratio(char)
        foreground_density = self._character_density(char)
        return (
            char.width <= 13
            and aspect_ratio <= 0.30
            and foreground_density <= 0.22
            and char.hole_count == 0
        )

    def _is_cjk_like(self, char: CharacterCandidate) -> bool:
        aspect_ratio = self._character_aspect_ratio(char)
        foreground_density = self._character_density(char)
        return (
            char.width >= 14
            and char.height >= 18
            and 0.45 <= aspect_ratio <= 1.35
            and 0.10 <= foreground_density <= 0.58
        )

    def _style_vector(self, char: CharacterCandidate) -> np.ndarray:
        stroke = self._stroke_feature(char.patch)
        edge = self._edge_feature(char.patch)
        clahe = self._clahe_feature(char.patch)
        return np.asarray(
            [
                self._character_aspect_ratio(char),
                stroke[0],
                stroke[1],
                stroke[4],
                stroke[5],
                edge[0],
                clahe[0],
                clahe[2],
                clahe[4],
                clahe[5],
            ],
            dtype=np.float32,
        )

    def _build_global_scores(
        self,
        characters: list[CharacterCandidate],
        style_cache: dict[int, np.ndarray],
    ) -> dict[int, float]:
        if not characters:
            return {}
        if len(characters) == 1:
            return {id(characters[0]): 0.0}

        feature_matrix = np.stack([style_cache[id(char)] for char in characters], axis=0)
        if len(characters) >= 4:
            median = np.median(feature_matrix, axis=0)
            deviation = np.median(np.abs(feature_matrix - median), axis=0)
            deviation = np.where(deviation < 1e-6, 1.0, deviation)

            scores: dict[int, float] = {}
            for char in characters:
                normalized = np.abs((style_cache[id(char)] - median) / deviation)
                scores[id(char)] = float(np.mean(np.clip(normalized, 0.0, 1e6)))
            return scores

        scores: dict[int, float] = {}
        for index, char in enumerate(characters):
            others = np.delete(feature_matrix, index, axis=0)
            center = np.mean(others, axis=0)
            deviation = np.std(others, axis=0)
            deviation = np.where(deviation < 1e-6, 1.0, deviation)
            normalized = np.abs((style_cache[id(char)] - center) / deviation)
            scores[id(char)] = float(np.mean(np.clip(normalized, 0.0, 1e6)))
        return scores

    def _region_feature_vector(self, patch: np.ndarray) -> np.ndarray:
        patch = self._resize_patch(patch, size=48)
        edges = cv2.Canny(patch, 60, 140)
        laplacian = cv2.Laplacian(patch, cv2.CV_32F)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(patch)
        blackhat = cv2.morphologyEx(
            patch,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        )

        return np.asarray(
            [
                float(patch.mean() / 255.0),
                float(patch.std() / 255.0),
                float((edges > 0).mean()),
                float(np.mean(np.abs(laplacian)) / 255.0),
                float(np.mean(np.abs(clahe.astype(np.float32) - patch.astype(np.float32))) / 255.0),
                float(blackhat.mean() / 255.0),
            ],
            dtype=np.float32,
        )

    def _region_context_score(
        self,
        gray: np.ndarray,
        bbox: tuple[int, int, int, int],
        patch_feature: np.ndarray,
    ) -> float:
        x, y, w, h = bbox
        margin_x = max(12, w // 2)
        margin_y = max(12, h // 2)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(gray.shape[1], x + w + margin_x)
        y1 = min(gray.shape[0], y + h + margin_y)
        context = gray[y0:y1, x0:x1]
        if context.size == 0:
            return 0.0

        context_feature = self._region_feature_vector(context)
        scale = np.maximum(np.abs(context_feature), 0.08)
        return float(np.mean(np.abs((patch_feature - context_feature) / scale)))

    def _detect_document_bbox(
        self,
        gray: np.ndarray,
        image: np.ndarray | None,
    ) -> tuple[int, int, int, int] | None:
        if self._is_id_card_like(gray):
            return 0, 0, gray.shape[1], gray.shape[0]
        if image is None:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        _, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
            iterations=2,
        )

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        image_height, image_width = gray.shape
        image_area = float(image_height * image_width)
        best_bbox: tuple[int, int, int, int] | None = None
        best_score = 0.0

        for component_id in range(1, component_count):
            x, y, w, h, area = stats[component_id]
            ratio = w / max(h, 1)
            area_ratio = area / max(image_area, 1.0)
            fill_ratio = area / max(w * h, 1)
            if not (1.40 <= ratio <= 1.80):
                continue
            if area_ratio < 0.18 or fill_ratio < 0.52:
                continue
            if w < image_width * 0.45 or h < image_height * 0.25:
                continue
            score = area_ratio * fill_ratio
            if score <= best_score:
                continue

            margin = 10
            x0 = max(0, int(x) - margin)
            y0 = max(0, int(y) - margin)
            x1 = min(image_width, int(x + w) + margin)
            y1 = min(image_height, int(y + h) + margin)
            best_bbox = (x0, y0, x1 - x0, y1 - y0)
            best_score = score
        return best_bbox

    def _text_component_boxes(
        self,
        gray: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        )
        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        boxes: list[tuple[int, int, int, int]] = []
        for component_id in range(1, component_count):
            x, y, w, h, area = stats[component_id]
            if area < 30 or w < 6 or h < 10 or w > 120 or h > 120:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))
        return boxes

    def _snap_text_cluster_to_components(
        self,
        bbox: tuple[int, int, int, int],
        component_boxes: list[tuple[int, int, int, int]],
    ) -> tuple[tuple[int, int, int, int], int]:
        x, y, w, h = bbox
        expanded = (
            x - 10,
            y - 8,
            w + 20,
            h + 16,
        )
        cluster_center_y = y + h / 2.0
        selected = [
            component
            for component in component_boxes
            if abs((component[1] + component[3] / 2.0) - cluster_center_y) <= max(14.0, h)
            and (
                self._bbox_overlap_ratio(expanded, component) > 0.0
                or self._bbox_iou(expanded, component) > 0.0
                or self._center_distance(expanded, component) <= 22.0
            )
        ]
        if not selected:
            return bbox, 0

        merged: list[tuple[int, int, int, int]] = []
        for component in sorted(selected, key=lambda item: (item[1], item[0])):
            if not merged:
                merged.append(component)
                continue
            previous = merged[-1]
            gap = component[0] - (previous[0] + previous[2])
            y_overlap = min(previous[1] + previous[3], component[1] + component[3]) - max(previous[1], component[1])
            if gap <= 10 and y_overlap >= min(previous[3], component[3]) * 0.35:
                merged[-1] = self._union_boxes(previous, component)
            else:
                merged.append(component)

        anchor_x = x + w / 2.0
        anchor_y = y + h / 2.0
        ranked = sorted(
            merged,
            key=lambda item: (
                abs((item[0] + item[2] / 2.0) - anchor_x)
                + 0.8 * abs((item[1] + item[3] / 2.0) - anchor_y),
                -(item[2] * item[3]),
            ),
        )
        primary = ranked[0]
        snapped = primary
        support_count = 1
        for component in ranked[1:]:
            gap = component[0] - (snapped[0] + snapped[2])
            reverse_gap = snapped[0] - (component[0] + component[2])
            horizontal_gap = gap if gap >= 0 else reverse_gap
            y_overlap = min(snapped[1] + snapped[3], component[1] + component[3]) - max(snapped[1], component[1])
            if horizontal_gap <= 6 and y_overlap >= min(snapped[3], component[3]) * 0.3:
                snapped = self._union_boxes(snapped, component)
                support_count += 1
        return snapped, support_count

    def _snap_large_region_to_rectangle(
        self,
        bbox: tuple[int, int, int, int],
        window_size: object,
    ) -> tuple[int, int, int, int]:
        try:
            window = int(window_size)
        except (TypeError, ValueError):
            window = 0
        if window < 90:
            return bbox

        x, y, w, h = bbox
        x0 = max(0, x - 40)
        y0 = max(0, y - 40)
        x1 = min(self._last_gray_shape[1], x + w + 40)
        y1 = min(self._last_gray_shape[0], y + h + 40)
        roi = self._last_gray[y0:y1, x0:x1]
        edges = cv2.Canny(roi, 80, 160)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = bbox
        best_score = 0.0
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            area = rw * rh
            if area < 18000:
                continue
            ratio = rw / max(rh, 1)
            if not (0.55 <= ratio <= 1.1):
                continue
            candidate_box = (x0 + rx, y0 + ry, rw, rh)
            overlap = self._bbox_overlap_ratio(candidate_box, bbox)
            if overlap <= 0.08:
                continue
            score = area * overlap
            if score > best_score:
                best_score = score
                best_box = candidate_box
        return best_box

    @staticmethod
    def _center_distance(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        ax = box_a[0] + box_a[2] / 2.0
        ay = box_a[1] + box_a[3] / 2.0
        bx = box_b[0] + box_b[2] / 2.0
        by = box_b[1] + box_b[3] / 2.0
        return float(np.hypot(ax - bx, ay - by))

    def _character_aspect_ratio(self, char: CharacterCandidate) -> float:
        return float(char.width / max(char.height, 1))

    def _character_density(self, char: CharacterCandidate) -> float:
        mask = self._patch_mask(self._resize_patch(char.patch))
        return float((mask > 0).mean())

    def _character_gaps(self, characters: list[CharacterCandidate]) -> list[int]:
        return [
            characters[idx + 1].bbox[0] - (characters[idx].bbox[0] + characters[idx].bbox[2])
            for idx in range(len(characters) - 1)
        ]

    def _union_bbox(
        self,
        characters: list[CharacterCandidate],
    ) -> tuple[int, int, int, int]:
        x0 = min(char.bbox[0] for char in characters)
        y0 = min(char.bbox[1] for char in characters)
        x1 = max(char.bbox[0] + char.bbox[2] for char in characters)
        y1 = max(char.bbox[1] + char.bbox[3] for char in characters)
        return x0, y0, x1 - x0, y1 - y0

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
    def _center_x(box: tuple[int, int, int, int]) -> float:
        return box[0] + box[2] / 2.0

    @staticmethod
    def _center_y(box: tuple[int, int, int, int]) -> float:
        return box[1] + box[3] / 2.0

    def _center_in_ratio_box(
        self,
        bbox: tuple[int, int, int, int],
        width: int,
        height: int,
        x0_ratio: float,
        x1_ratio: float,
        y0_ratio: float,
        y1_ratio: float,
    ) -> bool:
        center_x = self._center_x(bbox)
        center_y = self._center_y(bbox)
        return (
            width * x0_ratio <= center_x <= width * x1_ratio
            and height * y0_ratio <= center_y <= height * y1_ratio
        )

    @staticmethod
    def _is_id_card_like(gray: np.ndarray) -> bool:
        height, width = gray.shape
        ratio = width / max(height, 1)
        return 1.45 <= ratio <= 1.75 and width >= 900 and height >= 520

    @staticmethod
    def _bbox_looks_like_id_card(
        bbox: tuple[int, int, int, int],
    ) -> bool:
        _, _, width, height = bbox
        ratio = width / max(height, 1)
        return 1.45 <= ratio <= 1.75 and width >= 700 and height >= 420

    @staticmethod
    def _union_boxes(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x0 = min(box_a[0], box_b[0])
        y0 = min(box_a[1], box_b[1])
        x1 = max(box_a[0] + box_a[2], box_b[0] + box_b[2])
        y1 = max(box_a[1] + box_a[3], box_b[1] + box_b[3])
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def _bbox_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        ax0, ay0, aw, ah = box_a
        bx0, by0, bw, bh = box_b
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh

        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            return 0.0

        intersection = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        union = aw * ah + bw * bh - intersection
        return float(intersection / max(union, 1))

    @staticmethod
    def _bbox_overlap_ratio(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        ax0, ay0, aw, ah = box_a
        bx0, by0, bw, bh = box_b
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh

        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            return 0.0

        intersection = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        return float(intersection / max(min(aw * ah, bw * bh), 1))

    def _choose_outlier(
        self,
        method_scores: dict[str, list[float]],
        method_outliers: dict[str, int],
    ) -> int:
        vote_counter: Counter[int] = Counter()
        score_counter: Counter[int] = Counter()
        for method_name, outlier_index in method_outliers.items():
            weight = self.METHOD_WEIGHTS[method_name]
            vote_counter[outlier_index] += weight
            score_counter[outlier_index] += weight * np.log1p(method_scores[method_name][outlier_index])

        return max(
            range(4),
            key=lambda idx: (vote_counter[idx], score_counter[idx]),
        )

    def _score_window(
        self,
        method_scores: dict[str, list[float]],
        method_outliers: dict[str, int],
        outlier_index: int,
        window: list[CharacterCandidate],
        separator_gap: int,
        image_height: int,
    ) -> float:
        weighted_score = 0.0
        agreement_score = 0.0
        for method_name, scores in method_scores.items():
            weight = self.METHOD_WEIGHTS[method_name]
            weighted_score += weight * np.log1p(scores[outlier_index])
            if method_outliers[method_name] == outlier_index:
                agreement_score += weight

        center_y = window[outlier_index].bbox[1] + window[outlier_index].bbox[3] / 2.0
        bottom_bonus = 1.8 * (center_y / image_height)
        gap_bonus = 1.1 * min(separator_gap / 24.0, 1.0)

        hole_penalty = 0.0
        other_holes = sum(char.hole_count for idx, char in enumerate(window) if idx != outlier_index)
        if window[outlier_index].hole_count > 0 and other_holes == 0:
            hole_penalty = 1.25

        return weighted_score + agreement_score + bottom_bonus + gap_bonus - hole_penalty

    def _evaluate_methods(self, window: list[CharacterCandidate]) -> dict[str, list[float]]:
        features = {
            "texture": [self._texture_feature(char.patch) for char in window],
            "edge": [self._edge_feature(char.patch) for char in window],
            "jpeg": [self._jpeg_feature(char.patch) for char in window],
            "stroke": [self._stroke_feature(char.patch) for char in window],
            "clahe": [self._clahe_feature(char.patch) for char in window],
        }
        return {
            method_name: self._robust_scores(np.asarray(values, dtype=np.float32))
            for method_name, values in features.items()
        }

    @staticmethod
    def _resize_patch(patch: np.ndarray, size: int = 32) -> np.ndarray:
        height, width = patch.shape
        scale = min((size - 4) / max(height, 1), (size - 4) / max(width, 1))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            patch,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        canvas = np.full((size, size), 255, dtype=np.uint8)
        offset_x = (size - new_width) // 2
        offset_y = (size - new_height) // 2
        canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized
        return canvas

    @staticmethod
    def _patch_mask(patch: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(
            cv2.GaussianBlur(patch, (3, 3), 0),
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        return binary

    def _texture_feature(self, patch: np.ndarray) -> np.ndarray:
        patch = self._resize_patch(patch)
        equalized = cv2.equalizeHist(patch)
        lbp = _uniform_lbp(equalized)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

        grad_x = cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        angle = np.mod(angle, 180.0)
        hog_hist, _ = np.histogram(
            angle.ravel(),
            bins=9,
            range=(0, 180),
            weights=magnitude.ravel(),
            density=True,
        )
        return np.concatenate([lbp_hist, hog_hist])

    def _edge_feature(self, patch: np.ndarray) -> np.ndarray:
        patch = self._resize_patch(patch)
        edges = cv2.Canny(patch, 60, 140)
        edge_density = float(edges.mean() / 255.0)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        largest_contour = max((len(contour) for contour in contours), default=0)
        continuity = largest_contour / max(int(np.count_nonzero(edges)), 1)

        grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        angle = np.mod(angle, 180.0)
        orientation_hist, _ = np.histogram(
            angle.ravel(),
            bins=8,
            range=(0, 180),
            weights=(magnitude > 20).astype(np.float32).ravel(),
            density=True,
        )
        return np.concatenate([[edge_density, continuity], orientation_hist])

    def _jpeg_feature(self, patch: np.ndarray) -> np.ndarray:
        patch = self._resize_patch(patch, size=32).astype(np.float32) - 128.0
        blocks: list[list[float]] = []
        for y in range(0, 32, 8):
            for x in range(0, 32, 8):
                block = patch[y : y + 8, x : x + 8]
                dct = cv2.dct(block)
                magnitude = np.abs(dct)
                magnitude[0, 0] = 0
                blocks.append(
                    [
                        float(magnitude[0:3, 0:3].mean()),
                        float(magnitude[3:, 3:].mean()),
                        float(np.abs(np.diff(block, axis=1)).mean()),
                    ]
                )
        block_array = np.asarray(blocks, dtype=np.float32)
        grid = block_array.reshape(4, 4, -1)
        horizontal_diff = np.abs(grid[:, 1:, :] - grid[:, :-1, :]).mean(axis=(0, 1))
        vertical_diff = np.abs(grid[1:, :, :] - grid[:-1, :, :]).mean(axis=(0, 1))
        return np.concatenate(
            [
                block_array.mean(axis=0),
                block_array.std(axis=0),
                horizontal_diff,
                vertical_diff,
            ]
        )

    def _stroke_feature(self, patch: np.ndarray) -> np.ndarray:
        patch = self._resize_patch(patch)
        mask = self._patch_mask(patch)
        skeleton = _skeletonize_mask(mask)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        widths = distance[skeleton] * 2.0
        if widths.size == 0:
            widths = np.array([0.0], dtype=np.float32)
        return np.asarray(
            [
                widths.mean(),
                widths.std(),
                np.median(widths),
                np.percentile(widths, 90),
                float((mask > 0).mean()),
                float(skeleton.mean()),
            ],
            dtype=np.float32,
        )

    def _clahe_feature(self, patch: np.ndarray) -> np.ndarray:
        patch = self._resize_patch(patch)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(patch)
        residual = np.abs(enhanced.astype(np.float32) - patch.astype(np.float32))
        foreground_mask = self._patch_mask(patch) > 0
        foreground = residual[foreground_mask] if np.any(foreground_mask) else residual.ravel()
        background = residual[~foreground_mask] if np.any(~foreground_mask) else residual.ravel()
        return np.asarray(
            [
                foreground.mean(),
                foreground.std(),
                background.mean(),
                background.std(),
                residual.mean(),
                patch.std(),
                enhanced.std(),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _robust_scores(feature_matrix: np.ndarray) -> list[float]:
        median = np.median(feature_matrix, axis=0)
        deviation = np.median(np.abs(feature_matrix - median), axis=0)
        deviation = np.where(deviation < 1e-6, 1.0, deviation)
        scores = np.mean(np.abs((feature_matrix - median) / deviation), axis=1)
        scores = np.clip(scores, 0.0, 1e6)
        return [float(value) for value in scores]


def visualize_detection(
    image: np.ndarray,
    result: DetectionResult,
) -> np.ndarray:
    output = image.copy()
    overlay = output.copy()
    palette = [
        (0, 255, 255),
        (0, 165, 255),
        (255, 128, 0),
        (0, 220, 0),
    ]

    for index, region in enumerate(result.detections):
        x, y, w, h = region.bbox
        color = palette[index % len(palette)]
        cv2.rectangle(overlay, (x - 6, y - 6), (x + w + 6, y + h + 6), color, -1)
        cv2.rectangle(output, (x - 6, y - 6), (x + w + 6, y + h + 6), color, 2)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return cv2.addWeighted(overlay, 0.16, output, 0.84, 0)
