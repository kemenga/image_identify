from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# 文本行分割后的最小字符单元。
# 这里保留几何信息、裁切图块和孔洞数量，供后续字符类型判定与异常打分复用。
@dataclass(slots=True)
class CharacterCandidate:
    line_index: int
    index_in_line: int
    bbox: tuple[int, int, int, int]
    patch: np.ndarray
    width: int
    height: int
    hole_count: int


# 统一在预处理阶段完成图片读取与灰度化，
# 后续检测逻辑默认直接消费灰度图。
def load_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_text_lines(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    # 文本行检测的核心思路是：
    # 1. 先用模糊 + 自适应阈值把深色文字变成稳定的前景；
    # 2. 用横向较宽的闭运算把同一行里的字符连接起来；
    # 3. 再用连通域和经验阈值过滤掉明显不像文本行的区域。
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    inverse_binary = cv2.adaptiveThreshold(
        255 - blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -8,
    )

    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    connected_lines = cv2.morphologyEx(
        inverse_binary,
        cv2.MORPH_CLOSE,
        line_kernel,
        iterations=1,
    )

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(connected_lines, 8)
    boxes: list[tuple[int, int, int, int]] = []
    image_height, image_width = gray.shape
    min_width = max(24, int(image_width * 0.18))
    min_height = max(14, int(image_height * 0.04))
    max_height = max(min_height + 10, int(image_height * 0.65))
    min_area = max(120, int(image_height * image_width * 0.0012))
    for component_id in range(1, component_count):
        x, y, w, h, area = stats[component_id]
        width_ratio = w / max(image_width, 1)
        fill_ratio = area / max(w * h, 1)
        # 第一轮筛选偏保守，优先保留“横向较长、填充率像文本行”的连通域。
        if (
            w >= min_width
            and min_height <= h <= max_height
            and area >= min_area
            and width_ratio >= 0.16
            and fill_ratio >= 0.2
        ):
            boxes.append((int(x), int(y), int(w), int(h)))

    if boxes:
        return sorted(boxes, key=lambda item: (item[1], item[0]))

    relaxed_boxes: list[tuple[int, int, int, int]] = []
    relaxed_min_width = max(16, int(image_width * 0.12))
    relaxed_min_area = max(64, int(image_height * image_width * 0.001))
    for component_id in range(1, component_count):
        x, y, w, h, area = stats[component_id]
        fill_ratio = area / max(w * h, 1)
        # 裁切图或窄图上下文更少，因此这里放宽宽度、面积和填充率阈值，
        # 让检测器在局部截图场景下仍能拿到最基本的文本上下文。
        if (
            w >= relaxed_min_width
            and 10 <= h <= max(image_height - 2, 10)
            and area >= relaxed_min_area
            and fill_ratio >= 0.08
        ):
            relaxed_boxes.append((int(x), int(y), int(w), int(h)))

    if relaxed_boxes:
        return sorted(relaxed_boxes, key=lambda item: (item[1], item[0]))

    active_rows = np.where((inverse_binary > 0).sum(axis=1) > max(2, int(image_width * 0.04)))[0]
    if active_rows.size == 0:
        return []

    # 当前两轮连通域都失败时，退化成“至少给出一整条活跃行区域”，
    # 避免后续检测流程直接因为没有文本框而中断。
    y0 = max(0, int(active_rows.min()) - 2)
    y1 = min(image_height, int(active_rows.max()) + 3)
    return [(0, y0, image_width, max(1, y1 - y0))]


def _count_holes(patch: np.ndarray) -> int:
    # 孔洞数是区分数字、分隔符和某些中文形态的一个廉价结构特征。
    # 例如“0/6/8/9”通常更容易出现内部孔洞，而点号、短横线通常没有。
    _, binary = cv2.threshold(
        cv2.GaussianBlur(patch, (3, 3), 0),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    contour_result = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = contour_result[-1]
    if hierarchy is None:
        return 0
    return sum(1 for entry in hierarchy[0] if entry[3] != -1)


def segment_line_characters(
    gray: np.ndarray,
    line_box: tuple[int, int, int, int],
    line_index: int,
) -> list[CharacterCandidate]:
    # 单字符切分基于纵向投影完成：
    # 先在文本行 ROI 内得到二值前景，再找“活跃列”的连续区段作为字符候选。
    x, y, w, h = line_box
    roi = gray[y : y + h, x : x + w]

    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    inverse_binary = cv2.adaptiveThreshold(
        255 - blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        -6,
    )
    inverse_binary = cv2.morphologyEx(
        inverse_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    projection = (inverse_binary > 0).sum(axis=0)
    active_columns = projection > max(2, int(0.12 * inverse_binary.shape[0]))

    raw_segments: list[tuple[int, int]] = []
    start: int | None = None
    for column, is_active in enumerate(active_columns):
        if is_active and start is None:
            start = column
        elif not is_active and start is not None:
            if column - start >= 5:
                raw_segments.append((start, column - 1))
            start = None
    if start is not None and len(active_columns) - start >= 5:
        raw_segments.append((start, len(active_columns) - 1))

    # 某些字符在局部噪声或弱笔画影响下会被切成相邻小段，
    # 这里把间隔很小的段重新合并，减少过分碎片化。
    merged_segments: list[list[int]] = []
    for seg_start, seg_end in raw_segments:
        if not merged_segments or seg_start - merged_segments[-1][1] > 3:
            merged_segments.append([seg_start, seg_end])
        else:
            merged_segments[-1][1] = seg_end

    characters: list[CharacterCandidate] = []
    for index_in_line, (seg_start, seg_end) in enumerate(merged_segments):
        # 左右各留少量边距，让后续纹理、边缘和笔画宽度特征有更完整的上下文。
        x0 = max(0, seg_start - 2)
        x1 = min(roi.shape[1], seg_end + 3)
        patch = roi[:, x0:x1]
        ys, xs = np.where((255 - patch) > 40)
        if len(xs) == 0 or len(ys) == 0:
            continue

        # 最后按真实前景重新裁切上下边界，避免把大块空白带入字符特征。
        y0 = int(ys.min())
        y1 = int(ys.max() + 1)
        trimmed_patch = patch[y0:y1, :]
        bbox = (x + x0, y + y0, x1 - x0, y1 - y0)
        characters.append(
            CharacterCandidate(
                line_index=line_index,
                index_in_line=index_in_line,
                bbox=bbox,
                patch=trimmed_patch,
                width=x1 - x0,
                height=y1 - y0,
                hole_count=_count_holes(trimmed_patch),
            )
        )
    return characters
