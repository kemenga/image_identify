from __future__ import annotations

import argparse
import html
import io
import json
import random
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluation import discover_image_pairs, extract_ground_truth_boxes
from preprocessing import detect_text_lines, segment_line_characters


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "seed_dataset"
DEFAULT_DOC_KEYWORDS = [
    "receipt",
    "invoice",
    "ticket",
    "form",
    "menu",
    "poster",
    "notice",
    "certificate",
    "bill",
    "label",
]
DEFAULT_NATURAL_KEYWORDS = [
    "storefront",
    "desk",
    "food",
    "street",
    "vehicle",
    "room",
    "product",
    "package",
    "shelf",
    "sign",
]
DOC_TAMPER_METHODS = [
    "text_patch_replace",
    "cross_doc_splice",
    "copy_move_token",
]
NATURAL_TAMPER_METHODS = [
    "copy_move_region",
    "cross_image_splice",
    "erase_and_fill",
]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
REJECT_KEYWORDS = (
    "edited",
    "photoshop",
    "composite",
    "collage",
    "manipulation",
    "render",
    "ai-generated",
    "aigenerated",
    "ai generated",
)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 Codex Dataset Builder",
}


@dataclass(slots=True)
class DatasetPaths:
    root: Path
    manifests_dir: Path
    originals_docs_dir: Path
    originals_natural_dir: Path
    tampered_dir: Path
    labels_png_dir: Path
    labels_json_dir: Path
    search_manifest_path: Path
    original_manifest_path: Path
    tamper_manifest_path: Path
    verify_report_path: Path


def dataset_paths(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> DatasetPaths:
    root = Path(dataset_root)
    manifests_dir = root / "manifests"
    return DatasetPaths(
        root=root,
        manifests_dir=manifests_dir,
        originals_docs_dir=root / "originals" / "docs",
        originals_natural_dir=root / "originals" / "natural",
        tampered_dir=root / "tampered",
        labels_png_dir=root / "labels_png",
        labels_json_dir=root / "labels_json",
        search_manifest_path=manifests_dir / "search_manifest.json",
        original_manifest_path=manifests_dir / "original_manifest.json",
        tamper_manifest_path=manifests_dir / "tamper_manifest.json",
        verify_report_path=manifests_dir / "verify_report.json",
    )


def _now_string() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _ensure_dataset_dirs(paths: DatasetPaths) -> None:
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    paths.originals_docs_dir.mkdir(parents=True, exist_ok=True)
    paths.originals_natural_dir.mkdir(parents=True, exist_ok=True)
    paths.tampered_dir.mkdir(parents=True, exist_ok=True)
    paths.labels_png_dir.mkdir(parents=True, exist_ok=True)
    paths.labels_json_dir.mkdir(parents=True, exist_ok=True)


def _relative_to_root(paths: DatasetPaths, path: Path) -> str:
    return str(path.relative_to(paths.root))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _request_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _fetch_binary(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read()


def _decode_image_bytes(data: bytes) -> np.ndarray | None:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _read_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"无法写入图片: {path}")


def _strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _file_extension_from_url(url: str, fallback_title: str = "") -> str:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower().lstrip(".")
    if suffix:
        return suffix
    title_suffix = Path(fallback_title).suffix.lower().lstrip(".")
    return title_suffix


def _display_license(raw_license: str, version: str = "") -> str:
    license_text = (raw_license or "").strip()
    version_text = (version or "").strip()
    if not license_text:
        return ""
    if version_text and version_text not in license_text:
        return f"{license_text} {version_text}".strip()
    return license_text


def _is_allowed_license(license_text: str) -> bool:
    value = license_text.lower().strip()
    compact = value.replace(" ", "").replace("_", "-")
    if not value:
        return False
    if "nc" in compact or "nd" in compact:
        return False
    return any(
        marker in compact
        for marker in (
            "cc0",
            "publicdomain",
            "public-domain",
            "cc-by",
            "ccby",
            "by-sa",
            "bysa",
            "pdm",
            "cc-zero",
        )
    )


def _contains_reject_terms(*values: str) -> bool:
    text = " ".join(_strip_html(value).lower() for value in values if value)
    return any(term in text for term in REJECT_KEYWORDS)


def _candidate_filter_reason(
    *,
    title: str,
    description: str,
    license_text: str,
    width: int,
    height: int,
    extension: str,
) -> str | None:
    if not _is_allowed_license(license_text):
        return "许可不在白名单"
    if extension.lower() not in ALLOWED_EXTENSIONS:
        return "文件格式不支持"
    if min(width, height) < 1024:
        return "分辨率不足"
    if _contains_reject_terms(title, description):
        return "标题或描述包含疑似已编辑关键词"
    return None


def _candidate_sort_key(candidate: dict[str, object]) -> tuple[int, int, str]:
    width = int(candidate.get("width", 0))
    height = int(candidate.get("height", 0))
    provider = str(candidate.get("provider", ""))
    return (min(width, height), width * height, provider)


def _search_openverse(
    keyword: str,
    page: int,
    page_size: int,
    source_type: str,
) -> tuple[list[dict[str, object]], Counter]:
    params = urllib.parse.urlencode(
        {
            "q": keyword,
            "page": page,
            "page_size": page_size,
        }
    )
    data = _request_json(f"https://api.openverse.org/v1/images/?{params}")
    accepted: list[dict[str, object]] = []
    rejections: Counter = Counter()
    for item in data.get("results", []):
        title = _strip_html(str(item.get("title") or ""))
        description = _strip_html(str(item.get("description") or ""))
        download_url = str(item.get("url") or "")
        source_page_url = str(item.get("foreign_landing_url") or "")
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        license_text = _display_license(
            str(item.get("license") or ""),
            str(item.get("license_version") or ""),
        )
        extension = _file_extension_from_url(download_url, title)
        reason = _candidate_filter_reason(
            title=title,
            description=description,
            license_text=license_text,
            width=width,
            height=height,
            extension=extension,
        )
        if reason:
            rejections[reason] += 1
            continue
        accepted.append(
            {
                "provider": "openverse",
                "source_type": source_type,
                "keyword": keyword,
                "title": title or "未命名图片",
                "description": description,
                "license": license_text,
                "license_url": str(item.get("license_url") or ""),
                "source_page_url": source_page_url,
                "download_url": download_url,
                "creator": str(item.get("creator") or ""),
                "width": width,
                "height": height,
                "extension": extension.lower(),
            }
        )
    return accepted, rejections


def _search_wikimedia(
    keyword: str,
    page: int,
    page_size: int,
    source_type: str,
) -> tuple[list[dict[str, object]], Counter]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": keyword,
        "gsrnamespace": "6",
        "gsrlimit": str(page_size),
        "gsroffset": str((page - 1) * page_size),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "format": "json",
    }
    data = _request_json(
        "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    )
    pages = data.get("query", {}).get("pages", {})
    accepted: list[dict[str, object]] = []
    rejections: Counter = Counter()
    for page_item in pages.values():
        info = (page_item.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata") or {}
        title = _strip_html(str(page_item.get("title") or ""))
        description = _strip_html(str(metadata.get("ImageDescription", {}).get("value") or ""))
        download_url = str(info.get("url") or "")
        source_page_url = str(info.get("descriptionurl") or "")
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        license_text = _strip_html(
            str(metadata.get("LicenseShortName", {}).get("value") or "")
        ) or _strip_html(str(metadata.get("License", {}).get("value") or ""))
        extension = _file_extension_from_url(download_url, title)
        reason = _candidate_filter_reason(
            title=title,
            description=description,
            license_text=license_text,
            width=width,
            height=height,
            extension=extension,
        )
        if reason:
            rejections[reason] += 1
            continue
        accepted.append(
            {
                "provider": "wikimedia",
                "source_type": source_type,
                "keyword": keyword,
                "title": title or "未命名图片",
                "description": description,
                "license": license_text,
                "license_url": _strip_html(
                    str(metadata.get("LicenseUrl", {}).get("value") or "")
                ),
                "source_page_url": source_page_url,
                "download_url": download_url,
                "creator": _strip_html(str(metadata.get("Artist", {}).get("value") or "")),
                "width": width,
                "height": height,
                "extension": extension.lower(),
            }
        )
    return accepted, rejections


def search_sources(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    docs_target: int = 10,
    natural_target: int = 10,
    page_size: int = 12,
    max_pages: int = 3,
    oversample_factor: int = 6,
    doc_keywords: list[str] | None = None,
    natural_keywords: list[str] | None = None,
) -> dict[str, object]:
    paths = dataset_paths(dataset_root)
    _ensure_dataset_dirs(paths)
    keywords_by_type = {
        "docs": doc_keywords or list(DEFAULT_DOC_KEYWORDS),
        "natural": natural_keywords or list(DEFAULT_NATURAL_KEYWORDS),
    }
    target_by_type = {"docs": docs_target, "natural": natural_target}
    candidates: list[dict[str, object]] = []
    provider_errors: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for source_type, keywords in keywords_by_type.items():
        accepted_items: list[dict[str, object]] = []
        seen_urls: set[tuple[str, str]] = set()
        rejection_counter: Counter = Counter()
        target_pool_size = max(1, target_by_type[source_type] * max(1, oversample_factor))
        for keyword in keywords:
            if len(accepted_items) >= target_pool_size:
                break
            for page in range(1, max_pages + 1):
                for provider_name, provider_search in (
                    ("openverse", _search_openverse),
                    ("wikimedia", _search_wikimedia),
                ):
                    try:
                        provider_items, rejections = provider_search(
                            keyword=keyword,
                            page=page,
                            page_size=page_size,
                            source_type=source_type,
                        )
                    except Exception as exc:
                        provider_errors.append(
                            {
                                "provider": provider_name,
                                "source_type": source_type,
                                "keyword": keyword,
                                "page": page,
                                "error": str(exc),
                            }
                        )
                        continue
                    rejection_counter.update(rejections)
                    for item in provider_items:
                        unique_key = (str(item["provider"]), str(item["download_url"]))
                        if unique_key in seen_urls:
                            continue
                        seen_urls.add(unique_key)
                        accepted_items.append(item)
                        if len(accepted_items) >= target_pool_size:
                            break
                    if len(accepted_items) >= target_pool_size:
                        break
                if len(accepted_items) >= target_pool_size:
                    break
        accepted_items = sorted(accepted_items, key=_candidate_sort_key, reverse=True)
        for index, item in enumerate(accepted_items, start=1):
            prefix = "doc" if source_type == "docs" else "natural"
            item["candidate_id"] = f"{prefix}_candidate_{index:04d}"
        candidates.extend(accepted_items)
        summary[source_type] = {
            "keyword_count": len(keywords),
            "candidate_count": len(accepted_items),
            "target_original_count": target_by_type[source_type],
            "rejections": dict(rejection_counter),
        }

    manifest = {
        "generated_at": _now_string(),
        "dataset_root": str(paths.root),
        "search_config": {
            "docs_target": docs_target,
            "natural_target": natural_target,
            "page_size": page_size,
            "max_pages": max_pages,
            "oversample_factor": oversample_factor,
            "doc_keywords": keywords_by_type["docs"],
            "natural_keywords": keywords_by_type["natural"],
        },
        "provider_errors": provider_errors,
        "summary": summary,
        "candidates": candidates,
    }
    _write_json(paths.search_manifest_path, manifest)
    return manifest


def _dhash_hex(image: np.ndarray, size: int = 8) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    difference = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if value else "0" for value in difference.flatten())
    return f"{int(bits, 2):0{size * size // 4}x}"


def _hamming_distance(lhs: str, rhs: str) -> int:
    lhs_value = int(lhs, 16)
    rhs_value = int(rhs, 16)
    return (lhs_value ^ rhs_value).bit_count()


def _looks_like_document(image: np.ndarray) -> tuple[bool, dict[str, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    line_boxes = detect_text_lines(gray)
    character_count = 0
    for index, line_box in enumerate(line_boxes[:8]):
        character_count += len(segment_line_characters(gray, line_box, index))
    line_area = sum(box[2] * box[3] for box in line_boxes[:8])
    image_area = image.shape[0] * image.shape[1]
    stats = {
        "line_count": len(line_boxes),
        "character_count": int(character_count),
        "line_area_ratio_percent": int(round(100 * line_area / max(image_area, 1))),
    }
    accepted = len(line_boxes) >= 2 and character_count >= 8 and line_area >= image_area * 0.03
    return accepted, stats


def download_originals(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    docs_target: int = 10,
    natural_target: int = 10,
    search_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    paths = dataset_paths(dataset_root)
    _ensure_dataset_dirs(paths)
    search_path = Path(search_manifest_path) if search_manifest_path else paths.search_manifest_path
    if not search_path.exists():
        raise FileNotFoundError(f"缺少搜索清单: {search_path}")
    search_manifest = _read_json(search_path)
    candidates = list(search_manifest.get("candidates", []))
    target_by_type = {"docs": docs_target, "natural": natural_target}
    output_counts: Counter = Counter()
    rejection_counter: Counter = Counter()
    seen_hashes: dict[str, list[str]] = defaultdict(list)
    records: list[dict[str, object]] = []
    for candidate in candidates:
        source_type = str(candidate["source_type"])
        if output_counts[source_type] >= target_by_type[source_type]:
            continue
        try:
            data = _fetch_binary(str(candidate["download_url"]))
            image = _decode_image_bytes(data)
        except Exception as exc:
            rejection_counter[f"下载失败: {exc.__class__.__name__}"] += 1
            continue
        if image is None:
            rejection_counter["图片解码失败"] += 1
            continue
        height, width = image.shape[:2]
        if min(width, height) < 1024:
            rejection_counter["实际分辨率不足"] += 1
            continue
        if source_type == "docs":
            is_document, document_stats = _looks_like_document(image)
            if not is_document:
                rejection_counter["文档文字密度不足"] += 1
                continue
        else:
            document_stats = {
                "line_count": 0,
                "character_count": 0,
                "line_area_ratio_percent": 0,
            }
        image_hash = _dhash_hex(image)
        if any(_hamming_distance(image_hash, existing_hash) <= 4 for existing_hash in seen_hashes[source_type]):
            rejection_counter["感知哈希重复"] += 1
            continue
        seen_hashes[source_type].append(image_hash)
        output_counts[source_type] += 1
        prefix = "doc" if source_type == "docs" else "natural"
        asset_id = f"{prefix}_{output_counts[source_type]:04d}"
        output_dir = paths.originals_docs_dir if source_type == "docs" else paths.originals_natural_dir
        output_path = output_dir / f"{asset_id}_orig.png"
        _write_png(output_path, image)
        records.append(
            {
                "id": asset_id,
                "source_type": source_type,
                "provider": candidate["provider"],
                "keyword": candidate["keyword"],
                "title": candidate["title"],
                "description": candidate["description"],
                "license": candidate["license"],
                "license_url": candidate["license_url"],
                "source_page_url": candidate["source_page_url"],
                "download_url": candidate["download_url"],
                "creator": candidate["creator"],
                "width": width,
                "height": height,
                "dhash": image_hash,
                "original_image": _relative_to_root(paths, output_path),
                "document_stats": document_stats,
            }
        )

    manifest = {
        "generated_at": _now_string(),
        "dataset_root": str(paths.root),
        "target_counts": target_by_type,
        "actual_counts": dict(output_counts),
        "rejections": dict(rejection_counter),
        "records": records,
    }
    _write_json(paths.original_manifest_path, manifest)
    return manifest


def _union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _clip_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    image_height, image_width = image_shape[:2]
    x0 = max(0, min(image_width - 1, x))
    y0 = max(0, min(image_height - 1, y))
    x1 = max(1, min(image_width, x + w))
    y1 = max(1, min(image_height, y + h))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _overlap_ratio(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> float:
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


def _expand_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return _clip_box((x - pad_x, y - pad_y, w + pad_x * 2, h + pad_y * 2), image_shape)


def _extract_document_token_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    token_boxes: list[tuple[int, int, int, int]] = []
    line_boxes = detect_text_lines(gray)
    for line_index, line_box in enumerate(line_boxes[:16]):
        characters = segment_line_characters(gray, line_box, line_index)
        if not characters:
            x, y, w, h = line_box
            slice_width = max(20, int(h * 1.8))
            current_x = x
            while current_x < x + w:
                token_boxes.append((current_x, y, min(slice_width, x + w - current_x), h))
                current_x += slice_width + max(4, int(h * 0.25))
            continue
        boxes = [candidate.bbox for candidate in characters]
        gaps = [
            boxes[index + 1][0] - (boxes[index][0] + boxes[index][2])
            for index in range(len(boxes) - 1)
        ]
        positive_gaps = [gap for gap in gaps if gap > 0]
        gap_threshold = max(6, int(np.median(positive_gaps) * 1.8)) if positive_gaps else 8
        current_group = [boxes[0]]
        for index, next_box in enumerate(boxes[1:], start=1):
            gap = boxes[index][0] - (boxes[index - 1][0] + boxes[index - 1][2])
            if gap > gap_threshold:
                token_boxes.append(_union_boxes(current_group))
                current_group = [next_box]
            else:
                current_group.append(next_box)
        if current_group:
            token_boxes.append(_union_boxes(current_group))
    filtered = []
    for box in token_boxes:
        _, _, w, h = box
        area = w * h
        if w >= 14 and h >= 10 and area >= 180:
            filtered.append(box)
    return filtered


def _choose_document_box(gray: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    token_boxes = _extract_document_token_boxes(gray)
    if not token_boxes:
        height, width = gray.shape
        fallback = (
            max(0, width // 5),
            max(0, height // 4),
            max(40, width // 5),
            max(24, height // 18),
        )
        return _clip_box(fallback, (height, width, 3))
    sorted_boxes = sorted(token_boxes, key=lambda item: item[2] * item[3])
    middle_start = max(0, len(sorted_boxes) // 4)
    middle_end = max(middle_start + 1, len(sorted_boxes) * 3 // 4)
    return rng.choice(sorted_boxes[middle_start:middle_end])


def _background_fill_from_border(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    expanded = _expand_box(box, image.shape, 6, 4)
    x, y, w, h = box
    ex, ey, ew, eh = expanded
    neighborhood = image[ey : ey + eh, ex : ex + ew].copy()
    inner_x0 = max(0, x - ex)
    inner_y0 = max(0, y - ey)
    inner_x1 = min(neighborhood.shape[1], inner_x0 + w)
    inner_y1 = min(neighborhood.shape[0], inner_y0 + h)
    mask = np.ones(neighborhood.shape[:2], dtype=bool)
    mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
    if mask.any():
        border_pixels = neighborhood[mask]
        median_color = np.median(border_pixels.reshape(-1, 3), axis=0)
    else:
        median_color = np.array([245.0, 245.0, 245.0], dtype=np.float32)
    fill = np.tile(median_color.astype(np.uint8), (h, w, 1))
    return cv2.GaussianBlur(fill, (3, 3), 0)


def _fit_text_to_box(
    text: str,
    box: tuple[int, int, int, int],
) -> tuple[float, int]:
    _, _, w, h = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    for font_scale in np.linspace(max(0.45, h / 22.0), 0.35, 12):
        thickness = max(1, int(round(font_scale * 1.8)))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        if text_width <= max(10, w - 4) and text_height + baseline <= max(10, h - 2):
            return float(font_scale), thickness
    return 0.35, 1


def _draw_replacement_text(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    x, y, w, h = box
    char_count = max(2, min(8, int(round(w / max(h * 0.58, 1.0)))))
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    replacement = "".join(rng.choice(alphabet) for _ in range(char_count))
    font_scale, thickness = _fit_text_to_box(replacement, box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(replacement, font, font_scale, thickness)
    text_x = x + max(2, (w - text_width) // 2)
    text_y = y + max(text_height + 1, (h + text_height) // 2)
    cv2.putText(
        image,
        replacement,
        (text_x, text_y),
        font,
        font_scale,
        (32, 32, 32),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _paste_patch_with_soft_edge(
    canvas: np.ndarray,
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    feather: int = 5,
) -> None:
    x, y, w, h = box
    resized_patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
    roi = canvas[y : y + h, x : x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (1, 1), (max(1, w - 2), max(1, h - 2)), 255, thickness=-1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = resized_patch.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    canvas[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)


def _candidate_texture_boxes(image: np.ndarray, max_boxes: int = 24) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplace = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    image_height, image_width = gray.shape
    short_side = min(image_height, image_width)
    sizes = sorted(
        {
            max(40, int(short_side * 0.10)),
            max(56, int(short_side * 0.14)),
            max(72, int(short_side * 0.18)),
        }
    )
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for size in sizes:
        step = max(20, size // 2)
        for y in range(size // 3, max(size // 3 + 1, image_height - size), step):
            for x in range(size // 3, max(size // 3 + 1, image_width - size), step):
                patch = gray[y : y + size, x : x + size]
                edge_score = float(laplace[y : y + size, x : x + size].mean())
                std_score = float(patch.std())
                brightness = float(patch.mean())
                if std_score < 12.0 or not (18.0 <= brightness <= 240.0):
                    continue
                scored.append((edge_score * 1.8 + std_score, (x, y, size, size)))
    selected: list[tuple[int, int, int, int]] = []
    for _, box in sorted(scored, key=lambda item: item[0], reverse=True):
        if any(_overlap_ratio(box, existing) > 0.35 for existing in selected):
            continue
        selected.append(box)
        if len(selected) >= max_boxes:
            break
    if selected:
        return selected
    fallback_size = max(48, short_side // 6)
    return [
        _clip_box(
            (
                image_width // 3,
                image_height // 3,
                fallback_size,
                fallback_size,
            ),
            image.shape,
        )
    ]


def _choose_natural_target_box(
    image: np.ndarray,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    candidates = _candidate_texture_boxes(image)
    top_slice = max(1, min(8, len(candidates)))
    return rng.choice(candidates[:top_slice])


def _choose_non_overlapping_box(
    candidates: list[tuple[int, int, int, int]],
    forbidden: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    valid = [candidate for candidate in candidates if _overlap_ratio(candidate, forbidden) < 0.1]
    if valid:
        top_slice = max(1, min(8, len(valid)))
        return rng.choice(valid[:top_slice])
    x, y, w, h = forbidden
    shift_x = max(12, w // 2)
    return x + shift_x, y, w, h


def _tamper_text_patch_replace(
    image: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    base_box = _choose_document_box(gray, rng)
    tamper_box = _expand_box(base_box, image.shape, 3, 2)
    tampered = image.copy()
    fill = _background_fill_from_border(tampered, tamper_box)
    x, y, w, h = tamper_box
    tampered[y : y + h, x : x + w] = fill
    _draw_replacement_text(tampered, tamper_box, rng)
    return tampered, [tamper_box], "局部遮盖并写入新的短文本"


def _tamper_cross_doc_splice(
    image: np.ndarray,
    donor_image: np.ndarray | None,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    target_box = _expand_box(_choose_document_box(gray, rng), image.shape, 3, 2)
    donor = donor_image if donor_image is not None else image
    donor_gray = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
    donor_box = _expand_box(_choose_document_box(donor_gray, rng), donor.shape, 3, 2)
    dx, dy, dw, dh = donor_box
    donor_patch = donor[dy : dy + dh, dx : dx + dw]
    tampered = image.copy()
    _paste_patch_with_soft_edge(tampered, donor_patch, target_box, feather=2)
    note = "跨文档裁切文本块拼接"
    if donor_image is None:
        note = "缺少第二张文档图，退化为同图文本块拼接"
    return tampered, [target_box], note


def _tamper_copy_move_token(
    image: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    token_boxes = _extract_document_token_boxes(gray)
    if len(token_boxes) >= 2:
        sorted_boxes = sorted(token_boxes, key=lambda item: item[2] * item[3], reverse=True)
        source_box = sorted_boxes[0]
        target_box = next(
            (box for box in sorted_boxes[1:] if _overlap_ratio(box, source_box) < 0.08),
            sorted_boxes[-1],
        )
    else:
        source_box = _choose_document_box(gray, rng)
        target_box = _expand_box(
            (source_box[0] + source_box[2] + 8, source_box[1], source_box[2], source_box[3]),
            image.shape,
            0,
            0,
        )
    sx, sy, sw, sh = _expand_box(source_box, image.shape, 2, 2)
    patch = image[sy : sy + sh, sx : sx + sw]
    final_box = _expand_box(target_box, image.shape, 2, 2)
    tampered = image.copy()
    _paste_patch_with_soft_edge(tampered, patch, final_box, feather=1)
    return tampered, [final_box], "复制同图文本块并移动覆盖"


def _tamper_copy_move_region(
    image: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], str]:
    candidates = _candidate_texture_boxes(image)
    source_box = _choose_natural_target_box(image, rng)
    target_box = _clip_box(_choose_non_overlapping_box(candidates, source_box, rng), image.shape)
    sx, sy, sw, sh = source_box
    patch = image[sy : sy + sh, sx : sx + sw]
    tampered = image.copy()
    _paste_patch_with_soft_edge(tampered, patch, target_box, feather=4)
    return tampered, [target_box], "复制同图局部区域并移动覆盖"


def _tamper_cross_image_splice(
    image: np.ndarray,
    donor_image: np.ndarray | None,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], str]:
    target_box = _choose_natural_target_box(image, rng)
    donor = donor_image if donor_image is not None else image
    donor_box = _choose_natural_target_box(donor, rng)
    dx, dy, dw, dh = donor_box
    donor_patch = donor[dy : dy + dh, dx : dx + dw]
    tampered = image.copy()
    x, y, w, h = target_box
    resized_patch = cv2.resize(donor_patch, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.full((h, w), 255, dtype=np.uint8)
    center = (x + w // 2, y + h // 2)
    try:
        tampered = cv2.seamlessClone(resized_patch, tampered, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        _paste_patch_with_soft_edge(tampered, resized_patch, target_box, feather=5)
    note = "跨图拼接局部区域"
    if donor_image is None:
        note = "缺少第二张自然图，退化为同图区域拼接"
    return tampered, [target_box], note


def _tamper_erase_and_fill(
    image: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], str]:
    target_box = _choose_natural_target_box(image, rng)
    x, y, w, h = target_box
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    radius_x = max(10, w // 2)
    radius_y = max(10, h // 2)
    center = (x + w // 2, y + h // 2)
    cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 255, thickness=-1)
    tampered = cv2.inpaint(image, mask, max(3, min(w, h) // 5), cv2.INPAINT_TELEA)
    return tampered, [target_box], "局部擦除后使用修补填充"


def _load_original_records(dataset_root: str | Path) -> tuple[DatasetPaths, list[dict[str, object]]]:
    paths = dataset_paths(dataset_root)
    if not paths.original_manifest_path.exists():
        raise FileNotFoundError(f"缺少原图清单: {paths.original_manifest_path}")
    manifest = _read_json(paths.original_manifest_path)
    return paths, list(manifest.get("records", []))


def generate_tampered_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    seed: int = 42,
) -> dict[str, object]:
    paths, original_records = _load_original_records(dataset_root)
    _ensure_dataset_dirs(paths)
    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in original_records:
        by_type[str(record["source_type"])].append(record)

    tamper_records: list[dict[str, object]] = []
    counts: Counter = Counter()
    for record in original_records:
        source_type = str(record["source_type"])
        original_path = paths.root / str(record["original_image"])
        image = _read_color_image(original_path)
        donors = [item for item in by_type[source_type] if item["id"] != record["id"]]
        methods = DOC_TAMPER_METHODS if source_type == "docs" else NATURAL_TAMPER_METHODS
        for method_index, method_name in enumerate(methods, start=1):
            item_seed = seed + len(tamper_records) * 1009 + method_index * 97
            rng = random.Random(item_seed)
            donor_image = None
            if donors:
                donor_record = rng.choice(donors)
                donor_path = paths.root / str(donor_record["original_image"])
                donor_image = _read_color_image(donor_path)
            if method_name == "text_patch_replace":
                tampered_image, boxes, notes = _tamper_text_patch_replace(image, rng)
            elif method_name == "cross_doc_splice":
                tampered_image, boxes, notes = _tamper_cross_doc_splice(image, donor_image, rng)
            elif method_name == "copy_move_token":
                tampered_image, boxes, notes = _tamper_copy_move_token(image, rng)
            elif method_name == "copy_move_region":
                tampered_image, boxes, notes = _tamper_copy_move_region(image, rng)
            elif method_name == "cross_image_splice":
                tampered_image, boxes, notes = _tamper_cross_image_splice(image, donor_image, rng)
            elif method_name == "erase_and_fill":
                tampered_image, boxes, notes = _tamper_erase_and_fill(image, rng)
            else:
                raise ValueError(f"未知篡改方法: {method_name}")

            tampered_stem = f"{record['id']}_t{method_index:02d}_{method_name}"
            tampered_path = paths.tampered_dir / f"{tampered_stem}.png"
            _write_png(tampered_path, tampered_image)
            tamper_records.append(
                {
                    "id": tampered_stem,
                    "source_type": source_type,
                    "original_image": str(record["original_image"]),
                    "tampered_image": _relative_to_root(paths, tampered_path),
                    "label_png": "",
                    "label_json": "",
                    "license": record["license"],
                    "license_url": record["license_url"],
                    "source_page_url": record["source_page_url"],
                    "download_url": record["download_url"],
                    "tamper_method": method_name,
                    "boxes": [
                        {
                            "x": int(box[0]),
                            "y": int(box[1]),
                            "w": int(box[2]),
                            "h": int(box[3]),
                            "role": "tampered_region",
                        }
                        for box in boxes
                    ],
                    "notes": notes,
                }
            )
            counts[method_name] += 1

    manifest = {
        "generated_at": _now_string(),
        "dataset_root": str(paths.root),
        "seed": seed,
        "record_count": len(tamper_records),
        "method_counts": dict(counts),
        "records": tamper_records,
    }
    _write_json(paths.tamper_manifest_path, manifest)
    return manifest


def generate_labels(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> dict[str, object]:
    paths = dataset_paths(dataset_root)
    if not paths.tamper_manifest_path.exists():
        raise FileNotFoundError(f"缺少篡改清单: {paths.tamper_manifest_path}")
    manifest = _read_json(paths.tamper_manifest_path)
    records = list(manifest.get("records", []))
    for record in records:
        tampered_path = paths.root / str(record["tampered_image"])
        image = _read_color_image(tampered_path)
        label_image = image.copy()
        for box in record["boxes"]:
            x = int(box["x"])
            y = int(box["y"])
            w = int(box["w"])
            h = int(box["h"])
            cv2.rectangle(label_image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        tampered_stem = Path(str(record["tampered_image"])).stem
        label_png_path = paths.labels_png_dir / f"{tampered_stem}检测结果.png"
        label_json_path = paths.labels_json_dir / f"{tampered_stem}.json"
        _write_png(label_png_path, label_image)
        label_json = {
            "id": record["id"],
            "source_type": record["source_type"],
            "original_image": record["original_image"],
            "tampered_image": record["tampered_image"],
            "label_png": _relative_to_root(paths, label_png_path),
            "license": record["license"],
            "license_url": record["license_url"],
            "source_page_url": record["source_page_url"],
            "download_url": record["download_url"],
            "tamper_method": record["tamper_method"],
            "boxes": record["boxes"],
            "notes": record["notes"],
        }
        _write_json(label_json_path, label_json)
        record["label_png"] = _relative_to_root(paths, label_png_path)
        record["label_json"] = _relative_to_root(paths, label_json_path)
    manifest["generated_at"] = _now_string()
    manifest["records"] = records
    _write_json(paths.tamper_manifest_path, manifest)
    return manifest


def verify_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> dict[str, object]:
    paths = dataset_paths(dataset_root)
    issues: list[dict[str, object]] = []
    duplicates: list[dict[str, object]] = []
    original_manifest = _read_json(paths.original_manifest_path) if paths.original_manifest_path.exists() else {"records": []}
    tamper_manifest = _read_json(paths.tamper_manifest_path) if paths.tamper_manifest_path.exists() else {"records": []}
    original_records = list(original_manifest.get("records", []))
    tamper_records = list(tamper_manifest.get("records", []))
    original_hashes: list[tuple[str, str]] = []

    for record in original_records:
        image_path = paths.root / str(record["original_image"])
        if not image_path.exists():
            issues.append({"type": "missing_original", "path": str(image_path)})
            continue
        image = _read_color_image(image_path)
        image_hash = _dhash_hex(image)
        for existing_id, existing_hash in original_hashes:
            if _hamming_distance(image_hash, existing_hash) <= 4:
                duplicates.append(
                    {
                        "lhs": existing_id,
                        "rhs": record["id"],
                        "reason": "原图感知哈希过近",
                    }
                )
        original_hashes.append((str(record["id"]), image_hash))

    for record in tamper_records:
        original_path = paths.root / str(record["original_image"])
        tampered_path = paths.root / str(record["tampered_image"])
        if not original_path.exists():
            issues.append({"type": "missing_original_for_tamper", "path": str(original_path)})
            continue
        if not tampered_path.exists():
            issues.append({"type": "missing_tampered", "path": str(tampered_path)})
            continue
        original_image = _read_color_image(original_path)
        tampered_image = _read_color_image(tampered_path)
        if original_image.shape[:2] != tampered_image.shape[:2]:
            issues.append(
                {
                    "type": "size_mismatch",
                    "item": record["id"],
                    "original_shape": list(original_image.shape[:2]),
                    "tampered_shape": list(tampered_image.shape[:2]),
                }
            )
        label_png = str(record.get("label_png") or "")
        if label_png:
            label_path = paths.root / label_png
            if not label_path.exists():
                issues.append({"type": "missing_label_png", "path": str(label_path)})
            else:
                label_image = _read_color_image(label_path)
                if label_image.shape[:2] != tampered_image.shape[:2]:
                    issues.append(
                        {
                            "type": "label_size_mismatch",
                            "item": record["id"],
                            "label_shape": list(label_image.shape[:2]),
                            "tampered_shape": list(tampered_image.shape[:2]),
                        }
                    )
        else:
            issues.append({"type": "missing_label_png_path", "item": record["id"]})

        label_json = str(record.get("label_json") or "")
        if label_json:
            label_json_path = paths.root / label_json
            if not label_json_path.exists():
                issues.append({"type": "missing_label_json", "path": str(label_json_path)})
            else:
                label_data = _read_json(label_json_path)
                for box in label_data.get("boxes", []):
                    x = int(box["x"])
                    y = int(box["y"])
                    w = int(box["w"])
                    h = int(box["h"])
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        issues.append({"type": "invalid_box", "item": record["id"], "box": box})
                    if x + w > tampered_image.shape[1] or y + h > tampered_image.shape[0]:
                        issues.append({"type": "box_out_of_bounds", "item": record["id"], "box": box})
        else:
            issues.append({"type": "missing_label_json_path", "item": record["id"]})

    extracted_boxes = 0
    extraction_failures: list[dict[str, object]] = []
    for pair in discover_image_pairs(paths.root):
        try:
            boxes, _ = extract_ground_truth_boxes(pair.image_path, pair.answer_path)
            extracted_boxes += len(boxes)
        except Exception as exc:
            extraction_failures.append(
                {
                    "image": pair.image_path.name,
                    "answer": pair.answer_path.name,
                    "error": str(exc),
                }
            )
    passed = not issues and not duplicates and not extraction_failures
    report = {
        "generated_at": _now_string(),
        "dataset_root": str(paths.root),
        "original_count": len(original_records),
        "tampered_count": len(tamper_records),
        "pair_count": len(discover_image_pairs(paths.root)),
        "extracted_ground_truth_box_count": extracted_boxes,
        "duplicates": duplicates,
        "issues": issues,
        "extraction_failures": extraction_failures,
        "passed": passed,
    }
    _write_json(paths.verify_report_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="开放许可原图搜索与本地篡改数据集生成")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="数据集输出根目录，默认 ./seed_dataset",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="搜索开放许可原图候选")
    search_parser.add_argument("--docs-target", type=int, default=10, help="文档类原图目标数量")
    search_parser.add_argument("--natural-target", type=int, default=10, help="自然图原图目标数量")
    search_parser.add_argument("--page-size", type=int, default=12, help="每页抓取数量")
    search_parser.add_argument("--max-pages", type=int, default=3, help="每个关键词最多抓取页数")
    search_parser.add_argument(
        "--oversample-factor",
        type=int,
        default=6,
        help="搜索候选池相对目标数量的放大倍数",
    )

    download_parser = subparsers.add_parser("download", help="下载并筛选原图")
    download_parser.add_argument("--docs-target", type=int, default=10, help="文档类原图目标数量")
    download_parser.add_argument("--natural-target", type=int, default=10, help="自然图原图目标数量")
    download_parser.add_argument(
        "--search-manifest",
        default="",
        help="可选，自定义搜索清单路径",
    )

    tamper_parser = subparsers.add_parser("tamper", help="生成篡改图")
    tamper_parser.add_argument("--seed", type=int, default=42, help="随机种子")

    subparsers.add_parser("label", help="生成 PNG 标签和 JSON 标签")
    subparsers.add_parser("verify", help="检查尺寸、标签一致性和配对可评测性")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    if args.command == "search":
        manifest = search_sources(
            dataset_root=dataset_root,
            docs_target=args.docs_target,
            natural_target=args.natural_target,
            page_size=args.page_size,
            max_pages=args.max_pages,
            oversample_factor=args.oversample_factor,
        )
        print("搜索完成")
        print(f"候选总数: {len(manifest['candidates'])}")
        print(f"搜索清单: {dataset_paths(dataset_root).search_manifest_path}")
        return

    if args.command == "download":
        manifest = download_originals(
            dataset_root=dataset_root,
            docs_target=args.docs_target,
            natural_target=args.natural_target,
            search_manifest_path=args.search_manifest or None,
        )
        print("下载完成")
        print(f"文档类原图: {manifest['actual_counts'].get('docs', 0)}")
        print(f"自然图原图: {manifest['actual_counts'].get('natural', 0)}")
        print(f"原图清单: {dataset_paths(dataset_root).original_manifest_path}")
        return

    if args.command == "tamper":
        manifest = generate_tampered_dataset(dataset_root=dataset_root, seed=args.seed)
        print("篡改图生成完成")
        print(f"篡改图数量: {manifest['record_count']}")
        print(f"篡改清单: {dataset_paths(dataset_root).tamper_manifest_path}")
        return

    if args.command == "label":
        manifest = generate_labels(dataset_root=dataset_root)
        print("标签生成完成")
        print(f"带标签的篡改图数量: {len(manifest['records'])}")
        print(f"标签目录: {dataset_paths(dataset_root).labels_png_dir}")
        return

    if args.command == "verify":
        report = verify_dataset(dataset_root=dataset_root)
        print("校验完成")
        print(f"是否通过: {report['passed']}")
        print(f"配对数量: {report['pair_count']}")
        print(f"校验报告: {dataset_paths(dataset_root).verify_report_path}")
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
