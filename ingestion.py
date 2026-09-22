"""Document Ingestion Module for QASubjectBot.

Handles:
1. High-performance multi-format document loading (PDF, Word DOCX, PowerPoint PPTX, TXT/MD files).
2. Large PDF book streaming & memory optimization via PyMuPDF (fitz) with lazy page evaluation.
3. Scanned PDF detection and intelligent OCR fallback (PyMuPDF OCR / pytesseract).
4. Native PDF and Word table extraction with Markdown grid formatting.
5. Embedded media extraction (pictures/diagrams/figures saved to static/extracted_images).
6. Calculation and mathematical formula detection.
7. Text cleaning (stripping headers, footers, page numbers, line-broken hyphens, excess whitespace).
8. Token-aware document chunking (350-700 tokens with ~80-100 token overlap via tiktoken).
9. Chunk metadata preservation (source filename, page/slide number, chunk index, token count, images, tables, OCR tags, chunk ID).
10. Live progress callback support for asynchronous upload tracking.
"""

from dataclasses import dataclass, asdict
import gc
import io
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

from PIL import Image, ImageStat
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
DEFAULT_CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data", "processed_chunks.json")

try:
    import pymupdf  # PyMuPDF 1.24+
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats a 2D table grid into a clean GitHub-Flavored Markdown table."""
    if not headers and not rows:
        return ""
    if not headers and rows:
        headers = [f"Col {i+1}" for i in range(len(rows[0]))]

    num_cols = max(len(headers), max((len(r) for r in rows), default=0))
    if num_cols == 0:
        return ""

    padded_headers = [
        str(headers[i]).strip().replace("\n", " ") if i < len(headers) and headers[i] is not None else f"Col {i+1}"
        for i in range(num_cols)
    ]
    padded_rows = []
    for row in rows:
        r_cells = [
            str(row[i]).strip().replace("\n", " ") if i < len(row) and row[i] is not None else ""
            for i in range(num_cols)
        ]
        if any(r_cells):
            padded_rows.append(r_cells)

    col_widths = []
    for c_idx in range(num_cols):
        max_w = len(str(padded_headers[c_idx]))
        for r in padded_rows:
            max_w = max(max_w, len(str(r[c_idx])))
        col_widths.append(max(max_w, 3))

    lines = []
    header_str = "| " + " | ".join(str(padded_headers[i]).ljust(col_widths[i]) for i in range(num_cols)) + " |"
    sep_str = "| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |"
    lines.append(header_str)
    lines.append(sep_str)
    for r in padded_rows:
        row_str = "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(num_cols)) + " |"
        lines.append(row_str)

    return "\n" + "\n".join(lines) + "\n"


@dataclass
class DocumentChunk:
    """Represents a discrete text chunk with comprehensive source, media, and OCR metadata."""

    text: str
    metadata: Dict[str, Any]

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> int:
        return self.metadata.get("page", 1)

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)

    @property
    def token_count(self) -> int:
        return self.metadata.get("token_count", 0)

    @property
    def doc_type(self) -> str:
        if "doc_type" in self.metadata:
            return self.metadata["doc_type"]
        ext = os.path.splitext(self.source)[1].lower().lstrip(".")
        return ext if ext in {"pdf", "docx", "pptx", "txt", "md"} else "pdf"

    @property
    def unit_label(self) -> str:
        if "unit_label" in self.metadata:
            return self.metadata["unit_label"]
        return "Slide" if self.doc_type == "pptx" else "Page"

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", f"{self.source}:p{self.page}:c{self.chunk_index}")

    @property
    def images(self) -> List[Dict[str, Any]]:
        return self.metadata.get("images", [])

    @property
    def has_images(self) -> bool:
        return bool(self.metadata.get("images"))

    @property
    def has_tables(self) -> bool:
        return bool(self.metadata.get("has_tables", False))

    @property
    def has_calculations(self) -> bool:
        return bool(self.metadata.get("has_calculations", False))

    @property
    def is_ocr(self) -> bool:
        return bool(self.metadata.get("is_ocr", False))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(text=data["text"], metadata=data["metadata"])


class TextCleaner:
    """Cleans extracted raw document text:
    - Removes running headers and footers
    - Removes standalone page numbers ('Page X of Y', 'Page X', '- 3 -')
    - Removes boilerplate notices and divider bars
    - Reconnects hyphenated word breaks ('trans-\\nformer' -> 'transformer')
    - Normalizes unicode whitespace and quotation marks
    - Normalizes excess blank lines
    """

    HEADER_FOOTER_PATTERNS = [
        r"^QASubjectBot\s+Technical\s+Knowledge\s+Series.*$",
        r"^Confidential\s*&\s*Proprietary\s*-\s*AI\s*Research\s*Lab\s*Reference\s*Corpus.*$",
        r"^[=\-_]{4,}.*$",
        r"^Author:\s*.*$",
        r"^Classification:\s*.*$",
        r"^Version:\s*\d+(\.\d+)*.*$",
    ]

    PAGE_NUMBER_PATTERNS = [
        r"(?i)^Page\s+\d+(\s+of\s+\d+)?\s*$",
        r"^\[\s*Page\s+\d+\s*\]$",
        r"^[-—–]\s*\d+\s*[-—–]$",
        r"^\d+\s*/\s*\d+$",
    ]

    def __init__(self):
        self._header_footer_regexes = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.HEADER_FOOTER_PATTERNS
        ]
        self._page_num_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.PAGE_NUMBER_PATTERNS
        ]

    def clean(self, text: str) -> str:
        """Cleans and normalizes raw extracted page text."""
        if not text:
            return ""

        # 1. Normalize line endings and unicode whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ").replace("\u200b", "")

        # Normalize private-use area characters to bullets
        text = re.sub(r"[\ue000-\uf8ff]", " • ", text)

        # 2. Fix hyphenated line breaks (e.g. "trans-\nformer" -> "transformer")
        text = re.sub(r"(\b[a-zA-Z]{2,})-\n([a-zA-Z]{2,}\b)", r"\1\2", text)

        # 3. Process line-by-line to filter headers, footers, and page numbers
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                cleaned_lines.append("")
                continue

            # Check if line is a page number
            is_page_num = any(regex.match(stripped_line) for regex in self._page_num_regexes)
            if is_page_num:
                continue

            # Check if line matches known running headers/footers/boilerplate
            is_header_footer = any(
                regex.match(stripped_line) for regex in self._header_footer_regexes
            )
            if is_header_footer:
                continue

            # Inline removal of "Page X of Y" if embedded at start/end of line
            stripped_line = re.sub(r"(?i)\bPage\s+\d+\s+of\s+\d+\b", "", stripped_line).strip()

            if stripped_line:
                cleaned_lines.append(stripped_line)

        # 4. Reconstruct text and collapse multiple empty lines (max 2 newlines)
        reconstructed = "\n".join(cleaned_lines)
        reconstructed = re.sub(r"\n{3,}", "\n\n", reconstructed)

        # 5. Normalize consecutive spaces (preserve single spaces)
        reconstructed = re.sub(r"[ \t]{2,}", " ", reconstructed)

        return reconstructed.strip()


class MediaExtractor:
    """Extracts embedded pictures/diagrams and figures from PDF, Word, and PowerPoint files."""

    @staticmethod
    def sanitize_name(name: str) -> str:
        base = os.path.splitext(os.path.basename(name))[0]
        return re.sub(r"[^a-zA-Z0-9_-]", "_", base)

    @classmethod
    def extract_media(
        cls,
        filepath: str,
        output_base_dir: Optional[str] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Extracts useful images/diagrams from PDF, PPTX and DOCX.

        PDF filtering includes:
        - Very small image / icon removal (< 80px or < 1000 bytes)
        - Very thin / long object removal (aspect ratio > 10)
        - Almost completely black / white image removal
        - Full-page screenshot / background canvas removal (covers >= 95% of page)
        - Duplicate image removal via MD5
        - PNG normalization for clean browser rendering
        """
        if output_base_dir is None:
            output_base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "static",
                "extracted_images",
            )

        ext = os.path.splitext(filepath)[1].lower()
        doc_folder = cls.sanitize_name(filepath)
        target_dir = os.path.join(output_base_dir, doc_folder)
        os.makedirs(target_dir, exist_ok=True)

        # Remove previously extracted images for this document to avoid stale artifacts
        try:
            for old_file in os.listdir(target_dir):
                old_path = os.path.join(target_dir, old_file)
                if os.path.isfile(old_path):
                    os.remove(old_path)
        except Exception as cleanup_error:
            print(f"[MEDIA CLEANUP WARNING] {cleanup_error}")

        media_by_page: Dict[int, List[Dict[str, Any]]] = {}

        # =========================================================
        # PDF
        # =========================================================
        if ext == ".pdf":
            if pymupdf is not None:
                try:
                    doc = pymupdf.open(filepath)
                    seen_hashes = set()

                    for page_idx in range(len(doc)):
                        page_num = page_idx + 1
                        page = doc[page_idx]
                        page_images = []

                        try:
                            image_infos = page.get_image_info(xrefs=True)
                        except Exception:
                            image_infos = []

                        for img_idx, img_info in enumerate(image_infos):
                            try:
                                xref = img_info.get("xref", 0)
                                if not xref:
                                    continue

                                # 1. Bounding box & full-page screenshot check
                                bbox = img_info.get("bbox")
                                if bbox:
                                    if hasattr(bbox, "x0"):
                                        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                                    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                                    else:
                                        x0, y0, x1, y1 = 0, 0, 0, 0

                                    bbox_w = max(0.0, float(x1 - x0))
                                    bbox_h = max(0.0, float(y1 - y0))

                                    page_rect = page.rect
                                    page_w = max(1.0, float(page_rect.width))
                                    page_h = max(1.0, float(page_rect.height))

                                    width_ratio = bbox_w / page_w
                                    height_ratio = bbox_h / page_h

                                    # Reject full-page screenshots / full-page background captures
                                    if width_ratio >= 0.95 and height_ratio >= 0.95:
                                        print(
                                            f"[MEDIA SKIP] Page {page_num}, image {img_idx + 1}: "
                                            f"full-page image/screenshot ({width_ratio:.1%} x {height_ratio:.1%})"
                                        )
                                        continue

                                # 2. Extract image bytes
                                base_image = doc.extract_image(xref)
                                if not base_image or "image" not in base_image:
                                    continue

                                image_bytes = base_image["image"]
                                if len(image_bytes) < 1000:
                                    continue

                                # 3. Open image safely with PIL
                                try:
                                    im = Image.open(io.BytesIO(image_bytes))
                                    im.load()
                                except Exception:
                                    continue

                                # 4. Dimension filters (reject tiny icons, tracking pixels)
                                if im.width < 80 or im.height < 60:
                                    continue

                                # 5. Reject extreme thin strips / divider lines
                                smallest_side = max(1, min(im.width, im.height))
                                largest_side = max(im.width, im.height)
                                if (largest_side / smallest_side) > 10:
                                    continue

                                # 6. Mode normalization
                                if im.mode not in ("RGB", "L"):
                                    rgb_im = im.convert("RGB")
                                else:
                                    rgb_im = im.convert("RGB")

                                # 7. Reject solid / monochromatic decorative boxes & blank images
                                try:
                                    stat = ImageStat.Stat(rgb_im)
                                    max_std = max(stat.stddev) if stat.stddev else 0.0
                                    # Reject flat / solid color block fills (e.g. decorative peach/orange boxes)
                                    if max_std < 5.0:
                                        continue

                                    colors = rgb_im.getcolors(maxcolors=4)
                                    if colors is not None and len(colors) <= 2 and max_std < 10.0:
                                        continue

                                    # Reject pure white / pure dark images
                                    mean_val = sum(stat.mean) / len(stat.mean) if stat.mean else 128.0
                                    if mean_val >= 253.0 or mean_val <= 3.0:
                                        continue
                                except Exception:
                                    pass

                                # 8. Deduplication
                                image_hash = hashlib.md5(image_bytes).hexdigest()
                                if image_hash in seen_hashes:
                                    continue
                                seen_hashes.add(image_hash)

                                # 9. Save as clean PNG
                                img_name = f"page_{page_num}_img_{len(page_images) + 1}.png"
                                img_path = os.path.join(target_dir, img_name)
                                rgb_im.save(img_path, format="PNG")

                                # 10. Record metadata
                                rel_url = f"/static/extracted_images/{doc_folder}/{img_name}"
                                page_images.append({
                                    "url": rel_url,
                                    "filename": img_name,
                                    "source": os.path.basename(filepath),
                                    "page": page_num,
                                    "unit_label": "Page",
                                    "width": rgb_im.width,
                                    "height": rgb_im.height,
                                    "caption": f"Diagram from {os.path.basename(filepath)} (Page {page_num})",
                                })

                            except Exception as image_error:
                                print(f"[MEDIA IMAGE SKIP] Page {page_num}, image {img_idx + 1}: {image_error}")
                                continue

                        if page_images:
                            media_by_page[page_num] = page_images

                    doc.close()
                    total_extracted = sum(len(v) for v in media_by_page.values())
                    print(f"[MEDIA] Extracted {total_extracted} diagrams from {os.path.basename(filepath)}")
                    return media_by_page

                except Exception as e:
                    print(f"[MEDIA PYMUPDF WARNING] Failed PyMuPDF media extraction on {filepath}: {e}")

            # PyPDF fallback
            if pypdf is not None:
                try:
                    reader = pypdf.PdfReader(filepath)
                    for page_idx, page in enumerate(reader.pages):
                        page_num = page_idx + 1
                        page_images = []
                        try:
                            for img_idx, img_obj in enumerate(page.images):
                                try:
                                    img_data = img_obj.data
                                    if len(img_data) < 1000:
                                        continue
                                    im = Image.open(io.BytesIO(img_data))
                                    im.load()
                                    if im.width < 80 or im.height < 60:
                                        continue
                                    rgb_im = im.convert("RGB")
                                    stat = ImageStat.Stat(rgb_im)
                                    max_std = max(stat.stddev) if stat.stddev else 0.0
                                    if max_std < 5.0:
                                        continue
                                    colors = rgb_im.getcolors(maxcolors=4)
                                    if colors is not None and len(colors) <= 2 and max_std < 10.0:
                                        continue

                                    img_name = f"page_{page_num}_img_{len(page_images) + 1}.png"
                                    img_path = os.path.join(target_dir, img_name)
                                    rgb_im.save(img_path, format="PNG")
                                    rel_url = f"/static/extracted_images/{doc_folder}/{img_name}"
                                    page_images.append({
                                        "url": rel_url,
                                        "filename": img_name,
                                        "source": os.path.basename(filepath),
                                        "page": page_num,
                                        "unit_label": "Page",
                                        "width": rgb_im.width,
                                        "height": rgb_im.height,
                                        "caption": f"Diagram from {os.path.basename(filepath)} (Page {page_num})",
                                    })
                                except Exception:
                                    continue
                        except Exception:
                            pass

                        if page_images:
                            media_by_page[page_num] = page_images

                    return media_by_page
                except Exception as e:
                    print(f"[MEDIA PYPDF ERROR] Failed fallback extraction: {e}")

        # =========================================================
        # POWERPOINT (.pptx)
        # =========================================================
        elif ext == ".pptx":
            try:
                import pptx
                prs = pptx.Presentation(filepath)
                seen_hashes = set()

                for slide_idx, slide in enumerate(prs.slides):
                    slide_num = slide_idx + 1
                    slide_images = []
                    for sh_idx, shape in enumerate(slide.shapes):
                        img_blob = None
                        try:
                            if hasattr(shape, "image") and shape.image:
                                img_blob = shape.image.blob
                        except Exception:
                            pass

                        if img_blob:
                            try:
                                if len(img_blob) < 300:
                                    continue
                                im = Image.open(io.BytesIO(img_blob))
                                im.load()
                                if im.width < 40 or im.height < 40:
                                    continue

                                image_hash = hashlib.md5(img_blob).hexdigest()
                                if image_hash in seen_hashes:
                                    continue
                                seen_hashes.add(image_hash)

                                rgb_im = im.convert("RGB") if im.mode not in ("RGB", "RGBA") else im
                                img_name = f"slide_{slide_num}_img_{len(slide_images) + 1}.png"
                                img_path = os.path.join(target_dir, img_name)
                                rgb_im.save(img_path, format="PNG")

                                rel_url = f"/static/extracted_images/{doc_folder}/{img_name}"
                                slide_images.append({
                                    "url": rel_url,
                                    "filename": img_name,
                                    "source": os.path.basename(filepath),
                                    "page": slide_num,
                                    "unit_label": "Slide",
                                    "width": im.width,
                                    "height": im.height,
                                    "caption": f"Figure from {os.path.basename(filepath)} (Slide {slide_num})",
                                })
                            except Exception:
                                continue
                    if slide_images:
                        media_by_page[slide_num] = slide_images
            except Exception as e:
                print(f"[MEDIA ERROR] Failed PPTX media extraction on {filepath}: {e}")

        # =========================================================
        # WORD (.docx)
        # =========================================================
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(filepath)
                doc_images = []
                seen_hashes = set()

                for rel_idx, rel in enumerate(doc.part.rels.values()):
                    reltype = getattr(rel, "reltype", "").lower()
                    target_ref = getattr(rel, "target_ref", "").lower()
                    if "image" not in reltype and "image" not in target_ref and "media" not in target_ref:
                        continue

                    try:
                        img_part = rel.target_part
                        img_bytes = img_part.blob
                        if len(img_bytes) < 300:
                            continue
                        im = Image.open(io.BytesIO(img_bytes))
                        im.load()
                        if im.width < 40 or im.height < 40:
                            continue

                        image_hash = hashlib.md5(img_bytes).hexdigest()
                        if image_hash in seen_hashes:
                            continue
                        seen_hashes.add(image_hash)

                        rgb_im = im.convert("RGB") if im.mode not in ("RGB", "RGBA") else im
                        img_name = f"doc_img_{len(doc_images) + 1}.png"
                        img_path = os.path.join(target_dir, img_name)
                        rgb_im.save(img_path, format="PNG")

                        rel_url = f"/static/extracted_images/{doc_folder}/{img_name}"
                        doc_images.append({
                            "url": rel_url,
                            "filename": img_name,
                            "source": os.path.basename(filepath),
                            "page": 1,
                            "unit_label": "Page",
                            "width": im.width,
                            "height": im.height,
                            "caption": f"Document Figure from {os.path.basename(filepath)}",
                        })
                    except Exception:
                        continue

                if doc_images:
                    media_by_page[1] = doc_images
            except Exception as e:
                print(f"[MEDIA ERROR] Failed DOCX media extraction on {filepath}: {e}")

        return media_by_page


class OCRHelper:
    """Provides automated OCR detection and extraction for scanned document pages."""

    @staticmethod
    def is_scanned_page(text: str, has_images: bool = True) -> bool:
        """Determines if a page is scanned based on minimal extractable character count."""
        cleaned = re.sub(r"\s+", "", text or "")
        return len(cleaned) < 35 and has_images

    @classmethod
    def run_ocr_on_pixmap(cls, pixmap_or_image: Any) -> str:
        """Runs OCR on a PIL Image or PyMuPDF pixmap."""
        try:
            if isinstance(pixmap_or_image, Image.Image):
                pil_img = pixmap_or_image
            elif hasattr(pixmap_or_image, "tobytes"):
                img_bytes = pixmap_or_image.tobytes("png")
                pil_img = Image.open(io.BytesIO(img_bytes))
            else:
                return ""

            if pytesseract is not None:
                try:
                    ocr_text = pytesseract.image_to_string(pil_img)
                    if ocr_text and len(ocr_text.strip()) > 10:
                        return ocr_text.strip()
                except Exception:
                    pass

        except Exception as e:
            print(f"[OCR WARNING] OCR extraction error: {e}")

        return ""


class PDFLoader:
    """High-speed, memory-efficient PDF extractor using PyMuPDF (fitz) with OCR and Table support."""

    @staticmethod
    def load(
        filepath: str,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> List[Tuple[int, str]]:
        """Extracts text page by page, streaming lazily to support multi-hundred page PDF books.
        Returns a list of tuples: (page_num_1_indexed, raw_text).
        """
        pages_data: List[Tuple[int, str]] = []

        if pymupdf is not None:
            try:
                doc = pymupdf.open(filepath)
                total_pages = len(doc)

                for page_idx in range(total_pages):
                    page_num = page_idx + 1
                    page = doc[page_idx]
                    page_blocks: List[str] = []

                    # A. Native text extraction
                    raw_text = page.get_text("text") or ""

                    # B. Extract structured tables via PyMuPDF find_tables()
                    try:
                        tables = page.find_tables()
                        if tables and hasattr(tables, "tables"):
                            for t_idx, tab in enumerate(tables.tables, 1):
                                tab_rows = tab.extract()
                                if tab_rows and len(tab_rows) > 0:
                                    headers = tab_rows[0]
                                    data_rows = tab_rows[1:] if len(tab_rows) > 1 else []
                                    md_tbl = format_markdown_table(headers, data_rows)
                                    if md_tbl.strip():
                                        page_blocks.append(f"\n[Table {t_idx}]\n{md_tbl}\n")
                    except Exception:
                        pass

                    # C. Check if page is scanned (< 35 chars) and attempt OCR fallback
                    has_images = len(page.get_images(full=False)) > 0 or len(page.get_drawings()) > 0
                    if OCRHelper.is_scanned_page(raw_text, has_images=has_images):
                        try:
                            pix = page.get_pixmap(dpi=200)
                            ocr_result = OCRHelper.run_ocr_on_pixmap(pix)
                            if ocr_result:
                                raw_text = ocr_result
                            del pix
                        except Exception as ocr_err:
                            print(f"[OCR PAGE {page_num} WARNING] {ocr_err}")

                    if raw_text.strip():
                        page_blocks.insert(0, raw_text.strip())

                    full_page_text = "\n\n".join(page_blocks).strip()
                    pages_data.append((page_num, full_page_text))

                    if page_num % 25 == 0:
                        gc.collect()

                    if progress_callback:
                        progress_callback("extracting", page_num, total_pages, f"Extracted page {page_num}/{total_pages}")

                doc.close()
                return pages_data

            except Exception as e:
                print(f"[PYMUPDF ERROR] Failed PyMuPDF extraction on '{filepath}': {e}. Trying pypdf fallback...")

        if pypdf is not None:
            try:
                reader = pypdf.PdfReader(filepath)
                total_pages = len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    page_num = page_idx + 1
                    raw_text = page.extract_text() or ""
                    pages_data.append((page_num, raw_text))
                    if progress_callback:
                        progress_callback("extracting", page_num, total_pages, f"Extracted page {page_num}/{total_pages}")
                return pages_data
            except Exception as e:
                print(f"[PYPDF ERROR] Failed pypdf extraction on '{filepath}': {e}")
                raise

        return pages_data


class DocxLoader:
    """Extracts text, tables, and document structure from Word (.docx) files using python-docx."""

    @staticmethod
    def load(
        filepath: str,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> List[Tuple[int, str]]:
        """Extracts text from paragraphs and tables, formatting structured tables in Markdown.
        Returns a list of (section_or_page_1_indexed, raw_text).
        """
        try:
            import docx
            doc = docx.Document(filepath)
            content_blocks = []

            for p in doc.paragraphs:
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name if p.style else ""
                style_lower = style_name.lower()

                if style_name.startswith("Heading") or "title" in style_lower or "subtitle" in style_lower:
                    content_blocks.append(f"\n# {text}\n")
                elif "bullet" in style_lower or "list" in style_lower:
                    if not text.startswith(("- ", "• ", "* ")):
                        content_blocks.append(f"• {text}")
                    else:
                        content_blocks.append(text)
                else:
                    content_blocks.append(text)

            if doc.tables:
                for t_idx, table in enumerate(doc.tables, 1):
                    table_rows = []
                    for row in table.rows:
                        row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                        if any(row_cells):
                            table_rows.append(row_cells)
                    if table_rows:
                        headers = table_rows[0]
                        data_rows = table_rows[1:] if len(table_rows) > 1 else []
                        md_table = format_markdown_table(headers, data_rows)
                        content_blocks.append(f"\n[Table {t_idx}]\n{md_table}\n")

            full_text = "\n\n".join(content_blocks).strip()
            if progress_callback:
                progress_callback("extracting", 1, 1, "Extracted Word document content")
            return [(1, full_text)] if full_text else [(1, "")]

        except Exception as e:
            print(f"[ERROR] Failed to load Word document '{filepath}': {e}")
            raise


class PptxLoader:
    """Extracts text slide by slide from PowerPoint (.pptx) presentations using python-pptx."""

    @staticmethod
    def load(
        filepath: str,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> List[Tuple[int, str]]:
        """Extracts text from each slide (titles, body text, shapes, and tables),
        treating each slide as a distinct page-equivalent (1-indexed) for citations.
        Returns a list of (slide_number_1_indexed, slide_text).
        """
        slides_data = []
        try:
            import pptx
            prs = pptx.Presentation(filepath)
            total_slides = len(prs.slides)

            for slide_idx, slide in enumerate(prs.slides):
                slide_num = slide_idx + 1
                slide_blocks = []

                if slide.shapes.title and slide.shapes.title.has_text_frame:
                    title_text = slide.shapes.title.text_frame.text.strip()
                    if title_text:
                        slide_blocks.append(f"# {title_text}")

                for shape in slide.shapes:
                    if shape == slide.shapes.title:
                        continue

                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            p_text = paragraph.text.strip()
                            if not p_text:
                                continue
                            if paragraph.level > 0 and not p_text.startswith(("- ", "• ", "* ")):
                                slide_blocks.append(f"• {p_text}")
                            else:
                                slide_blocks.append(p_text)

                    elif shape.has_table:
                        table_rows = []
                        for row in shape.table.rows:
                            row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                            if any(row_cells):
                                table_rows.append(row_cells)
                        if table_rows:
                            headers = table_rows[0]
                            data_rows = table_rows[1:] if len(table_rows) > 1 else []
                            md_table = format_markdown_table(headers, data_rows)
                            slide_blocks.append(md_table)

                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes_text = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        slide_blocks.append(f"[Speaker Notes]: {notes_text}")

                slide_text = "\n\n".join(slide_blocks).strip()
                slides_data.append((slide_num, slide_text))

                if progress_callback:
                    progress_callback("extracting", slide_num, total_slides, f"Extracted slide {slide_num}/{total_slides}")

        except Exception as e:
            print(f"[ERROR] Failed to load PowerPoint presentation '{filepath}': {e}")
            raise

        return slides_data


class TextLoader:
    """Loads plain text or markdown files."""

    @staticmethod
    def load(
        filepath: str,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> List[Tuple[int, str]]:
        """Loads text file content, returning [(1, full_text)]."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if progress_callback:
                progress_callback("extracting", 1, 1, "Extracted text file")
            return [(1, content)]
        except Exception as e:
            print(f"[ERROR] Failed to load text file '{filepath}': {e}")
            raise


class DocumentLoader:
    """Unified document loader supporting PDF, Word (DOCX), PowerPoint (PPTX), TXT, and MD files."""

    @classmethod
    def load(
        cls,
        filepath: str,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> List[Tuple[int, str]]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return PDFLoader.load(filepath, progress_callback=progress_callback)
        elif ext == ".docx":
            return DocxLoader.load(filepath, progress_callback=progress_callback)
        elif ext == ".pptx":
            return PptxLoader.load(filepath, progress_callback=progress_callback)
        elif ext in [".txt", ".md"]:
            return TextLoader.load(filepath, progress_callback=progress_callback)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Supported formats are .pdf, .docx, .pptx, .txt, .md"
            )


class TokenChunker:
    """Chunks cleaned document text into segments bounded between 350-750 tokens
    with ~80-100 token overlap, using tiktoken tokenization boundaries.
    """

    def __init__(
        self,
        min_tokens: int = 350,
        max_tokens: int = 700,
        target_tokens: int = 450,
        overlap_tokens: int = 80,
        tokenizer_name: str = "cl100k_base",
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_name = tokenizer_name

        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Returns the exact number of tokens for given text."""
        return len(self.tokenizer.encode(text))

    def split_page_into_blocks(self, text: str, page_num: int) -> List[Tuple[str, int]]:
        """Splits page text into semantic blocks (paragraphs, tables, sentences) tagged with page number."""
        paragraphs = text.split("\n\n")
        blocks = []
        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue

            # Don't split markdown tables into sentences
            if "|" in p_strip and "---" in p_strip:
                blocks.append((p_strip, page_num))
                continue

            # If paragraph itself is larger than target_tokens, split into sentences
            if self.count_tokens(p_strip) > self.target_tokens:
                sentences = re.split(r"(?<=[.?!])\s+", p_strip)
                for s in sentences:
                    s_strip = s.strip()
                    if s_strip:
                        blocks.append((s_strip, page_num))
            else:
                blocks.append((p_strip, page_num))
        return blocks

    def chunk_document_pages(
        self,
        pages_cleaned: List[Tuple[int, str]],
        source: str,
        media_by_page: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[DocumentChunk]:
        """Chunks multi-page document into segments with token overlap, preserving
        page/slide origin, pictures, tables, and formula metadata.
        """
        ext = os.path.splitext(source)[1].lower().lstrip(".")
        doc_type = ext if ext in {"pdf", "docx", "pptx", "txt", "md"} else "pdf"
        unit_label = "Slide" if doc_type == "pptx" else "Page"
        media_by_page = media_by_page or {}

        all_blocks: List[Tuple[str, int]] = []
        for page_num, cleaned_text in pages_cleaned:
            if cleaned_text.strip():
                blocks = self.split_page_into_blocks(cleaned_text, page_num)
                all_blocks.extend(blocks)

        if not all_blocks:
            return []

        chunks: List[DocumentChunk] = []
        current_blocks: List[Tuple[str, int]] = []
        current_tokens = 0
        chunk_idx = 0

        def build_chunk(blocks: List[Tuple[str, int]], c_idx: int) -> DocumentChunk:
            c_text = "\n\n".join(b[0] for b in blocks)
            p_page = blocks[0][1]
            e_page = blocks[-1][1]
            t_count = self.count_tokens(c_text)

            chunk_images = []
            seen_img_urls = set()
            # Attach verified images across all pages covered by this chunk
            for p in range(p_page, e_page + 1):
                for img in media_by_page.get(p, []):
                    if not isinstance(img, dict):
                        continue
                    img_url = img.get("url")
                    if not img_url:
                        continue
                    # Ensure the image strictly belongs to this document
                    if img.get("source") != source:
                        continue
                    if img_url not in seen_img_urls:
                        seen_img_urls.add(img_url)
                        chunk_images.append(img)

            has_tbl = "|" in c_text and "---" in c_text
            has_calc = bool(
                re.search(
                    r"(=|\+|\-|\*|/|%|₹|\$|\bformula\b|\bcalculate\b|\bcalc\b|\bratio\b|\bmargin\b|\bcost\b|\bdepreciation\b)",
                    c_text,
                    re.IGNORECASE,
                )
            )

            meta = {
                "source": source,
                "page": p_page,
                "end_page": e_page,
                "chunk_index": c_idx,
                "token_count": t_count,
                "char_count": len(c_text),
                "doc_type": doc_type,
                "unit_label": unit_label,
                "images": chunk_images,
                "has_images": len(chunk_images) > 0,
                "has_tables": has_tbl,
                "has_calculations": has_calc,
                "chunk_id": f"{source}:p{p_page}:c{c_idx}",
            }
            return DocumentChunk(text=c_text, metadata=meta)

        i = 0
        while i < len(all_blocks):
            block_text, page_num = all_blocks[i]
            block_tokens = self.count_tokens(block_text)

            if current_blocks and (current_tokens + block_tokens > self.max_tokens):
                chunk = build_chunk(current_blocks, chunk_idx)
                chunks.append(chunk)
                chunk_idx += 1

                overlap_acc = []
                overlap_tokens_count = 0
                for b_item in reversed(current_blocks):
                    b_tok = self.count_tokens(b_item[0])
                    if overlap_tokens_count + b_tok <= self.overlap_tokens or not overlap_acc:
                        overlap_acc.insert(0, b_item)
                        overlap_tokens_count += b_tok
                    else:
                        break

                current_blocks = list(overlap_acc)
                current_tokens = overlap_tokens_count

            current_blocks.append((block_text, page_num))
            current_tokens += block_tokens

            if current_tokens >= self.target_tokens and (i + 1 < len(all_blocks)):
                next_block_tokens = self.count_tokens(all_blocks[i + 1][0])
                if current_tokens + next_block_tokens > self.target_tokens + (self.max_tokens - self.target_tokens) // 2:
                    chunk = build_chunk(current_blocks, chunk_idx)
                    chunks.append(chunk)
                    chunk_idx += 1

                    overlap_acc = []
                    overlap_tokens_count = 0
                    for b_item in reversed(current_blocks):
                        b_tok = self.count_tokens(b_item[0])
                        if overlap_tokens_count + b_tok <= self.overlap_tokens or not overlap_acc:
                            overlap_acc.insert(0, b_item)
                            overlap_tokens_count += b_tok
                        else:
                            break
                    current_blocks = list(overlap_acc)
                    current_tokens = overlap_tokens_count

            i += 1

        if current_blocks:
            chunks.append(build_chunk(current_blocks, chunk_idx))

        return chunks


def ingest_documents(
    docs_dir: str = DEFAULT_DOCUMENTS_DIR,
    output_path: Optional[str] = DEFAULT_CHUNKS_PATH,
    min_tokens: int = 350,
    max_tokens: int = 700,
    target_tokens: int = 450,
    overlap_tokens: int = 80,
    extract_media: bool = True,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> List[DocumentChunk]:
    """Orchestrates loading, cleaning, media extraction, chunking, and metadata tagging across all documents."""
    docs_dir = os.path.abspath(docs_dir)
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Documents directory '{docs_dir}' not found.")

    cleaner = TextCleaner()
    chunker = TokenChunker(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )

    all_chunks: List[DocumentChunk] = []
    supported_extensions = {".pdf", ".docx", ".pptx", ".txt", ".md"}

    files = sorted(
        [
            f
            for f in os.listdir(docs_dir)
            if os.path.splitext(f)[1].lower() in supported_extensions
            and not f.startswith("~$")
            and not f.startswith(".")
        ]
    )

    print(f"\n========================================================")
    print(f"Ingesting {len(files)} documents from: {docs_dir}")
    print(f"Extract media: {extract_media} | Target tokens: {target_tokens}")
    print(f"========================================================\n")

    for file_idx, filename in enumerate(files, 1):
        filepath = os.path.join(docs_dir, filename)
        raw_pages = DocumentLoader.load(filepath, progress_callback=progress_callback)

        media_by_page = {}
        if extract_media:
            try:
                media_by_page = MediaExtractor.extract_media(filepath)
            except Exception as e:
                print(f"[MEDIA WARNING] Media extraction failed for {filename}: {e}")

        cleaned_pages = []
        for page_num, raw_text in raw_pages:
            cleaned_text = cleaner.clean(raw_text)
            if cleaned_text:
                cleaned_pages.append((page_num, cleaned_text))

        doc_chunks = chunker.chunk_document_pages(
            pages_cleaned=cleaned_pages,
            source=filename,
            media_by_page=media_by_page,
        )

        all_chunks.extend(doc_chunks)
        total_tokens = sum(c.token_count for c in doc_chunks)
        img_count = sum(len(c.images) for c in doc_chunks)
        print(
            f"[{file_idx:02d}/{len(files):02d}] {filename:<45} | "
            f"Pages: {len(raw_pages):2d} | Chunks: {len(doc_chunks):2d} | Images: {img_count:2d} | Tokens: {total_tokens:5d}"
        )

    if output_path:
        out_abs = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Serialized {len(all_chunks)} chunks to: {out_abs}")

    return all_chunks


def ingest_single_file(
    filepath: str,
    min_tokens: int = 350,
    max_tokens: int = 700,
    target_tokens: int = 450,
    overlap_tokens: int = 80,
    extract_media: bool = True,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> List[DocumentChunk]:
    """Ingests, cleans, extracts media, and chunks a single document file with streaming & progress callbacks."""
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")

    filename = os.path.basename(filepath)
    cleaner = TextCleaner()
    chunker = TokenChunker(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )

    media_by_page = {}
    if extract_media:
        try:
            media_by_page = MediaExtractor.extract_media(filepath)
        except Exception as e:
            print(f"[MEDIA WARNING] Media extraction failed for {filename}: {e}")

    raw_pages = DocumentLoader.load(filepath, progress_callback=progress_callback)
    cleaned_pages = []
    for page_num, raw_text in raw_pages:
        cleaned_text = cleaner.clean(raw_text)
        if cleaned_text:
            cleaned_pages.append((page_num, cleaned_text))

    chunks = chunker.chunk_document_pages(
        pages_cleaned=cleaned_pages,
        source=filename,
        media_by_page=media_by_page,
    )
    return chunks


def save_processed_chunks(
    chunks: List[DocumentChunk],
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> List[DocumentChunk]:
    """Overwrites processed_chunks.json with a new list of chunks (replacing old corpus)."""
    chunks_path = os.path.abspath(chunks_path)
    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
    return chunks


def clear_directory(directory_path: str) -> int:
    """Removes all files in the given directory without deleting the directory itself."""
    dir_abs = os.path.abspath(directory_path)
    if not os.path.exists(dir_abs):
        os.makedirs(dir_abs, exist_ok=True)
        return 0

    removed_count = 0
    for filename in os.listdir(dir_abs):
        file_path = os.path.join(dir_abs, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                removed_count += 1
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
                removed_count += 1
        except Exception as e:
            print(f"[CLEANUP WARNING] Could not remove '{file_path}': {e}")

    return removed_count


def append_chunks_to_processed_file(
    new_chunks: List[DocumentChunk],
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> List[DocumentChunk]:
    """Appends new chunks to the processed_chunks.json file (avoiding duplicates)."""
    chunks_path = os.path.abspath(chunks_path)
    existing_chunks = []
    if os.path.exists(chunks_path):
        existing_chunks = load_processed_chunks(chunks_path)

    new_chunk_ids = set(c.chunk_id for c in new_chunks)
    retained_chunks = [c for c in existing_chunks if c.chunk_id not in new_chunk_ids]
    all_chunks = retained_chunks + new_chunks

    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in all_chunks], f, indent=2, ensure_ascii=False)

    return all_chunks


def load_processed_chunks(file_path: str = DEFAULT_CHUNKS_PATH) -> List[DocumentChunk]:
    """Loads previously saved processed chunks from JSON."""
    file_abs = os.path.abspath(file_path)
    if not os.path.exists(file_abs):
        raise FileNotFoundError(f"Processed chunks file not found at '{file_abs}'")
    with open(file_abs, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [DocumentChunk.from_dict(item) for item in data]


def print_ingestion_summary(chunks: List[DocumentChunk]):
    """Displays detailed metrics and sample inspection of generated chunks."""
    if not chunks:
        print("No chunks to summarize.")
        return

    token_counts = [c.token_count for c in chunks]
    char_counts = [c.metadata.get("char_count", len(c.text)) for c in chunks]
    sources = set(c.source for c in chunks)

    print("\n" + "=" * 65)
    print("                    INGESTION SUMMARY METRICS                    ")
    print("=" * 65)
    print(f"Total Source Documents Ingested : {len(sources)}")
    print(f"Total Chunks Generated          : {len(chunks)}")
    print(f"Average Tokens per Chunk        : {sum(token_counts) / len(chunks):.1f}")
    print(f"Min Tokens in a Chunk           : {min(token_counts)}")
    print(f"Max Tokens in a Chunk           : {max(token_counts)}")
    print(f"Average Characters per Chunk    : {sum(char_counts) / len(chunks):.1f}")
    print("=" * 65)


if __name__ == "__main__":
    chunks = ingest_documents()
    print_ingestion_summary(chunks)
