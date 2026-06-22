"""dots.ocr parser: rasterize PDF/image pages, call the dots.ocr vLLM server
(OpenAI-compatible), and map its layout JSON to RAGAnything content_list blocks."""
import os
import re
import json
import base64

import httpx
import fitz  # PyMuPDF

DOTS_OCR_URL = os.getenv("DOTS_OCR_URL", "http://localhost:8000/v1")
DOTS_OCR_MODEL = os.getenv("DOTS_OCR_MODEL", "model")
DOTS_DPI = int(os.getenv("DOTS_DPI", "200"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")

PROMPT = os.getenv("DOTS_PROMPT") or (
    "Please output the layout information from the image, including each element's "
    "bbox, its category, and the text content within the bbox. Use [x1, y1, x2, y2] "
    "for bbox, LaTeX for formulas, HTML for tables. Return a single JSON array of "
    "objects with keys: bbox, category, text."
)

_HEADER = {"Page-header": "header", "Page-footer": "footer"}


def _zoom():
    return DOTS_DPI / 72.0


def _extract_json(content):
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("layout") or data.get("elements") or []
    return data if isinstance(data, list) else []


async def _parse_page(client, png_bytes):
    b64 = base64.b64encode(png_bytes).decode()
    payload = {
        "model": DOTS_OCR_MODEL,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    r = await client.post(f"{DOTS_OCR_URL}/chat/completions", json=payload)
    r.raise_for_status()
    return _extract_json(r.json()["choices"][0]["message"]["content"])


def _to_block(el, page_idx, page, idx):
    cat = el.get("category") or ""
    text = (el.get("text") or "").strip()
    if cat == "Table":
        return {"type": "table", "table_body": text, "page_idx": page_idx}
    if cat == "Formula":
        return {"type": "equation", "text": text, "text_format": "latex", "page_idx": page_idx}
    if cat == "Picture":
        bbox = el.get("bbox")
        if not bbox:
            return None
        z = _zoom()
        rect = fitz.Rect(bbox[0] / z, bbox[1] / z, bbox[2] / z, bbox[3] / z)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img_path = os.path.join(OUTPUT_DIR, f"page{page_idx}_img{idx}.png")
        page.get_pixmap(matrix=fitz.Matrix(z, z), clip=rect).save(img_path)
        return {"type": "image", "img_path": img_path, "page_idx": page_idx}
    if cat in _HEADER:
        return {"type": _HEADER[cat], "text": text, "page_idx": page_idx}
    return {"type": "text", "text": text, "page_idx": page_idx}


async def parse_document(path, start=None, end=None):
    """Parse a PDF/image into a RAGAnything content_list (start/end are 1-indexed inclusive, PDFs)."""
    doc = fitz.open(path)
    mat = fitz.Matrix(_zoom(), _zoom())
    first = (start - 1) if start else 0
    last = (end - 1) if end else (doc.page_count - 1)
    content_list = []
    async with httpx.AsyncClient(timeout=600) as client:
        for pno in range(first, last + 1):
            page = doc[pno]
            png = page.get_pixmap(matrix=mat).tobytes("png")
            for i, el in enumerate(await _parse_page(client, png)):
                block = _to_block(el, pno, page, i)
                if block and (block.get("text") or block.get("table_body") or block.get("img_path")):
                    content_list.append(block)
    doc.close()
    return content_list
