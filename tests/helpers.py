"""Shared test builders."""

from __future__ import annotations


def make_pdf(pages: list[str]) -> bytes:
    """Builds a minimal valid PDF with one Helvetica text line per page line.

    An empty string makes a page with no text layer (like a scanned page).
    """
    objects: list[bytes] = []
    kids = []
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_id} 0 R")
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
        for line in text.splitlines():
            safe = line.replace("\\", "\\\\").replace("(", "\\(")
            safe = safe.replace(")", "\\)")
            ops.append(f"({safe}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode() if text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream)
            + stream
            + b"\nendstream"
        )
    catalog = b"<< /Type /Catalog /Pages 2 0 R >>"
    pages_obj = (
        f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    ).encode()
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    ordered = [catalog, pages_obj, *objects, font]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(ordered, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(ordered) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(ordered) + 1,
        xref,
    )
    return bytes(out)
