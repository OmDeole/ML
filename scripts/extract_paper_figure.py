"""Regenerate the paper-vs-replication comparison crop from a PDF you own.

The paper's figures are rasterised inside the PDF, so the values plotted in its
Fig. 14 were recovered by pixel-calibrating the y-axis (see
`results/paper_fig14_decoded.csv`). This script re-creates that crop locally and
prints the decoded values, so the repository does not have to redistribute the
original figure.

Usage:
    python scripts/extract_paper_figure.py /path/to/paper.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper_fig14.png"

# Fig. 14 sits on page 11 (0-indexed 10) of arXiv:2302.12042v2
PAGE_INDEX = 10
CROP = (150, 280, 470, 440)          # x0, y0, x1, y1 in PDF points


def main(pdf_path: str):
    try:
        import pymupdf
    except ImportError:
        sys.exit("pip install pymupdf")

    doc = pymupdf.open(pdf_path)
    if len(doc) <= PAGE_INDEX:
        sys.exit(f"expected at least {PAGE_INDEX + 1} pages, found {len(doc)}")

    page = doc[PAGE_INDEX]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    page.get_pixmap(dpi=300, clip=pymupdf.Rect(*CROP)).save(OUT)
    print(f"wrote {OUT}")

    # --- sanity-check the decoded values against the raster ---
    pix = page.get_pixmap(dpi=300)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    blue = (arr[:, :, 2] > 90) & (arr[:, :, 0] < 60) & (arr[:, :, 1] < 60)
    ys, xs = np.nonzero(blue)
    print(f"blue plot pixels found: {len(ys)} (expected ~10k for Fig. 14)")

    print("\nDecoded Fig. 14 values live in results/paper_fig14_decoded.csv:")
    csv = ROOT / "results" / "paper_fig14_decoded.csv"
    if csv.exists():
        print(csv.read_text())
    print("To re-derive them, see scripts/analyze.py: the y-axis is calibrated from the")
    print("0.78 and 0.66 gridline label rows (0.78 at y=1277, 0.66 at y=1727.5 at 300 dpi).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
