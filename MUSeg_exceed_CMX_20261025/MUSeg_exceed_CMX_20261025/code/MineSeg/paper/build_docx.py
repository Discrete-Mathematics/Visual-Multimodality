"""Build paper.docx from paper.md headings, paragraphs and tables."""

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = Path("/root/autodl-tmp/MineSeg/paper")
lines = (ROOT / "paper.md").read_text(encoding="utf-8").splitlines()
doc = Document()
style = doc.styles["Normal"]
style.font.name = "WenQuanYi Micro Hei"
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "WenQuanYi Micro Hei")


def add_runs(paragraph, text):
    paragraph.add_run(text)


i = 0
while i < len(lines):
    line = lines[i]
    if not line.strip():
        i += 1
        continue
    if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
            i += 1
        table = doc.add_table(rows=len(rows), cols=len(rows[0]))
        table.style = "Table Grid"
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.rows[r].cells[c].text = value
        continue
    if line.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].startswith("```"):
            buf.append(lines[i])
            i += 1
        p = doc.add_paragraph("\n".join(buf))
        for run in p.runs:
            run.font.name = "Courier New"
            run.font.size = Pt(9)
        i += 1
        continue
    if line.startswith("# "):
        doc.add_heading(line[2:].strip(), level=1)
    elif line.startswith("## "):
        doc.add_heading(line[3:].strip(), level=2)
    elif line.startswith("### "):
        doc.add_heading(line[4:].strip(), level=3)
    else:
        text = line.strip()
        text = text.replace("**", "")
        text = text.replace("`", "")
        doc.add_paragraph(text)
    i += 1

out = ROOT / "paper.docx"
doc.save(out)
print(out, out.stat().st_size)
