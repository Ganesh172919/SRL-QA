from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from build_presentation_assets import generate_assets
from presentation.builder import (
    build_markdown_sections,
    build_presentation_context,
    combine_master_lines,
    ensure_minimum_line_count,
)


INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="DeckTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#12324a"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DeckHeading2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0f766e"),
            spaceAfter=8,
            spaceBefore=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DeckHeading3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=6,
            spaceBefore=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DeckHeading4",
            parent=styles["Heading4"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#7a4b17"),
            spaceAfter=4,
            spaceBefore=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DeckBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=12.2,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DeckBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=11.9,
            leftIndent=12,
            firstLineIndent=-8,
            bulletIndent=0,
            spaceAfter=3,
        )
    )
    return styles


def _format_inline(text: str) -> str:
    escaped = escape(text)
    return INLINE_CODE_PATTERN.sub(lambda match: f"<font face='Courier'>{escape(match.group(1))}</font>", escaped)


def _build_table(table_lines: list[str], styles) -> Table:
    rows = []
    for raw_line in table_lines:
        stripped = raw_line.strip().strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]
        if set(cells[0]) == {"-"}:
            continue
        rows.append([Paragraph(_format_inline(cell), styles["DeckBody"]) for cell in cells])
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12324a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _image_flowable(match: re.Match[str], base_dir: Path) -> list:
    image_path = (base_dir / match.group("path")).resolve()
    if not image_path.exists():
        return [Paragraph(f"Missing image: {match.group('path')}", _styles()["DeckBody"])]
    image = Image(str(image_path))
    image.drawHeight = min(image.drawHeight, 4.8 * inch)
    image.drawWidth = min(image.drawWidth, 6.8 * inch)
    return [image, Spacer(1, 0.08 * inch)]


def _markdown_to_story(lines: list[str], base_dir: Path) -> list:
    styles = _styles()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            story.append(Spacer(1, 0.08 * inch))
            index += 1
            continue
        image_match = IMAGE_PATTERN.match(line)
        if image_match:
            story.extend(_image_flowable(image_match, base_dir))
            index += 1
            continue
        if line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].startswith("|"):
                block.append(lines[index])
                index += 1
            story.append(_build_table(block, styles))
            story.append(Spacer(1, 0.12 * inch))
            continue
        if line.startswith("# "):
            story.append(Paragraph(_format_inline(line[2:]), styles["DeckTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(_format_inline(line[3:]), styles["DeckHeading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(_format_inline(line[4:]), styles["DeckHeading3"]))
        elif line.startswith("#### "):
            story.append(Paragraph(_format_inline(line[5:]), styles["DeckHeading4"]))
        elif line.startswith("- "):
            story.append(Paragraph(_format_inline(line[2:]), styles["DeckBullet"], bulletText="•"))
        elif re.match(r"^\d+\.\s", line):
            story.append(Paragraph(_format_inline(line), styles["DeckBody"]))
        else:
            story.append(Paragraph(_format_inline(line), styles["DeckBody"]))
        index += 1
    return story


def _write_pdf(path: Path, lines: list[str], base_dir: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=title,
    )

    def add_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#4b5563"))
        canvas.drawRightString(A4[0] - 0.55 * inch, 0.35 * inch, f"Page {doc_obj.page}")
        canvas.restoreState()

    story = _markdown_to_story(lines, base_dir)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def build_docs() -> dict[str, str]:
    generate_assets()
    ctx = build_presentation_context()
    master_sections = build_markdown_sections(ctx, asset_prefix="assets")
    section_sections = build_markdown_sections(ctx, asset_prefix="../assets")

    ordered_section_ids = [section["id"] for section in ctx.manifest["sections"]]
    master_lines = combine_master_lines(OrderedDict((section_id, master_sections[section_id]) for section_id in ordered_section_ids))
    master_lines = ensure_minimum_line_count(master_lines, ctx, minimum=2000)
    if next((line for line in master_lines if line.strip()), "") != "# Survey":
        raise RuntimeError("Master markdown must start with the Survey section.")

    master_markdown_path = ctx.outputs["master_markdown"]
    master_markdown_path.write_text("\n".join(master_lines) + "\n", encoding="utf-8")

    if len(master_lines) < 2000:
        raise RuntimeError("Master markdown did not reach 2000 lines.")

    for section in ctx.manifest["sections"]:
        section_path = ctx.presentation_dir / section["section_markdown"]
        section_lines = section_sections[section["id"]]
        section_path.write_text("\n".join(section_lines) + "\n", encoding="utf-8")
        section_pdf_path = ctx.presentation_dir / section["section_pdf"]
        _write_pdf(section_pdf_path, section_lines, section_path.parent, section["title"])

    _write_pdf(ctx.outputs["master_pdf"], master_lines, ctx.presentation_dir, ctx.manifest["project"]["title"])

    summary = {
        "master_markdown": str(master_markdown_path.relative_to(ctx.root)),
        "master_pdf": str(ctx.outputs["master_pdf"].relative_to(ctx.root)),
        "line_count": len(master_lines),
        "section_pdfs": {
            section["id"]: str((ctx.presentation_dir / section["section_pdf"]).relative_to(ctx.root))
            for section in ctx.manifest["sections"]
        },
    }
    (ctx.outputs["pdfs_dir"] / "pdf_index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = build_docs()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
