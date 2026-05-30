"""Markdown → PDF 导出 (CJK 字体支持)"""

import os
import re
import unicodedata
from fpdf import FPDF


def _font_path():
    for p in [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc"),
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]:
        if os.path.exists(p):
            return p
    return None

_FONT = _font_path()


def _md_to_plain(md):
    lines = md.split("\n")
    result = []
    in_table = False

    for line in lines:
        s = line.strip()

        if s in ("---", "***"):
            result.append(("sep", ""))
            in_table = False
            continue

        if s.startswith("# "):
            result.append(("h1", s[2:]))
            in_table = False
            continue
        if s.startswith("## "):
            result.append(("h2", s[2:]))
            in_table = False
            continue
        if s.startswith("### "):
            result.append(("h3", s[3:]))
            in_table = False
            continue

        if re.fullmatch(r"[\|\s\-:]+", s):
            continue

        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            if not in_table:
                col_count = len(cells)
                result.append(("row_h", str(col_count)))
                for c in cells:
                    result.append(("cell_h", c))
                result.append(("row_end", ""))
                in_table = True
            else:
                result.append(("row_d", str(col_count)))
                for c in cells:
                    result.append(("cell_d", c))
                result.append(("row_end", ""))
            continue
        else:
            in_table = False

        m = re.match(r"^(\d+)\.\s+(.+)", s)
        if m:
            result.append(("item_n", m.group(1), m.group(2)))
            continue

        if re.match(r"^[\-\*]\s+", s):
            result.append(("item", s[2:]))
            continue

        if not s:
            result.append(("blank", ""))
            continue

        result.append(("text", s))

    return result


def _filter_emoji(s):
    result = []
    for c in s:
        cat = unicodedata.category(c)
        cp = ord(c)
        if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
            continue
        if cat.startswith("S") and cp > 0x7E:
            continue
        if cat == "So" and cp > 0x7E:
            continue
        result.append(c)
    return "".join(result)


def _clean(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", s)
    s = re.sub(r"!\[.*?\]\(.+?\)", "", s)
    return s


def _cjk_width(c):
    w = unicodedata.east_asian_width(c)
    return 2 if w in ("W", "F") else 1


def _wrap(text, max_w):
    lines = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur_line, cur_w = [], 0
        for ch in para:
            cw = _cjk_width(ch)
            if cur_w + cw > max_w and cur_line:
                lines.append("".join(cur_line))
                cur_line, cur_w = [], 0
            cur_line.append(ch)
            cur_w += cw
        if cur_line:
            lines.append("".join(cur_line))
    return lines


def _render_table_row(pdf, cells, col_w, font, page_h, b_margin):
    TABLE_FONT_SIZE = 9
    LINE_HT = 5
    PAD_X = 1.2
    PAD_Y = 1.5

    cell_max_w = int(col_w / 0.78)

    wrapped = []
    max_lines = 1
    for elem in cells:
        text = _filter_emoji(_clean(elem[1]))
        lines = _wrap(text, cell_max_w)
        if not lines:
            lines = [""]
        wrapped.append(lines)
        max_lines = max(max_lines, len(lines))

    row_h = max_lines * LINE_HT + PAD_Y * 2

    if pdf.get_y() + row_h > page_h - b_margin:
        pdf.add_page()

    y_start = pdf.get_y()
    x_start = pdf.l_margin

    for i, (elem, lines) in enumerate(zip(cells, wrapped)):
        x_pos = x_start + i * col_w
        is_header = (elem[0] == "cell_h")

        if is_header:
            pdf.set_fill_color(13, 110, 143)
            pdf.set_draw_color(13, 110, 143)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_draw_color(200, 200, 200)
        pdf.rect(x_pos, y_start, col_w, row_h, style="DF")

        if is_header:
            pdf.set_font(font, "B", TABLE_FONT_SIZE)
            pdf.set_text_color(255, 255, 255)
        else:
            pdf.set_font(font, "", TABLE_FONT_SIZE)
            pdf.set_text_color(0, 0, 0)

        for j, line in enumerate(lines):
            pdf.set_xy(x_pos + PAD_X, y_start + PAD_Y + j * LINE_HT)
            pdf.cell(col_w - PAD_X * 2, LINE_HT, line)

    pdf.set_xy(x_start, y_start + row_h)


def markdown_to_pdf(md_text: str) -> bytes:
    m = re.search(r"^#\s", md_text, re.MULTILINE)
    if m:
        md_text = md_text[m.start():]
    elements = _md_to_plain(md_text)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=22)

    font = "Helvetica"
    if _FONT:
        pdf.add_font("CJK", "", _FONT)
        pdf.add_font("CJK", "B", _FONT)
        font = "CJK"

    pdf.add_page()
    page_h = pdf.h
    b_margin = 22

    eff_w = pdf.w - pdf.l_margin - pdf.r_margin
    max_w = 90
    tbl_cols = 0
    tbl_col_w = 0
    row_buffer = []

    for elem in elements:
        kind = elem[0]

        if kind in ("row_h", "row_d"):
            if row_buffer:
                _render_table_row(pdf, row_buffer, tbl_col_w, font, page_h, b_margin)
                row_buffer = []
            tbl_cols = int(elem[1])
            tbl_col_w = eff_w / tbl_cols
            continue

        if kind == "row_end":
            if row_buffer:
                _render_table_row(pdf, row_buffer, tbl_col_w, font, page_h, b_margin)
                row_buffer = []
            continue

        if kind in ("cell_h", "cell_d"):
            row_buffer.append(elem)
            continue

        if kind == "blank":
            pdf.ln(2.5)
            continue

        if kind == "sep":
            pdf.ln(4)
            y = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin + 20, y, pdf.w - pdf.r_margin - 20, y)
            pdf.ln(5)
            continue

        if kind == "h1":
            text = _filter_emoji(_clean(elem[1]))
            pdf.set_font(font, "B", 18)
            pdf.set_text_color(13, 110, 143)
            pdf.multi_cell(eff_w, 9, "\n".join(_wrap(text, max_w)), align="C")
            pdf.ln(3)
        elif kind == "h2":
            text = _filter_emoji(_clean(elem[1]))
            pdf.set_font(font, "B", 14)
            pdf.set_text_color(13, 110, 143)
            pdf.multi_cell(eff_w, 7.5, "\n".join(_wrap(text, max_w)))
            pdf.ln(2)
        elif kind == "h3":
            text = _filter_emoji(_clean(elem[1]))
            pdf.set_font(font, "B", 12)
            pdf.set_text_color(13, 110, 143)
            pdf.multi_cell(eff_w, 7, "\n".join(_wrap(text, max_w)))
            pdf.ln(2)
        elif kind == "item":
            text = _filter_emoji(_clean(elem[1]))
            pdf.set_font(font, "", 11)
            pdf.set_text_color(0, 0, 0)
            lines = _wrap(text, max_w - 6)
            out = ""
            for i, l in enumerate(lines):
                out += ("  - " if i == 0 else "    ") + l + "\n"
            pdf.multi_cell(eff_w, 6, out.rstrip("\n"))
            pdf.ln(1)
        elif kind == "item_n":
            num, text = elem[1], elem[2]
            text = _filter_emoji(_clean(text))
            pdf.set_font(font, "", 11)
            pdf.set_text_color(0, 0, 0)
            lines = _wrap(text, max_w - 8)
            out = ""
            for i, l in enumerate(lines):
                out += (f"  {num}. " if i == 0 else "     ") + l + "\n"
            pdf.multi_cell(eff_w, 6, out.rstrip("\n"))
            pdf.ln(1)
        elif kind == "text":
            text = _filter_emoji(_clean(elem[1]))
            pdf.set_font(font, "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(eff_w, 6, "\n".join(_wrap(text, max_w)))
            pdf.ln(2)

    if row_buffer:
        _render_table_row(pdf, row_buffer, tbl_col_w, font, page_h, b_margin)

    pdf.set_auto_page_break(auto=False)
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.2)
    pdf.set_y(-16)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_font(font, "", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.set_y(-10)
    pdf.cell(0, 4, "UVAIS-Agent 自动生成", align="C")

    return bytes(pdf.output())
