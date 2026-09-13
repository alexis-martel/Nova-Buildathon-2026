import re
import tkinter as tk

INLINE_PATTERN = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")


def configure_markdown_tags(
    text_widget: tk.Text, base_family: str = "Helvetica"
) -> None:
    """Set up the tags insert_markdown() relies on. Call once per widget."""
    text_widget.tag_configure(
        "h1", font=(base_family, 20, "bold"), spacing1=4, spacing3=14
    )
    text_widget.tag_configure(
        "h2", font=(base_family, 15, "bold"), spacing1=12, spacing3=8
    )
    text_widget.tag_configure("body", font=(base_family, 11), spacing3=10)
    # bold/italic are combined with h1/h2/body via multiple tags on the
    # same range, so they only need to toggle the weight/slant, not the
    # base size -- Tk applies the last-configured tag's font wholesale, so
    # give bold/italic explicit sizes matching "body" (the common case).
    text_widget.tag_configure("bold", font=(base_family, 11, "bold"))
    text_widget.tag_configure("italic", font=(base_family, 11, "italic"))


def insert_markdown(text_widget: tk.Text, markdown: str) -> None:
    """Parse `markdown` and insert it into `text_widget` with styling tags.

    Supports: '# ' and '## ' line-prefixes as headers, **bold**, *italic*,
    and blank lines as paragraph breaks. Anything else is inserted as plain
    body text. Leaves the widget in a disabled (read-only) state -- call
    text_widget.config(state="normal") first if you need to edit it after.
    """

    def insert_inline(line: str, base_tag: str) -> None:
        for part in INLINE_PATTERN.split(line):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                text_widget.insert("end", part[2:-2], (base_tag, "bold"))
            elif part.startswith("*") and part.endswith("*"):
                text_widget.insert("end", part[1:-1], (base_tag, "italic"))
            else:
                text_widget.insert("end", part, (base_tag,))

    text_widget.config(state="normal")
    text_widget.delete("1.0", "end")

    for raw_line in markdown.strip("\n").splitlines():
        line = raw_line.strip()
        if not line:
            text_widget.insert("end", "\n")
            continue
        if line.startswith("## "):
            insert_inline(line[3:], "h2")
        elif line.startswith("# "):
            insert_inline(line[2:], "h1")
        else:
            insert_inline(line, "body")
        text_widget.insert("end", "\n")

    text_widget.config(state="disabled")
