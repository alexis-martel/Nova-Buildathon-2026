"""
markdown_to_tk.py

Tkinter has no built-in markdown rendering, so this converts a small subset
of markdown (# / ## headers, **bold**, *italic*) into a tk.Text widget with
styled tags. It's intentionally not a general-purpose markdown parser --
just enough to drop static app copy (like a welcome screen) into a Tk
window without hand-writing tag calls for every line.

Run this file directly (`python markdown_to_tk.py`) to preview the
NeuroEval welcome screen in its own window.
"""

import re
import tkinter as tk
from tkinter import font as tkfont

INLINE_PATTERN = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")


def configure_markdown_tags(text_widget: tk.Text, base_family: str = "Helvetica") -> None:
    """Set up the tags insert_markdown() relies on. Call once per widget."""
    text_widget.tag_configure(
        "h1", font=(base_family, 20, "bold"), spacing1=4, spacing3=14
    )
    text_widget.tag_configure(
        "h2", font=(base_family, 15, "bold"), spacing1=12, spacing3=8
    )
    text_widget.tag_configure(
        "body", font=(base_family, 11), spacing3=10
    )
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


WELCOME_MARKDOWN = """# New Evaluation

## Welcome to NeuroEval!

You will be taking a series of N-Back tests **while wearing an EEG cap** to monitor the progression of your mental health.

The first test establishes a **baseline** reading that will serve as a reference for evaluating the difficulty of the subsequent tests. They will go up in difficulty based on your **computed mental effort**.

## Test Directions

The N-Back test shows a **sequence** of images, each staying on-screen for a short time. If you are taking a **1-Back** test, you must press the *Match* button when you see the **same** image twice in a row. If taking a 2-Back test, click on *Match* when you see the image that appeared **two** steps earlier, and so on for 3-Back, 4-Back and 5-Back.

Press the *Start Evaluation* button below when you're ready.
"""



def build_welcome_window(on_start) -> tk.Tk:
    """Assemble the welcome screen. `on_start` is called when the button is clicked."""
    root = tk.Tk()
    root.title("NeuroEval")
    root.geometry("560x420")

    text = tk.Text(
        root,
        wrap="word",
        borderwidth=0,
        highlightthickness=0,
        padx=24,
        pady=20,
        font=tkfont.Font(family="Helvetica", size=11),
    )
    configure_markdown_tags(text)
    insert_markdown(text, WELCOME_MARKDOWN)
    text.pack(fill="both", expand=True)

    tk.Button(
        root,
        text="Start Evaluation",
        command=lambda: (root.destroy(), on_start()),
    ).pack(side=tk.BOTTOM, pady=20)

    return root


if __name__ == "__main__":
    def _on_start():
        print("Start Evaluation clicked.")

    build_welcome_window(_on_start).mainloop()
