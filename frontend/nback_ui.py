import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def start_n_back_ui(seq: list[Path], interval: float, window_title: str) -> list[int]:
    """Run the n-back test UI and return the indices where "Match" was pressed.

    Args:
        seq: PNG image paths to display, in the order they must appear.
        interval: Seconds each image stays on screen before advancing.

    Returns:
        The indices into `seq` for which the user pressed "Match" while
        that image was on screen. An index can appear more than once if
        "Match" was pressed multiple times for the same image.
    """
    results: list[int] = []
    current_index = 0
    current_image = None  # kept alive so Tk doesn't garbage-collect it

    root = tk.Tk()
    root.title(window_title)
    root.geometry("500x500")

    image_label = tk.Label(root)
    image_label.pack(expand=True)

    def on_match() -> None:
        if current_index < len(seq):
            results.append(current_index)

    match_button = tk.Button(root, text="Match", command=on_match)
    match_button.pack(side=tk.BOTTOM, pady=20)

    def show_current() -> None:
        nonlocal current_image
        if current_index >= len(seq):
            root.destroy()
            return
        img = Image.open(seq[current_index])
        img.thumbnail((250, 250))
        current_image = ImageTk.PhotoImage(img)
        image_label.configure(image=current_image)
        root.after(int(interval * 1000), hide_current)

    def hide_current() -> None:
        image_label.configure(image="")
        root.after(300, advance)

    def advance() -> None:
        nonlocal current_index
        try:
            current_index += 1
            show_current()
        except tk.TclError:
            pass  # window was already closed by the user

    show_current()
    root.mainloop()

    return results
