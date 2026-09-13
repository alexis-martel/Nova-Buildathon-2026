from frontend import markdown_to_tk
import tkinter as tk


WELCOME_MARKDOWN = """# New Evaluation

## Welcome to NeuroEval!

You will be taking a series of N-Back tests **while wearing an EEG cap** to monitor the progression of your mental health.

The first test establishes a **baseline** reading that will serve as a reference for evaluating the difficulty of the subsequent tests. They will go up in difficulty based on your **computed mental effort**.

## Test Directions

The N-Back test shows a **sequence** of images, each staying on-screen for a short time. If you are taking a **1-Back** test, you must press the *Match* button when you see the **same** image twice in a row. If taking a 2-Back test, click on *Match* when you see the image that appeared **two** steps earlier, and so on for 3-Back, 4-Back and 5-Back.

Press the *Start Evaluation* button below when you're ready."""


def show_welcome_ui(on_click):
    """Shows a welcome window from where to launch the `on_click` function"""
    root = tk.Tk()
    root.title("NeuroEval")
    root.geometry("500x550")
    text = tk.Text(
        root,
        wrap="word",
        borderwidth=0,
        highlightthickness=0,
        bg=root.cget("bg"),
        selectbackground=root.cget("bg"),
    )
    markdown_to_tk.configure_markdown_tags(text)
    markdown_to_tk.insert_markdown(text, WELCOME_MARKDOWN)
    tk.Button(
        root, text="Start Evaluation", command=lambda: (root.destroy(), on_click())
    ).pack(side=tk.BOTTOM, pady=20)
    text.pack(fill="both", expand=True, padx=20, pady=20)
    root.mainloop()
