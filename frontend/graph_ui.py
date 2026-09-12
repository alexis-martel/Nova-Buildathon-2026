import io
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.ticker import MaxNLocator


def _make_accuracy_figure(scores: list[dict]) -> plt.Figure:
    entries = scores[1:]
    x = range(1, len(entries) + 1)
    y = [e["accuracy"] for e in entries]
    colors = ["red" if not e["high_effort"] else "green" for e in entries]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.scatter(x, y, c=colors)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig

def show_graph_window(scores: list[dict]):
    fig = _make_accuracy_figure(scores) 
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    root = tk.Tk()
    img = tk.PhotoImage(data=buf.getvalue(), master=root)
    tk.Label(root, image=img).pack()
    root.mainloop()
