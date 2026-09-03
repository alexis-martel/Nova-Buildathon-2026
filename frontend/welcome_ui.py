import tkinter as tk


def show_welcome_ui(on_click):
    """Shows a welcome window from where to launch the `on_click` function"""
    root = tk.Tk()
    root.title("welcome")
    root.geometry("500x500")
    tk.Label(root, text="welcome to the awesome n-back test with eeg").pack()
    tk.Button(root, text="let's go", command=lambda: (root.destroy(), on_click())).pack(side=tk.BOTTOM, pady=20)
    root.mainloop()
