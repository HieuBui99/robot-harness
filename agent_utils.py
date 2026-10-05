"""Optional camera viewer, isolated from the agent event loop."""

import base64
import io
import multiprocessing
import queue

from llm_providers.base import Image as ImageBlock
from llm_providers.base import Text


def _viewer(messages):
    import tkinter as tk

    from PIL import Image, ImageTk
    root = tk.Tk()
    root.title("OMX camera feedback")
    labels = []

    def tick():
        latest = None
        try:
            while True:
                latest = messages.get_nowait()
                if latest is None:
                    root.destroy()
                    return
        except queue.Empty:
            pass
        if latest:
            for label in labels:
                label.destroy()
            labels.clear()
            for index, (title, data) in enumerate(latest):
                picture = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
                picture.thumbnail((640, 480))
                photo = ImageTk.PhotoImage(picture)
                label = tk.Label(root, text=title, image=photo, compound="top")
                label.image = photo
                label.grid(row=0, column=index)
                labels.append(label)
        root.after(100, tick)

    root.after(100, tick)
    root.mainloop()


class ImageViewer:
    def __init__(self):
        context = multiprocessing.get_context("spawn")
        self.queue = context.Queue(maxsize=2)
        self.process = context.Process(target=_viewer, args=(self.queue,), daemon=True)
        self.process.start()

    def update(self, result):
        images = []
        title = "Camera"
        for block in result.blocks:
            if isinstance(block, Text):
                title = block.text
            elif isinstance(block, ImageBlock):
                images.append((title, block.data))
        if images:
            try:
                self.queue.put_nowait(images)
            except queue.Full:
                pass

    def close(self):
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.queue.close()
