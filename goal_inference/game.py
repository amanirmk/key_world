import typing
import tkinter as tk
from PIL import Image, ImageTk  # type: ignore[import-untyped]
from goal_inference.world import World, Door, Key, Lookups
from itertools import product
import numpy as np


class Game:
    def __init__(self, world: World) -> None:
        self.BOX_SIZE = 20
        self.WALL_THICKNESS = 3
        self.FLOOR_IMAGE = "goal_inference/images/floor.png"
        self.KEY_IMAGE = "goal_inference/images/key.png"
        self.AGENT_IMAGE = "goal_inference/images/agent.png"

        self.window = tk.Tk()
        self.world = world
        self.window.geometry(
            f"{self.world.shape[0]*self.BOX_SIZE}x{self.world.shape[1]*self.BOX_SIZE}"
        )
        self.window.resizable(0, 0)  # type: ignore[call-overload]
        self.window.title("Goal Inference Game")
        self.grid = [
            [
                tk.Label(
                    self.window,
                    borderwidth=0,
                    bg="black",
                    padx=self.WALL_THICKNESS,
                    pady=self.WALL_THICKNESS,
                )
                for _ in range(self.world.shape[0])
            ]
            for _ in range(self.world.shape[1])
        ]
        for x, y in product(range(self.world.shape[0]), range(self.world.shape[1])):
            self.grid[y][x].grid(row=y, column=x)

        self.update_images()
        self.window.mainloop()
        # TODO: add controls etc

    def key_colors(self, key_id):
        colors = [
            (6, 68, 191),
            (7, 199, 242),
            (242, 227, 19),
            (242, 159, 5),
            (242, 5, 5),
        ]
        assert 0 <= key_id < len(colors)
        return colors[key_id]

    def update_images(self) -> None:
        # TODO: load images once, then only update *keys* (and agent) for efficiency
        for y in range(self.world.shape[1]):
            for x in range(self.world.shape[0]):
                key: typing.Optional[Key] = self.world.lookup((x, y), Lookups.KEY)  # type: ignore[assignment]
                if key:
                    key_color = self.key_colors(key.identifier)
                    key_image = Image.open(self.KEY_IMAGE).convert("1")
                    image = Image.new("RGB", key_image.size, key_color)
                    npimg = np.array(image)
                    npimg[np.where(np.array(key_image))] = (255, 255, 255)
                    image = Image.fromarray(npimg)
                else:
                    image = Image.open(self.FLOOR_IMAGE)

                image = image.resize(
                    (self.BOX_SIZE, self.BOX_SIZE), Image.Resampling.LANCZOS
                )
                image = ImageTk.PhotoImage(image)

                h_barrier = self.world.lookup((x, y), Lookups.HORIZONTAL)
                v_barrier = self.world.lookup((x, y), Lookups.VERTICAL)
                if h_barrier is None and v_barrier is None:
                    anchor = "center"
                else:
                    anchor = ""
                    if h_barrier:
                        anchor += "n"
                    if v_barrier:
                        anchor += "w"
                if isinstance(h_barrier, Door) or isinstance(v_barrier, Door):
                    door: Door = h_barrier or v_barrier  # type: ignore[assignment]
                    r, g, b = self.key_colors(door.key_id)
                    bg = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    bg = "black"
                self.grid[y][x].config(image=image, anchor=anchor, bg=bg)  # type: ignore[call-overload]
                self.grid[y][x].image = image  # type: ignore[attr-defined]
