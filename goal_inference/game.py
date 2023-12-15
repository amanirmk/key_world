import typing
import tkinter as tk
import time
from PIL import Image, ImageTk  # type: ignore[import-untyped]
from goal_inference.world import World, Door, Key, Lookups, Pos, MainDoor
from goal_inference.agents import Knower, Watcher
from itertools import product
import numpy as np
from enum import Enum
import pandas as pd  # type: ignore[import-untyped]
import os
import pathlib

Updated = Enum("Updated", ["WATCHER", "KNOWER"])


class Game:
    def __init__(
        self,
        world: World,
        human_player: bool,
        alpha: int,
        update_criteria: typing.Tuple[str, float],
        record: bool = False,
        output_folder: str = "./outputs",
        csv_name: str = "log.csv",
    ) -> None:
        self.BOX_SIZE = 20
        self.WALL_THICKNESS = 3
        self.FLOOR_IMAGE = "goal_inference/images/floor.png"
        self.KEY_IMAGE = "goal_inference/images/key.png"
        self.AGENT_IMAGE = "goal_inference/images/agent.png"
        self.record = record
        self.output_folder = output_folder
        self.csv_name = csv_name
        self.world = world
        self.knower = Knower(
            self.world.knower_start,
            self.world,
            self.world.maindoor.key_id,  # the Knower knows the goal key
        )
        self.watcher = Watcher(
            self.world.watcher_start,
            self.world,
            self.knower,
            self.wait_for_key_press,
            is_human=human_player,
            alpha=alpha,
            update_criteria=update_criteria,
        )

        self.window = tk.Tk()
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

        self.key_pressed = tk.StringVar()
        self.window.bind("<KeyPress>", self.on_key_press)

        self.play()

    def log_step(self, log_dict):
        path = pathlib.Path(self.output_folder) / self.csv_name
        if os.path.exists(path):
            keys = set(pd.read_csv(path, index_col=False, nrows=0).columns)
            assert (
                keys == log_dict.keys()
            ), f"mismatch in csv keys: {keys.symmetric_difference(log_dict.keys())}"
            pd.DataFrame.from_dict(log_dict).to_csv(
                path, mode="a", index=False, header=False
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict(log_dict).to_csv(path, index=False)

    def on_key_press(self, event):
        self.key_pressed.set(event.char)

    def wait_for_key_press(self):
        self.key_pressed.set("")
        self.window.wait_variable(self.key_pressed)
        pressed_key = self.key_pressed.get()
        return pressed_key

    def key_colors(self, key_id: int) -> typing.Tuple[int, int, int]:
        if key_id == -1:  # door
            return (200, 200, 200)
        colors = [
            (6, 68, 191),
            (7, 199, 242),
            (242, 227, 19),
            (242, 159, 5),
            (242, 5, 5),
        ]
        assert 0 <= key_id < len(colors)
        return colors[key_id]

    def update_images(
        self,
        update_positions: typing.List[Pos] = [],
        last_updated: typing.Optional[Updated] = None,
    ) -> None:
        positions = update_positions or [
            Pos((x, y))
            for x, y in product(range(self.world.shape[0]), range(self.world.shape[1]))
        ]
        for pos in positions:
            x, y = pos
            key: typing.Optional[Key] = self.world.lookup(pos, Lookups.KEY)  # type: ignore[assignment]
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

            h_barrier = self.world.lookup(pos, Lookups.HORIZONTAL)
            v_barrier = self.world.lookup(pos, Lookups.VERTICAL)
            if h_barrier is None and v_barrier is None:
                anchor = "center"
            else:
                anchor = ""
                if h_barrier:
                    anchor += "n"
                if v_barrier:
                    anchor += "w"
            if isinstance(h_barrier, Door) or isinstance(v_barrier, Door):
                r, g, b = self.key_colors(-1)
                bg = f"#{r:02x}{g:02x}{b:02x}"
            else:
                bg = "black"
            self.grid[y][x].config(image=image, anchor=anchor, bg=bg)  # type: ignore[call-overload]
            self.grid[y][x].image = image  # type: ignore[attr-defined]

        to_update = []
        if not last_updated or last_updated is Updated.KNOWER:
            to_update.append(self.knower.pos)
        if not last_updated or last_updated is Updated.WATCHER:
            to_update.append(self.watcher.pos)
        for x, y in to_update:
            image = Image.open(self.AGENT_IMAGE)
            image = image.resize(
                (self.BOX_SIZE, self.BOX_SIZE), Image.Resampling.LANCZOS
            )
            image = ImageTk.PhotoImage(image)
            self.grid[y][x].config(image=image)  # type: ignore[call-overload]
            self.grid[y][x].image = image  # type: ignore[attr-defined]

    def play(self):
        turn = 0
        self.update_images()
        self.window.update_idletasks()
        self.window.update()

        while not self.world.maindoor.is_open:
            turn += 1
            if turn % 2:
                old_pos = self.watcher.pos
                start_time = time.time()
                new_pos = self.watcher.move()
                end_time = time.time()
                last_updated = Updated.WATCHER
                reaction_time = end_time - start_time
                if self.record:
                    log_dict = {
                        "watcher_pos": [self.watcher.pos],
                        "knower_pos": [self.knower.pos],
                        "reaction_time": [reaction_time],
                    }
                    self.log_step(log_dict)
            else:
                old_pos = self.knower.pos
                new_pos = self.knower.move()
                last_updated = Updated.KNOWER

            self.update_images([old_pos, new_pos], last_updated)
            if self.world.at_main_door(
                self.watcher.pos,
                self.watcher.key.identifier if self.watcher.key else None,
            ) and self.world.at_main_door(
                self.knower.pos, self.knower.key.identifier if self.knower.key else None
            ):
                # if both agents are at the door with the correct key, open the door
                self.world.maindoor.is_open = True
            self.window.update_idletasks()
            self.window.update()
