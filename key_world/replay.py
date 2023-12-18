import typing
from key_world.world import World, Pos
from key_world.agents import Knower, Watcher
import pandas as pd  # type: ignore[import-untyped]
import numpy as np
import os
import copy


class Replay:
    def __init__(
        self,
        world: World,
        human_csv: str,
        alpha: float,
        update_criteria: typing.Tuple[str, float],
    ) -> None:
        self.world = copy.deepcopy(world)
        self.human_csv = human_csv
        self.alpha = alpha
        self.update_criteria = update_criteria

        assert os.path.exists(self.human_csv)
        self.human_data = pd.read_csv(self.human_csv, index_col=False)
        knower_move_list = [Pos(eval(pos)) for pos in self.human_data["knower_pos"]]
        watcher_move_list = [Pos(eval(pos)) for pos in self.human_data["watcher_pos"]]
        self.knower = Knower(
            self.world.knower_start,
            self.world,
            self.world.maindoor.key_id,
            move_list=knower_move_list,
        )
        self.watcher = Watcher(
            self.world.watcher_start,
            self.world,
            self.knower,
            None,
            mode="replay",
            move_list=watcher_move_list,
            alpha=alpha,
            update_criteria=update_criteria,
        )

    def replay(self):
        turn = 0
        while not self.world.maindoor.is_open:
            turn += 1
            if turn % 2:
                self.watcher.move()
            else:
                self.knower.move()
            if self.world.at_main_door(
                self.watcher.pos,
                self.watcher.key.identifier if self.watcher.key else None,
            ) and self.world.at_main_door(
                self.knower.pos, self.knower.key.identifier if self.knower.key else None
            ):
                # if both agents are at the door with the correct key, open the door
                self.world.maindoor.is_open = True
        replay_data = pd.DataFrame.from_dict(
            {
                "log_likelihood": self.watcher.log_likelihood,
                "action_prob": self.watcher.action_prob,
                "goal_prob": self.watcher.goal_prob,
                "action_surprisal": -np.log2(np.array(self.watcher.action_prob)),
                "goal_surprisal": -np.log2(np.array(self.watcher.goal_prob)),
                "reaction_time": self.human_data["reaction_time"],
            }
        )
        return replay_data
