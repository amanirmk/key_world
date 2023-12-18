import abc
import typing
import numpy as np
from key_world.world import World, Door, Key, Pos, Lookups, MainDoor
from key_world.algs_knower import get_moves
from key_world.algs_watcher import (
    predict_knower_move,
    choose_move_given_beliefs,
    update_beliefs,
    init_beliefs,
)


class Agent(abc.ABC):
    def __init__(
        self,
        pos: Pos,
        world: World,
    ) -> None:
        self.pos = pos
        self.world = world
        self.key: typing.Optional[Key] = None

    def move(self) -> Pos:
        pos = Pos((-1, -1))
        options = self.world.get_accessible_neighbors(
            self.pos, self.key.identifier if self.key else None
        )
        options.append((self.pos, None))
        valid_positions = [o[0] for o in options]
        while pos not in valid_positions:
            pos = self.choose_move()

        door: typing.Optional[Door] = options[valid_positions.index(pos)][1]
        if door:
            # otherwise remove door
            self.world.remove_door(door)

        self.pos = pos
        key: typing.Optional[Key] = self.world.lookup(self.pos, Lookups.KEY)  # type: ignore[assignment]
        if key:
            self.pickup_key(key)
        return self.pos

    def putdown_key(self, key):
        key.pos = self.pos
        self.world.add_key(key)
        self.key = None

    def pickup_key(self, new_key: Key):
        if self.key:
            old_key = self.key
            self.world.remove_key(new_key)
            self.putdown_key(old_key)
            self.key = new_key
        else:
            self.world.remove_key(new_key)
            self.key = new_key

    @abc.abstractmethod
    def choose_move(self) -> Pos:
        return NotImplemented


class Knower(Agent):
    # Knower agent knows the main door key id
    def __init__(
        self,
        pos: Pos,
        world: World,
        main_key_id: int,
        move_list: typing.List[Pos] = [],
    ) -> None:
        super().__init__(pos, world)
        self.main_key_id = main_key_id
        if move_list:
            self.move_list = move_list
        else:
            self.move_list = get_moves(world)
        self.num_moves = 0

    def choose_move(self) -> Pos:
        assert self.move_list is not None
        move = self.move_list[min(self.num_moves, len(self.move_list) - 1)]
        self.num_moves += 1
        return move


class Watcher(Agent):
    def __init__(
        self,
        pos: Pos,
        world: World,
        knower: Knower,
        wait_for_key_press,
        mode: str = "model",
        move_list: typing.Optional[typing.List[Pos]] = None,
        alpha: float = 1,
        update_criteria: typing.Tuple[str, float] = ("turn", 1),
    ) -> None:
        super().__init__(pos, world)
        self.predictions = None
        self.knower = knower
        self.alpha = alpha
        self.beliefs = init_beliefs(self.world, self.knower, self.alpha)
        self.mode = mode
        self.move_list = move_list
        self._wait_for_key_press = wait_for_key_press
        self.num_moves = 0
        self.update_criteria = update_criteria
        if self.mode == "replay":
            self.log_likelihood: typing.List[float] = []
            self.action_prob: typing.List[float] = []
            self.goal_prob: typing.List[float] = []

    def choose_move(self) -> Pos:
        if self.mode == "human":
            self.num_moves += 1
            return self.get_user_move()
        if self.should_update():
            self.beliefs = update_beliefs(self.knower, self.predictions, self.beliefs)
        move_dist = choose_move_given_beliefs(
            self, self.world, self.beliefs, alpha=self.alpha
        )
        if self.mode == "replay":
            assert self.move_list and self.num_moves < len(self.move_list)
            move = self.move_list[self.num_moves]
            self.log_likelihood.append(np.log(move_dist[move]))
            if self.predictions:
                self.action_prob.append(self.predictions[0][self.knower.pos])
                self.goal_prob.append(
                    self.predictions[1][max(self.beliefs, key=self.beliefs.get)][
                        self.knower.pos
                    ]
                )
            else:
                self.action_prob.append(np.nan)
                self.goal_prob.append(np.nan)
        elif self.mode == "model":
            move = max(move_dist, key=move_dist.get)
        else:
            return NotImplemented
        self.predictions = predict_knower_move(
            self.world, self.knower, self.beliefs, self.alpha
        )
        self.num_moves += 1
        return move

    def get_user_move(self) -> Pos:
        assert self.mode == "human"
        x, y = self.pos
        new_pos = None
        while new_pos is None:
            key = self._wait_for_key_press()
            if key == "w":
                new_pos = Pos((x, y - 1))
            elif key == "s":
                new_pos = Pos((x, y + 1))
            elif key == "a":
                new_pos = Pos((x - 1, y))
            elif key == "d":
                new_pos = Pos((x + 1, y))
            elif key == " ":
                new_pos = Pos((x, y))
        return new_pos

    def should_update(self) -> bool:
        if self.predictions is None:
            return False
        if self.update_criteria[0] == "turn":
            return self.num_moves % self.update_criteria[1] == 0
        elif self.update_criteria[0] == "action":
            return self.predictions[0][self.knower.pos] < self.update_criteria[1]
        elif self.update_criteria[0] == "goal":
            goal = max(self.beliefs, key=self.beliefs.get)
            return self.predictions[1][goal][self.knower.pos] < self.update_criteria[1]
        else:
            return NotImplemented
