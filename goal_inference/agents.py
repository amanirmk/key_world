import abc
import typing
import numpy as np
from goal_inference.world import World, Door, Key, Pos, Lookups


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
        valid_positions = [o[0] for o in options]
        while pos not in valid_positions:
            pos = self.choose_move()

        door: typing.Optional[Door] = options[valid_positions.index(pos)][1]
        if door:
            self.world.doors.remove(door)

        self.pos = pos
        key: typing.Optional[Key] = self.world.lookup(self.pos, Lookups.KEY)  # type: ignore[assignment]
        if key:
            self.pickup_key(key)
        return self.pos

    def putdown_key(self):
        self.key.pos = self.pos
        self.world.keys.append(self.key)
        self.key = None

    def pickup_key(self, key: Key):
        if self.key:
            self.putdown_key()
        self.world.keys.remove(key)
        self.key = key

    @abc.abstractmethod
    def choose_move(self) -> Pos:
        raise NotImplementedError()


class Knower(Agent):
    def choose_move(self) -> Pos:
        # TODO: this is temporary, replace with real algorithm
        options = self.world.get_accessible_neighbors(
            self.pos, self.key.identifier if self.key else None
        )
        valid_positions = [o[0] for o in options]
        return valid_positions[np.random.randint(len(valid_positions))]


class Watcher(Agent):
    def choose_move(self) -> Pos:
        # TODO: this is temporary, replace with real algorithm
        options = self.world.get_accessible_neighbors(
            self.pos, self.key.identifier if self.key else None
        )
        valid_positions = [o[0] for o in options]
        return valid_positions[np.random.randint(len(valid_positions))]
