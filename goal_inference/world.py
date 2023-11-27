import typing
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

Orientation = Enum("Orientation", ["HORIZONTAL", "VERTICAL"])
Lookups = Enum("Lookups", ["KEY", "HORIZONTAL", "VERTICAL"])


class Wall(BaseModel):
    pos: typing.Tuple[int, int]
    orientation: Orientation


class Door(Wall):
    key_id: int


class Key(BaseModel):
    pos: typing.Tuple[int, int]
    identifier: int


class World:
    def __init__(
        self,
        shape: typing.Tuple[int, int],
        keys: typing.List[Key],
        doors: typing.List[Door],
        walls: typing.List[Wall],
    ) -> None:
        self.shape = shape
        self.keys = keys
        self.doors = doors
        self.walls = walls
        self.lookup = self.validate_and_create_lookup()

    def validate_and_create_lookup(
        self,
    ) -> typing.Callable[
        [typing.Tuple[int, int], Lookups],
        typing.Optional[typing.Union[Wall, Door, Key]],
    ]:
        key_lookup: typing.Dict[
            typing.Tuple[int, int], typing.Optional[Key]
        ] = defaultdict(lambda: None)
        horizontal_lookup: typing.Dict[
            typing.Tuple[int, int], typing.Optional[typing.Union[Wall, Door]]
        ] = defaultdict(lambda: None)
        vertical_lookup: typing.Dict[
            typing.Tuple[int, int], typing.Optional[typing.Union[Wall, Door]]
        ] = defaultdict(lambda: None)

        key_ids = [key.identifier for key in self.keys]
        door_keys = [door.key_id for door in self.doors]
        assert len(key_ids) == len(set(key_ids))
        assert len(door_keys) == len(set(door_keys))
        assert set(key_ids) == set(door_keys)

        for barrier in self.doors + self.walls:
            assert (
                0 <= barrier.pos[0] < self.shape[0]
                and 0 <= barrier.pos[1] < self.shape[1]
            )
            if barrier.orientation is Orientation.HORIZONTAL:
                assert (
                    barrier.pos not in horizontal_lookup
                ), "Cannot have two barriers in same location"
                horizontal_lookup[barrier.pos] = barrier
            else:
                assert (
                    barrier.pos not in vertical_lookup
                ), "Cannot have two barriers in same location"
                vertical_lookup[barrier.pos] = barrier

        for door in self.doors:
            if door.orientation is Orientation.HORIZONTAL:
                assert door.pos not in vertical_lookup, "Doors cannot share corners"
            else:
                assert door.pos not in horizontal_lookup, "Doors cannot share corners"

        for key in self.keys:
            assert 0 <= key.pos[0] < self.shape[0] and 0 <= key.pos[1] < self.shape[1]
            assert key.pos not in key_lookup
            key_lookup[key.pos] = key

        def lookup(
            pos: typing.Tuple[int, int],
            lookup_type: Lookups,
        ) -> typing.Optional[typing.Union[Wall, Door, Key]]:
            if lookup_type is Lookups.KEY:
                return key_lookup[pos]
            elif lookup_type is Lookups.HORIZONTAL:
                return horizontal_lookup[pos]
            else:
                return vertical_lookup[pos]

        return lookup

    def get_accessible_neighbors(
        self, pos: typing.Tuple[int, int], key_id: typing.Optional[int] = None
    ) -> typing.List[typing.Tuple[int, int]]:
        assert 0 <= pos[0] < self.shape[0] and 0 <= pos[1] < self.shape[1]
        accessible_neighbors = []
        if pos[0] > 0:  # left
            barrier = self.lookup((pos[0], pos[1]), Lookups.VERTICAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((pos[0] - 1, pos[1]))
        if pos[0] < self.shape[0] - 1:  # right
            barrier = self.lookup((pos[0] + 1, pos[1]), Lookups.VERTICAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((pos[0] + 1, pos[1]))
        if pos[1] > 0:  # up
            barrier = self.lookup((pos[0], pos[1]), Lookups.HORIZONTAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((pos[0], pos[1] - 1))
        if pos[1] < self.shape[1] - 1:  # down
            barrier = self.lookup((pos[0], pos[1] + 1), Lookups.HORIZONTAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((pos[0], pos[1] + 1))
        return accessible_neighbors


def generate_world(seed):
    # TODO: randomly generate world from seed
    # e.g. choose layout (from existing set?), place keys, pick which key is for middle door, etc
    pass


example_world = World(
    shape=(30, 40),
    keys=[Key(pos=(2, 1), identifier=1)],
    doors=[Door(pos=(14, 20), orientation=Orientation.HORIZONTAL, key_id=1)],
    walls=[
        Wall(pos=(i, 20), orientation=Orientation.HORIZONTAL)
        for i in list(range(14)) + list(range(15, 30))
    ],
)
