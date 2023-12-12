import typing
import random
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

Orientation = Enum("Orientation", ["HORIZONTAL", "VERTICAL"])
Lookups = Enum("Lookups", ["KEY", "HORIZONTAL", "VERTICAL"])

Pos = typing.NewType("Pos", typing.Tuple[int, int])


class Wall(BaseModel):
    pos: Pos
    orientation: Orientation


class Door(Wall):
    key_id: int

    def __hash__(self):
        return 3 * hash(self.key_id) + hash("Door")


class MainDoor(Door):
    is_open: bool


class Key(BaseModel):
    pos: Pos
    identifier: int


class World:
    # TODO: designate end door and implement end game logic
    def __init__(
        self,
        shape: typing.Tuple[int, int],
        knower_start: Pos,
        watcher_start: Pos,
        keys: typing.List[Key],
        doors: typing.List[Door],
        maindoor: MainDoor,
        walls: typing.List[Wall],
    ) -> None:
        self.shape = shape
        self.knower_start = knower_start
        self.watcher_start = watcher_start
        self.keys = keys
        self.doors = doors
        self.doors.append(maindoor)
        self.maindoor = maindoor
        self.walls = walls
        lookups = self.validate_and_create_lookup()
        self.key_lookup = lookups[0]
        self.horizontal_lookup = lookups[1]
        self.vertical_lookup = lookups[2]

    def lookup(
        self, pos: Pos, lookup_type: Lookups
    ) -> typing.Optional[typing.Union[Wall, Door, Key]]:
        if lookup_type is Lookups.KEY:
            return self.key_lookup[pos]
        elif lookup_type is Lookups.HORIZONTAL:
            return self.horizontal_lookup[pos]
        else:
            return self.vertical_lookup[pos]

    def remove_door(self, door: Door):
        self.doors.remove(door)
        if door.orientation is Orientation.HORIZONTAL:
            del self.horizontal_lookup[door.pos]
        else:
            del self.vertical_lookup[door.pos]

    def remove_key(self, key: Key):
        self.keys.remove(key)
        del self.key_lookup[key.pos]

    def add_key(self, key: Key):
        self.keys.append(key)
        self.key_lookup[key.pos] = key
        lookups = self.validate_and_create_lookup()
        self.key_lookup = lookups[0]
        self.horizontal_lookup = lookups[1]
        self.vertical_lookup = lookups[2]

    def lookup(
        self, pos: Pos, lookup_type: Lookups
    ) -> typing.Optional[typing.Union[Wall, Door, Key]]:
        if lookup_type is Lookups.KEY:
            return self.key_lookup[pos]
        elif lookup_type is Lookups.HORIZONTAL:
            return self.horizontal_lookup[pos]
        else:
            return self.vertical_lookup[pos]

    def remove_door(self, door: Door):
        self.doors.remove(door)
        if door.orientation is Orientation.HORIZONTAL:
            del self.horizontal_lookup[door.pos]
        else:
            del self.vertical_lookup[door.pos]

    def remove_key(self, key: Key):
        self.keys.remove(key)
        del self.key_lookup[key.pos]

    def add_key(self, key: Key):
        self.keys.append(key)
        self.key_lookup[key.pos] = key

    def validate_and_create_lookup(
        self,
    ) -> typing.Tuple[
        typing.Dict[Pos, typing.Optional[Key]],
        typing.Dict[Pos, typing.Optional[typing.Union[Wall, Door]]],
        typing.Dict[Pos, typing.Optional[typing.Union[Wall, Door]]],
    ) -> typing.Tuple[
        typing.Dict[Pos, typing.Optional[Key]],
        typing.Dict[Pos, typing.Optional[typing.Union[Wall, Door]]],
        typing.Dict[Pos, typing.Optional[typing.Union[Wall, Door]]],
    ]:
        key_lookup: typing.Dict[Pos, typing.Optional[Key]] = defaultdict(lambda: None)
        horizontal_lookup: typing.Dict[
            Pos, typing.Optional[typing.Union[Wall, Door]]
        ] = defaultdict(lambda: None)
        vertical_lookup: typing.Dict[
            Pos, typing.Optional[typing.Union[Wall, Door]]
        ] = defaultdict(lambda: None)

        assert (
            0 <= self.knower_start[0] < self.shape[0]
            and 0 <= self.knower_start[1] < self.shape[1]
        )
        assert (
            0 <= self.watcher_start[0] < self.shape[0]
            and 0 <= self.watcher_start[1] < self.shape[1]
        )

        key_ids = [key.identifier for key in self.keys]
        door_keys = [door.key_id for door in self.doors]
        assert len(key_ids) == 2 * len(set(key_ids))
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

        return key_lookup, horizontal_lookup, vertical_lookup
        return key_lookup, horizontal_lookup, vertical_lookup

    def get_accessible_neighbors(
        self, pos: Pos, key_id: typing.Optional[int]
    ) -> typing.List[typing.Tuple[Pos, typing.Optional[typing.Union[Door, MainDoor]]]]:
        assert 0 <= pos[0] < self.shape[0] and 0 <= pos[1] < self.shape[1]
        accessible_neighbors = []
        if pos[0] > 0:  # left
            barrier = self.lookup(pos, Lookups.VERTICAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((Pos((pos[0] - 1, pos[1])), barrier))
        if pos[0] < self.shape[0] - 1:  # right
            barrier = self.lookup(Pos((pos[0] + 1, pos[1])), Lookups.VERTICAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((Pos((pos[0] + 1, pos[1])), barrier))
        if pos[1] > 0:  # up
            barrier = self.lookup(pos, Lookups.HORIZONTAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((Pos((pos[0], pos[1] - 1)), barrier))
        if pos[1] < self.shape[1] - 1:  # down
            barrier = self.lookup(Pos((pos[0], pos[1] + 1)), Lookups.HORIZONTAL)
            if barrier is None or (
                isinstance(barrier, Door) and barrier.key_id == key_id
            ):
                accessible_neighbors.append((Pos((pos[0], pos[1] + 1)), barrier))
        return accessible_neighbors

    def at_main_door(self, pos: Pos, key_id: typing.Optional[int]):
        x, y = self.maindoor.pos
        return (
            pos[0] == x
            and pos[1] == y
            or pos[1] == y - 1
            and self.maindoor.key_id == key_id
        )


# Example usage
example_world = World(
    shape=(30, 40),
    knower_start=Pos((14, 2)),
    watcher_start=Pos((17, 30)),
    keys=[Key(pos=Pos((2, 5)), identifier=1), Key(pos=Pos((21, 26)), identifier=1)],
    doors=[],
    maindoor=MainDoor(
        pos=Pos((14, 20)), orientation=Orientation.HORIZONTAL, key_id=1, is_open=False
    ),
    walls=[
        Wall(pos=Pos((i, 20)), orientation=Orientation.HORIZONTAL)
        for i in list(range(14)) + list(range(15, 30))
    ],
)
