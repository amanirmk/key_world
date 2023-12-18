import typing
from collections import deque
from key_world.world import World, Key, Door, Pos, Lookups, Orientation
from pydantic import BaseModel
import copy


class Node(BaseModel):
    pos: Pos
    key_id: typing.Optional[int] = None
    used_key_ids: typing.List[int] = []
    opened_door_ids: typing.List[int] = []
    dropped_keys: typing.Dict[Pos, Key] = {}
    parent: typing.Optional["Node"] = None

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.pos == other.pos and self.key_id == other.key_id

    def __hash__(self):
        return (
            5 * hash(self.pos)
            + 3 * hash(self.key_id)
            + sum(
                hash(pos) * hash(k.identifier) for pos, k in self.dropped_keys.items()
            )
        )


def get_moves(world: World) -> typing.List[Pos]:
    start = Node(
        pos=world.knower_start,
    )
    assert world.maindoor.orientation is Orientation.HORIZONTAL
    goal_pos = Pos((world.maindoor.pos[0], world.maindoor.pos[1] - 1))
    goal = Node(
        pos=goal_pos,
        key_id=world.maindoor.key_id,
    )
    get_neighbors = make_get_neighbors(world)
    path = find_path(start, goal, get_neighbors)
    return path


def make_get_neighbors(
    world: World, key_agnostic=False
) -> typing.Callable[[Node], typing.List[Node]]:
    get_node = make_get_node(world, key_agnostic)

    def get_neighbors(node: Node) -> typing.List[Node]:
        options = set()
        key_ids: typing.Set[typing.Optional[int]] = set(node.used_key_ids)
        key_ids.add(node.key_id)
        if key_agnostic and node.key_id:
            # if you have a key, you have all keys
            key_ids.update([k.identifier for k in world.keys])
        for key_id in key_ids:
            options.update(world.get_accessible_neighbors(node.pos, key_id))
        return [get_node(node, opt) for opt in options]

    return get_neighbors


def make_get_node(
    world: World, key_agnostic: bool = False
) -> typing.Callable[[Node, typing.Tuple[Pos, typing.Optional[Door]]], Node]:
    def get_node(
        curr_node: Node,
        option: typing.Tuple[Pos, typing.Optional[Door]],
    ) -> Node:
        pos, door = option
        curr_key_id = curr_node.key_id
        used_key_ids = copy.deepcopy(curr_node.used_key_ids)
        opened_door_ids = copy.deepcopy(curr_node.opened_door_ids)
        dropped_keys = copy.deepcopy(curr_node.dropped_keys)

        if not key_agnostic:
            assert len(opened_door_ids) == 0

        # open door
        if key_agnostic:
            if door and door.key_id not in opened_door_ids:
                assert curr_key_id is not None
                opened_door_ids.append(door.key_id)
                used_key_ids.append(curr_key_id)
        else:
            if door and door.key_id == curr_key_id:
                used_key_ids.append(curr_key_id)

        # pick up key
        new_key: typing.Optional[Key] = world.lookup(pos, Lookups.KEY)  # type: ignore[assignment]
        if new_key and (
            new_key.identifier == curr_key_id
            or new_key.identifier in used_key_ids
            or any(new_key.identifier == k.identifier for k in dropped_keys.values())
        ):
            # key can no longer be picked up from original spot
            new_key = None
        assert not (
            new_key and pos in dropped_keys
        ), f"{pos}, {new_key}, {dropped_keys}"
        if pos in dropped_keys:
            # picking up previously-dropped key
            new_key = dropped_keys[pos]
            del dropped_keys[pos]
        if new_key and curr_key_id:
            dropped_key = Key(pos=pos, identifier=curr_key_id)
            dropped_keys[pos] = dropped_key
        new_key_id = new_key.identifier if new_key else curr_key_id

        child_node = Node(
            pos=pos,
            key_id=new_key_id,
            used_key_ids=used_key_ids,
            opened_door_ids=opened_door_ids,
            dropped_keys=dropped_keys,
            parent=curr_node,
        )
        return child_node

    return get_node


def find_path(start: Node, goal: Node, get_neighbors) -> typing.List[Pos]:
    queue = deque([start])
    seen = set([start])
    while queue:
        curr_node = queue.popleft()
        if curr_node == goal:
            return get_path_to(curr_node)
        for child_node in get_neighbors(curr_node):
            if child_node not in seen:
                seen.add(child_node)
                queue.append(child_node)
    raise RuntimeError("Could not find a path to the goal")


def get_path_to(node: Node, pos_seq: typing.List[Pos] = []) -> typing.List[Pos]:
    if node.parent:
        return get_path_to(node.parent, [node.pos] + pos_seq)
    return pos_seq
