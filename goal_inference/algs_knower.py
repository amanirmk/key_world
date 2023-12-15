import typing
from collections import deque
from goal_inference.world import World, Key, Door, Pos, Lookups, Orientation
from pydantic import BaseModel
import copy


class Node(BaseModel):
    pos: Pos
    key_id: typing.Optional[int]
    used_key_ids: typing.List[int]
    parent: typing.Optional["Node"]

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.pos == other.pos and self.key_id == other.key_id

    def __hash__(self):
        return 3 * hash(self.pos) + hash(self.key_id)


def get_moves(world: World) -> typing.Optional[typing.List[Pos]]:
    start = Node(pos=world.knower_start, key_id=None, used_key_ids=[], parent=None)
    assert world.maindoor.orientation is Orientation.HORIZONTAL
    goal_pos = Pos((world.maindoor.pos[0], world.maindoor.pos[1] - 1))
    goal = Node(
        pos=goal_pos,
        key_id=world.maindoor.key_id,
        used_key_ids=[],
        parent=None,
    )
    get_neighbors = make_get_neighbors(world)
    return find_path(start, goal, get_neighbors)


def make_get_neighbors(world: World) -> typing.Callable[[Node], typing.List[Node]]:
    get_node = make_get_node(world)

    def get_neighbors(node: Node) -> typing.List[Node]:
        options = set()
        for key_id in node.used_key_ids + [node.key_id]:
            options.update(world.get_accessible_neighbors(node.pos, key_id))
        return [get_node(node, opt) for opt in options]

    return get_neighbors


def make_get_node(
    world: World,
) -> typing.Callable[[Node, typing.Tuple[Pos, typing.Optional[Door]]], Node]:
    def get_node(
        curr_node: Node, option: typing.Tuple[Pos, typing.Optional[Door]]
    ) -> Node:
        pos, door = option
        curr_key_id = curr_node.key_id
        used_key_ids = copy.deepcopy(curr_node.used_key_ids)
        if door and door.key_id == curr_key_id:
            used_key_ids.append(curr_key_id)
            curr_key_id = None
        new_key: typing.Optional[Key] = world.lookup(pos, Lookups.KEY)  # type: ignore[assignment]
        new_key_id = new_key.identifier if new_key else curr_key_id
        return Node(
            pos=pos,
            key_id=new_key_id,
            used_key_ids=used_key_ids,
            parent=curr_node,
        )

    return get_node


def find_path(
    start: Node, goal: Node, get_neighbors
) -> typing.Optional[typing.List[Pos]]:
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
    # can't reach goal if requires picking up dropped key
    # just assume rational agent wouldn't do this
    return None


def get_path_to(node: Node, pos_seq: typing.List[Pos] = []) -> typing.List[Pos]:
    if node.parent:
        return get_path_to(node.parent, [node.pos] + pos_seq)
    return pos_seq
