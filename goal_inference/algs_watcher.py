# all the juicy stuff for the Watcher
import typing
import copy
import numpy as np
from goal_inference.world import World, Pos
from goal_inference.algs_knower import Node, find_path, make_get_neighbors
from math import isclose

T = typing.TypeVar("T")


def confirm_normalized(prob_dict):
    total = sum(prob_dict.values())
    assert isclose(total, 1, abs_tol=1e-4), f"sum={total}, dict={prob_dict}"


def normalize(prob_dict: typing.Dict[T, float], alpha: float) -> typing.Dict[T, float]:
    denom = sum(np.power(p, alpha) for p in prob_dict.values())
    assert denom > 0
    new_dict = {k: np.power(p, alpha) / denom for k, p in prob_dict.items()}
    confirm_normalized(new_dict)
    return new_dict


def get_knower_path(
    world: World, knower, goal_key_id: int
) -> typing.Dict[Pos, typing.Optional[typing.List[Pos]]]:
    start_node = Node(
        pos=knower.pos,
        key_id=knower.key.identifier if knower.key else None,
        used_key_ids=[],
        parent=None,
    )
    goal_node = Node(
        pos=Pos((world.maindoor.pos[0], world.maindoor.pos[1] - 1)),
        key_id=goal_key_id,
        used_key_ids=[],
        parent=None,
    )
    all_paths = {}
    stay_node = copy.deepcopy(start_node)
    stay_node.parent = start_node
    get_neighbors = make_get_neighbors(world)
    possible_nodes = get_neighbors(start_node) + [stay_node]
    for next_node in possible_nodes:
        path = find_path(next_node, goal_node, get_neighbors)
        all_paths[next_node.pos] = path
    return all_paths


def init_beliefs(world: World, knower, alpha: float) -> typing.Dict[int, float]:
    potential_goals = set(k.identifier for k in world.keys)
    if knower.key:
        potential_goals.add(knower.key.identifier)
    shortest_path_lengths = {}
    for goal in potential_goals:
        paths_to_goal = get_knower_path(world, knower, goal)
        shortest_path_lengths[goal] = min(len(p) for p in paths_to_goal)
    total_len = sum(shortest_path_lengths.values())
    beliefs = {
        goal: 1 - (length / total_len) for goal, length in shortest_path_lengths.items()
    }
    beliefs = normalize(beliefs, alpha)
    return beliefs


def get_p_next_given_goal_from_paths(
    paths: typing.Dict[Pos, typing.Optional[typing.List[Pos]]], alpha: float
) -> typing.Dict[Pos, float]:
    bad_nexts = []
    for pos, path in paths.items():
        if path is None:
            bad_nexts.append(pos)
            paths[pos] = []
    if len(bad_nexts) == len(paths):
        return {pos: 1/len(paths) for pos in paths} # uniform if unreachable goal
    total_len = sum(len(path) for path in paths.values())  # type: ignore[misc, arg-type]
    assert total_len > 0
    p_next_given_goal = {
        next_pos: 0 if pos in bad_nexts else 1 - (len(path) / total_len)  # type: ignore[arg-type]
        for next_pos, path in paths.items()
    }
    p_next_given_goal = normalize(p_next_given_goal, alpha)
    return p_next_given_goal


def get_all_p_next_given_goals(
    world: World, knower, beliefs, alpha: float
) -> typing.Dict[int, typing.Dict[Pos, float]]:
    p_next_given_goals = {}
    for goal in beliefs:
        paths_to_goal = get_knower_path(world, knower, goal)
        p_next_given_goal = get_p_next_given_goal_from_paths(paths_to_goal, alpha)
        p_next_given_goals[goal] = p_next_given_goal
    return p_next_given_goals


def predict_knower_move(world: World, knower, beliefs, alpha: float):
    p_next_given_goals = get_all_p_next_given_goals(world, knower, beliefs, alpha)
    p_next = infer_knower_move(beliefs, p_next_given_goals)
    return p_next, p_next_given_goals


def infer_knower_move(p_goal_priors, p_nexts_given_goals):
    p_nexts = {}
    for potential_goal, prior in p_goal_priors.items():
        for pos, prob in p_nexts_given_goals[potential_goal].items():
            if pos not in p_nexts:
                p_nexts[pos] = 0
            p_nexts[pos] += prob * prior
    confirm_normalized(p_nexts)
    return p_nexts


def choose_move_given_beliefs(watcher, world, beliefs, alpha: float):
    start_node = Node(
        pos=watcher.pos,
        key_id=watcher.key.identifier if watcher.key else None,
        used_key_ids=[],
        parent=None,
    )
    paths = {}
    get_neighbors = make_get_neighbors(world)
    stay = copy.deepcopy(start_node)
    stay.parent = start_node
    available_nodes = get_neighbors(start_node) + [stay]
    p_nexts_watcher = {node.pos: 0 for node in available_nodes}
    for potential_goal in beliefs:
        goal_node = Node(
            pos=world.maindoor.pos,
            key_id=potential_goal,
            used_key_ids=[],
            parent=None,
        )
        for next_node in available_nodes:
            path = find_path(next_node, goal_node, get_neighbors)
            paths[next_node.pos] = path
        p_nexts_given_goal_watcher = get_p_next_given_goal_from_paths(paths, alpha)
        for next_pos in p_nexts_given_goal_watcher:
            p_nexts_watcher[next_pos] += (
                p_nexts_given_goal_watcher[next_pos] * beliefs[potential_goal]
            )
    confirm_normalized(p_nexts_watcher)
    return p_nexts_watcher


def update_beliefs(knower, predictions, current_beliefs):
    p_action, p_action_given_goal = predictions
    new_beliefs = {}
    for goal in current_beliefs:
        new_beliefs[goal] = (
            p_action_given_goal[goal][knower.pos]
            * current_beliefs[goal]
            / p_action[knower.pos]
        )
    confirm_normalized(new_beliefs)
    return new_beliefs
