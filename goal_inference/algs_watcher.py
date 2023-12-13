# all the juicy stuff for the Watcher
import typing

from goal_inference.world import World, Pos
from goal_inference.algs_knower import Node, find_path, make_get_neighbors
from goal_inference.agents import Knower


def get_knower_path(
    world: World, knower: Knower, potential_goal
) -> typing.Dict[Pos, typing.List[Pos]]:
    # shortest paths for the knower to move from its all next positions to the potential goal
    (x, y) = knower.pos
    # TODO: if start node is immediately through a door, then put key in used_keys
    start_node = Node(
        pos=knower.pos,
        key_id=knower.key.identifier if knower.key else None,
        used_key_ids=[],
        parent=None,
    )
    maxX = world.shape[0]
    assert world.shape[1] % 2 == 0
    maxY = world.shape[1] // 2

    all_paths = {}
    avalaible_pos = set(
        [
            Pos((min(x + 1, maxX), y)),
            Pos((max(x - 1, 0), y)),
            Pos((x, min(maxY, y + 1))),
            Pos((x, max(0, y - 1))),
        ]
    )
    for next_pos in avalaible_pos:
        start = Node(
            pos=next_pos,
            key_id=knower.key.identifier if knower.key else None,
            used_key_ids=[],
            parent=start_node,
        )
        goal = Node(
            pos=potential_goal.pos,
            key_id=knower.key.identifier if knower.key else None,
            used_key_ids=[],
            parent=None,
        )

        get_neighbors = make_get_neighbors(world)
        path = find_path(start, goal, get_neighbors)
        all_paths[next_pos] = path
    return all_paths


def get_p_next_given_goal(paths) -> typing.Dict[Pos, float]:
    """
    Given a dictionary of all possible paths from the knower's current position to a potential goal,
    returns a dictionary of the probability of moving to each possible next position, given the goal.

    The probability of moving to a given position is determined by the length of the shortest path from that position to the goal,
    normalized by the total length of all possible paths.
    """
    # p(a/pos |g) determined by the length of the shortest path from the next pos to the goal
    total_len = sum(len(path) for path in paths.values())
    if total_len == 0:
        # Handling the case where the knower is already at the goal
        return {next_pos: 0 for next_pos in paths.keys()}
    # TODO: add temperature
    p_next_given_goal = {
        next_pos: 1 - (len(path) / total_len) for next_pos, path in paths.items()
    }
    return p_next_given_goal


def get_knower_priors(world, knower):
    # find the potential goals for the knower
    potential_goals = set()
    if knower.key != None:
        # if the knower has a key, potential goals are the doors and the main door
        # TODO: make sure to do correct positions
        potential_goals.update(world.doors)
        potential_goals.update(world.maindoor)
    else:
        # otherwise, potential goals are the keys
        potential_goals.update(world.keys)  # TODO: filter keys here (AND NOT LINE 71)
        # TODO: make goals actually being in front of door with a certain key

    # a dictionary of all the paths for the knower to move from its current position to all potential goals
    all_paths = {potential_goal: {} for potential_goal in potential_goals}
    # a dictionary of the shortest path for the knower to move from its current position to all potential goals
    shortest_paths = {potential_goal: [] for potential_goal in potential_goals}
    # a dictionary of dictionary of the probability of moving to each possible next position, given the goal
    p_nexts_given_goals = {potential_goal: {} for potential_goal in potential_goals}

    for potential_goal in potential_goals:
        (_, goal_y) = potential_goal.pos
        maxY = world.shape[1] / 2
        if goal_y > maxY:  # in thw watcher's world
            potential_goals.remove(potential_goal)
            del all_paths[potential_goal]
            del shortest_paths[potential_goal]
            del p_nexts_given_goals[potential_goal]
            continue
        # get all the paths for the knower to move from its current position to a potential goal
        paths = get_knower_path(world, knower, potential_goal)
        # p(a/pos |g)
        p_next_given_goal = get_p_next_given_goal(paths)

        shortest_paths[potential_goal] = min(paths.values(), key=len)
        all_paths[potential_goal] = paths
        p_nexts_given_goals[potential_goal] = p_next_given_goal

    # a dictionary of the probability of the goal
    total_len = sum(len(path) for path in shortest_paths.values())

    p_goal_priors = {
        potential_goal: 1 - (len(path) / total_len)
        for potential_goal, path in shortest_paths.items()
    }

    return p_goal_priors, p_nexts_given_goals


def infer_knower_move(p_goal_priors, p_nexts_given_goals):
    # P(next_pos) for the knower to move from its current position
    p_nexts = {}
    for potential_goal, prior in p_goal_priors.items():
        for pos, prob in p_nexts_given_goals[potential_goal].items():
            if pos not in p_nexts:
                p_nexts[pos] = 0
            p_nexts[pos] += prob * prior
    # when accessing, need to pay attention to if the next_pos is in the p_nexts
    return p_nexts


def infer_goal_given_move(p_goal_priors, p_nexts_given_goals):
    # P(g|next_pos) infer the knower to move from knower's move
    p_goals_given_move = {}
    for next_pos in p_nexts_given_goals.values():
        for pos in next_pos:
            if pos not in p_goals_given_move:
                p_goals_given_move[pos] = {}

            total_prob = sum(
                p_nexts_given_goals[goal][pos] * p_goal_priors[goal]
                for goal in p_goal_priors
                if pos in p_nexts_given_goals[goal]
            )

            for goal in p_goal_priors:
                if pos in p_nexts_given_goals[goal]:
                    prob = (
                        (p_nexts_given_goals[goal][pos] * p_goal_priors[goal])
                        / total_prob
                        if total_prob > 0
                        else 0
                    )
                    p_goals_given_move[pos][goal] = prob

    return p_goals_given_move


def get_moves_dist(self, world: World, knower: Knower):
    p_goal_priors, p_nexts_given_goals = get_knower_priors(world, knower)
    p_knower_nexts = infer_knower_move(p_goal_priors, p_nexts_given_goals)
    p_goals_given_moves = infer_goal_given_move(p_goal_priors, p_nexts_given_goals)

    (x, y) = self.pos
    start_node = Node((x, y), key_id=knower.key.identifier, used_key_ids=[], parent=None)  # TODO: to add used_key, need to change either world or agent class
    maxX = world.shape[0]
    minY = world.shape[1] / 2

    all_paths = {}
    avalaible_pos = set([(min(x + 1, maxX), y), (max(x - 1, 0), y), (x, min(world.shape[1], y + 1)), (x, max(minY, y - 1))])
   
    potential_goal = max(p_goals_given_moves[knower.pos])

    for next_pos in avalaible_pos:
        start = Node(pos=next_pos, key_id=self.key.identifier, used_key_ids=[], parent=start_node)
        goal = Node(Pos(potential_goal), key_id=self.key.identifier, used_key_ids=[], parent=None)

        get_neighbors = make_get_neighbors(world)
        path = find_path(start, goal, get_neighbors)
        all_paths[next_pos] = path
   
    p_nexts_given_goal_knower = get_p_next_given_goal(all_paths)

    p_nexts_watcher = {pos:0 for pos in avalaible_pos}
    for next in p_nexts_given_goal_knower:
        p_nexts_watcher[next] += p_nexts_given_goal_knower[next] * p_knower_nexts[next]
   
    return p_nexts_watcher


def predict_knower_move(world, knower):
    pass

def choose_move_given_knower_beliefs(knower_beliefs):
    pass

def init_knower_beliefs(world, knower):
    pass

def infer_knower_beliefs(world, knower, move_predictions):
    pass



# in agents (watcher class)
# self.knower_beliefs = init_knower_beliefs
# self.move_predictions = predict_knower_move(world, knower)
# choose_move_given_knower_beliefs(self.knower_beliefs)
# --- wait --- (knower makes move)
# (if self.move_predictions[knower.pos] < threshold)
# self.knower_beliefs = infer_knower_beliefs(world, knower, self.move_predictions) <- old predictions for where they would go
