import math
import numpy as np
from numba import njit


class RootNode:
    def __init__(self):
        self.parent = None
        self.visits = 0
        self.children = None


class Node(RootNode):
    def __init__(self, idx, parent):
        self.idx = idx
        self.parent = parent
        self.children = None

    @property
    def visits(self):
        return self.parent.children_visits[self.idx]

    @visits.setter
    def visits(self, x):
        self.parent.children_visits[self.idx] = x

    @property
    def action(self):
        return self.parent.children_actions[self.idx]

    @property
    def value(self):
        return self.parent.children_values[self.idx]

    @value.setter
    def value(self, x):
        self.parent.children_values[self.idx] = x


@njit(fastmath=True, parallel=True)
def get_ucb_scores_jitted(children_values, children_priors, visits, children_visits, c_puct):
    return children_values + c_puct * children_priors * math.sqrt(visits) / (children_visits + 1)


def get_ucb_scores(node, c_puct):
    return get_ucb_scores_jitted(
        node.children_values, node.children_priors,
        node.visits, node.children_visits, c_puct
    )


def select(root, game, c_puct):
    current = root
    while current.children:
        ucb_scores = get_ucb_scores(current, c_puct)
        ucb_scores[current.children_visits == 0] = np.inf
        current = current.children[np.argmax(ucb_scores)]
        game.step(current.action)
    return current


def expand(leaf, children_actions, children_priors):
    leaf.children = [Node(idx, leaf) for idx, _ in enumerate(children_actions)]
    leaf.children_actions = children_actions
    leaf.children_priors = children_priors
    leaf.children_values = np.zeros_like(leaf.children_priors)
    leaf.children_visits = np.zeros_like(leaf.children_priors)


def backpropagate(leaf, game, result):
    current = leaf
    while current.parent:
        result = game.swap_result(result)
        current.value = (current.value * current.visits + result) / (current.visits + 1)
        current.visits += 1
        current = current.parent
        game.undo_last_action()
    current.visits += 1


def search(game, value_fn, policy_fn, iterations, c_puct=1.0, dirichlet_alpha=None):
    root = RootNode()
    children_actions = np.array(game.get_legal_actions())
    if len(children_actions) == 0:
        return root
    children_priors = policy_fn(game)[children_actions]
    if dirichlet_alpha:
        children_priors = 0.75 * children_priors + 0.25 * np.random.default_rng().dirichlet(
            dirichlet_alpha * np.ones_like(children_priors)
        )
    expand(root, children_actions, children_priors)

    for _ in range(iterations):
        leaf = select(root, game, c_puct)
        result = game.get_first_person_result()
        if result is None:
            children_actions = np.array(game.get_legal_actions())
            if len(children_actions) == 0:
                result = game.get_first_person_result()
                if result is None:
                    result = 0.0
            else:
                children_priors = policy_fn(game)[children_actions]
                expand(leaf, children_actions, children_priors)
                result = value_fn(game)
        backpropagate(leaf, game, result)
    return root


def play(game, agent, search_iterations, c_puct=1.0, dirichlet_alpha=None):
    root = search(
        game, agent.value_fn, agent.policy_fn, search_iterations,
        c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
    )
    if root.children is None:
        return None
    return root.children_actions[np.argmax(root.children_visits)]


def pit(game, agent1, agent2, agent1_play_kwargs, agent2_play_kwargs):
    current_agent, other_agent = agent1, agent2
    current_kwargs, other_kwargs = agent1_play_kwargs, agent2_play_kwargs
    while (result := game.get_result()) is None:
        action = play(game, current_agent, **current_kwargs)
        if action is None:
            break
        game.step(action)
        current_agent, other_agent = other_agent, current_agent
        current_kwargs, other_kwargs = other_kwargs, current_kwargs
    return game.get_result()
