import numpy as np


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


def get_ucb_scores(node, c_puct):
    return (node.children_values
            + c_puct * node.children_priors * np.sqrt(node.visits)
            / (node.children_visits + 1))


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


def search(game, value_fn=None, policy_fn=None, iterations=64, c_puct=1.0,
           dirichlet_alpha=None, eval_fn=None):
    """MCTS search.

    Accepts either a combined ``eval_fn(game) -> (value, policy_np)`` (preferred,
    one forward pass) or separate ``value_fn`` / ``policy_fn`` (legacy, two passes).
    """
    root = RootNode()
    children_actions = np.array(game.get_legal_actions())
    if len(children_actions) == 0:
        return root

    if eval_fn is not None:
        _, policy = eval_fn(game)
        children_priors = policy[children_actions]
    else:
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
                if eval_fn is not None:
                    result, policy = eval_fn(game)
                    children_priors = policy[children_actions]
                else:
                    children_priors = policy_fn(game)[children_actions]
                    result = value_fn(game)
                expand(leaf, children_actions, children_priors)
        backpropagate(leaf, game, result)
    return root


def batched_search(games, batch_eval_fn, iterations, c_puct=1.0, dirichlet_alpha=None):
    """Run MCTS on multiple games simultaneously with batched neural network evaluation.

    Instead of evaluating one leaf at a time (batch=1), this collects leaves
    from all active games per iteration and evaluates them in a single batched
    forward pass on the GPU/MPS device.

    Args:
        games: list of N PylosGame instances
        batch_eval_fn: callable(obs_np) -> (values_np, policies_np)
            obs_np: (B, obs_dim) numpy array
            values_np: (B,) numpy array
            policies_np: (B, action_space) numpy array
        iterations: number of MCTS iterations per game
        c_puct: exploration constant
        dirichlet_alpha: if set, add Dirichlet noise to root priors

    Returns:
        list of N root nodes
    """
    N = len(games)
    roots = [RootNode() for _ in range(N)]

    # Determine which games have legal actions
    active = []
    active_actions = {}
    for i in range(N):
        actions = np.array(games[i].get_legal_actions())
        if len(actions) > 0:
            active.append(i)
            active_actions[i] = actions

    if not active:
        return roots

    # Batch evaluate initial root policies
    obs_batch = np.array([games[i].to_observation() for i in active])
    _, policies = batch_eval_fn(obs_batch)

    rng = np.random.default_rng()
    for j, i in enumerate(active):
        priors = policies[j][active_actions[i]]
        if dirichlet_alpha:
            noise = rng.dirichlet(dirichlet_alpha * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise
        expand(roots[i], active_actions[i], priors)

    # Main MCTS iterations
    for _ in range(iterations):
        leaves = {}
        need_eval = []
        terminal = {}

        # Phase 1: Select leaf for each active game
        for i in active:
            leaf = select(roots[i], games[i], c_puct)
            leaves[i] = leaf

            result = games[i].get_first_person_result()
            if result is not None:
                terminal[i] = result
            else:
                actions = np.array(games[i].get_legal_actions())
                if len(actions) == 0:
                    result = games[i].get_first_person_result()
                    terminal[i] = result if result is not None else 0.0
                else:
                    need_eval.append((i, actions))

        # Phase 2: Batch evaluate non-terminal leaves
        eval_results = {}
        if need_eval:
            obs_batch = np.array([games[i].to_observation() for i, _ in need_eval])
            values, policies = batch_eval_fn(obs_batch)

            for k, (i, actions) in enumerate(need_eval):
                priors = policies[k][actions]
                expand(leaves[i], actions, priors)
                eval_results[i] = values[k]

        # Phase 3: Backpropagate all
        for i in active:
            if i in terminal:
                result = terminal[i]
            elif i in eval_results:
                result = eval_results[i]
            else:
                result = 0.0
            backpropagate(leaves[i], games[i], result)

    return roots


def play(game, agent, search_iterations, c_puct=1.0, dirichlet_alpha=None):
    eval_fn = getattr(agent, 'eval_fn', None)
    if eval_fn is not None:
        root = search(game, iterations=search_iterations, c_puct=c_puct,
                       dirichlet_alpha=dirichlet_alpha, eval_fn=eval_fn)
    else:
        root = search(game, agent.value_fn, agent.policy_fn, search_iterations,
                       c_puct=c_puct, dirichlet_alpha=dirichlet_alpha)
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
