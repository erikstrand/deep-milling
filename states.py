import sys
import random
import numpy as np
random.seed(2016)


# Global constants
# W and H determine the width and height of our grid.
# The stock piece initially covers all (i, j) such that 1 <= i < W - 1 and 1 <= j < H - 2.
# Thus W and H should both be larger than three, so the stock piece isn't trivially small.
W = 6
H = 6
S = 4 # number of states (R, U, L, D)


# States are (conceptually) immutable.
class State:
    def __init__(self, goal, material, pos, terminal):
        # goal and material are W by H matrices.
        # 0.0 represents empty space, and 1.0 represents material.
        # One element of self.material is 2.0, namely self.pos.
        self.goal = goal
        self.material = material
        self.pos = pos
        # 0 => non-terminal, 1 => failure, 2 => success
        self.terminal_code = terminal

    def __hash__(self):
        return hash(self.material.data)

    def perform_action(self, action):
        # 0 => right
        # 1 => up
        # 2 => left
        # 3 => down
        if action == 0:
            if self.pos[0] < W - 1:
                return self._transition_to_new_state((self.pos[0] + 1, self.pos[1]))
        elif action == 1:
            if self.pos[1] < H - 1:
                return self._transition_to_new_state((self.pos[0], self.pos[1] + 1))
        elif action == 2:
            if self.pos[0] > 0:
                return self._transition_to_new_state((self.pos[0] - 1, self.pos[1]))
        elif action == 3:
            if self.pos[1] > 0:
                return self._transition_to_new_state((self.pos[0], self.pos[1] - 1))

        # If we haven't returned already it means we tried to go out of bounds (or used an invalid
        # action token). This leads to a terminal state.
        new_state = State(self.goal, self.material, self.pos, 1)
        return 0.0, new_state

    def as_numpy_array(self):
        return np.array([self.goal, self.material])

    def terminal(self):
        return self.terminal_code != 0

    def successful(self):
        return self.terminal_code == 2

    def discount_factor(self):
        if self.terminal():
            return 0.
        else:
            return 1.

    def max_cumulative_reward(self):
        if self.terminal():
            return 0.0
        # If we're not in a terminal state, self.goal[i, j] == 1.0 => self.material[i, j] == 1.0.
        # Thus self.material - self.goal only contains positive entries. The sum of these entries
        # is the number of blocks that remain to be milled plus 2.0.
        return np.sum(self.material - self.goal) - 2.0

    def _transition_to_new_state(self, new_pos):
        material = np.copy(self.material)
        material[self.pos] = 0.0
        material[new_pos] = 0.0
        r = 0.0
        terminal_code = 0
        if self.goal[new_pos] == 1.0:
            terminal_code = 1
        else:
            if self.material[new_pos] == 1.0:
                r = 1.0
            # This sum gives the number of remaining blocks that should be milled.
            if np.sum(material - self.goal) == 0:
                terminal_code = 2
        # This is how we convey the current position to the network.
        material[new_pos] = 2.0
        material.flags.writeable = False
        return r, State(self.goal, material, new_pos, terminal_code)


class Episode:
    def __init__(self):
        self.current_state = generate_initial_state()
        self.history = []
        self.past_transitions = {}
        self.cumulative_reward = 0.0
        self.max_cumulative_reward = self.current_state.max_cumulative_reward()

    def perform_action(self, a):
        # Record the action
        self.history.append(a)
        h = hash(self.current_state)
        if h not in self.past_transitions:
            self.past_transitions[h] = set()
        self.past_transitions[h].add(a)

        # Perform the action
        r, s = self.current_state.perform_action(a)
        self.current_state = s
        self.cumulative_reward += r

        return r, s

    def in_repeated_state(self):
        return hash(self.current_state) in self.past_transitions

    def unexplored_actions(self):
        h = hash(self.current_state)
        if h not in self.past_transitions:
            return set(range(0, 4))
        #return set(range(0, 4)) - self.past_transitions[h]
        explored = self.past_transitions[h]
        return set(i for i in range(0, 4) if i not in explored)

    def in_terminal_state(self):
        return self.current_state.terminal()

    def actions(self):
        return len(self.history)


# Given a point within the stock region, yields all neighboring points in the stock region.
def get_neighbors(p):
    if p[0] < W - 2:
        yield (p[0] + 1, p[1])
    if p[1] < H - 2:
        yield (p[0], p[1] + 1)
    if p[0] > 1:
        yield (p[0] - 1, p[1])
    if p[1] > 1:
        yield (p[0], p[1] - 1)


# Given a point, returns half its neighbors: those we have already encountered if we iterate from
# 0 to W and from 0 to H.
def get_half_neighbors(p):
    if p[0] > 0:
        yield (p[0] - 1, p[1])
    if p[1] > 0:
        yield (p[0], p[1] - 1)


def on_boundary(p):
    if p[0] == 0 or p[0] == W - 1 or p[1] == 0 or p[1] == H - 1:
        return True
    return False


# Arguments: p is the "bottom left" corner, w and h are the width and height of the rectangle.
def random_point_on_rectangle(p, w, h):
    # It's more convenient to work with max indices than widths.
    w -= 1
    h -= 1
    i = random.randint(0, 2 * (w + h))
    if i < w:
        return (p[0] + i, p[1])
    elif i < w + h:
        return (p[0] + w, p[1] + i - w)
    elif i < 2 * w + h:
        return (p[0] + i - (w + h), p[1] + h)
    else:
        return (p[0], p[1] + i - (2 * w + h))


def generate_goal():
    goal = np.zeros((W, H), dtype=np.float32)

    # Choose a seed point for the shape.
    x = random.randint(1, W - 2)
    y = random.randint(1, H - 2)
    goal[x, y] = 1.0
    boundary = set()
    boundary.update(get_neighbors((x, y)))

    # Expand the shape by adding additional points on the boundary.
    n = random.randint(0, 2 * W * H / 3)
    for _ in range(1, n):
        p = random.choice(list(boundary))
        goal[p] = 1.0
        boundary.update(neighbor for neighbor in get_neighbors(p) if goal[neighbor] == 0.0)

    # Partition the empty spaces into connected components.
    component_sets = []
    point_to_component = dict()
    for i in range(0, W):
        for j in range(0, H):
            if goal[i, j] == 0.0:
                neighbors = [nbr for nbr in get_half_neighbors((i, j)) if goal[nbr] == 0.0]
                if len(neighbors):
                    component_1 = point_to_component[neighbors[0]]
                    point_to_component[(i, j)] = component_1
                    component_1.add((i, j))
                    if len(neighbors) == 2:
                        component_2 = point_to_component[neighbors[1]]
                        if component_1 is not component_2:
                            for p in component_2:
                                point_to_component[p] = component_1
                            component_1.update(component_2)
                            component_sets.remove(component_2)
                else:
                    new_component = {(i, j)}
                    point_to_component[(i, j)] = new_component
                    component_sets.append(new_component)

    # Fill in inaccessible regions
    for component in component_sets:
        if not any(on_boundary(p) for p in component):
            for p in component:
                goal[p] = 1.0

    # If we filled in the entire millable area, we need to remove a block of material.
    if np.sum(goal) == (W - 2) * (H - 2):
        # Accessible blocks are in a rectangle with corners (1, 1), (W - 2, 1), (W - 2, H - 2),
        # and (1, H - 2).
        goal[random_point_on_rectangle((1, 1), W - 2, H - 2)] = 0.0

    # Finally done
    goal.flags.writeable = False
    return goal


def generate_stock(pos = (0, 0)):
    material = np.ones((W, H), dtype=np.float32)
    for i in range(0, W):
        material[i, 0] = 0.0
        material[i, H - 1] = 0.0
    for j in range(1, H - 1):
        material[0, j] = 0.0
        material[W - 1, j] = 0.0
    material[pos] = 2.0
    material.flags.writeable = False
    return material


def generate_initial_state():
    start = random_point_on_rectangle((0, 0), W, H)
    return State(generate_goal(), generate_stock(start), start, False)


def display_ascii(grid):
    print(' ' + W*'-')
    for j in reversed(range(0, H)):
        sys.stdout.write('|')
        for i in reversed(range(0, W)):
            if grid[i, j] == 2.0:
                sys.stdout.write('*')
            elif grid[i, j] == 1.0:
                sys.stdout.write('#')
            elif grid[i, j] == 0.0:
                sys.stdout.write(' ')
            else:
                sys.stdout.write('?')
        sys.stdout.write('|\n')
    print(' ' + W*'-')


if __name__ == "__main__":
    for _ in range(0, 100):
        grid = generate_goal()
        display_ascii(grid)
        print('\n')

