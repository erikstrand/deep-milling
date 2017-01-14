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
A = 4


# States are (conceptually) immutable.
class State:
    def __init__(self, stock, pos):
        # stock is a W by H NumPy array.
        # 0.0 represents empty space, and 1.0 represents material.
        # self.stock[self.pos] == 2.0 to indicate the location of the endmill.
        self.stock = stock
        self.pos = pos

    def __eq__(self, other):
        return (self.stock == other.stock).all()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.stock.data)

    # Takes an action, and returns the successor State and a status code.
    # Action codes:
    #   0 => right
    #   1 => up
    #   2 => left
    #   3 => down
    # Status codes:
    #   0 means we moved to an empty block
    #   1 means we moved a block we want to mill
    #   -1 means we moved onto a part block (i.e. gouged the part)
    #   -2 means we went out of bounds
    #   -3 means we tried to perform an invalid action
    def step(self, part, action):
        if action == 0:
            if self.pos[0] < W - 1:
                return self._successor(part, (self.pos[0] + 1, self.pos[1]))
        elif action == 1:
            if self.pos[1] < H - 1:
                return self._successor(part, (self.pos[0], self.pos[1] + 1))
        elif action == 2:
            if self.pos[0] > 0:
                return self._successor(part, (self.pos[0] - 1, self.pos[1]))
        elif action == 3:
            if self.pos[1] > 0:
                return self._successor(part, (self.pos[0], self.pos[1] - 1))
        else:
            return State(self.stock, self.pos), -3

        # If we haven't returned already it means we tried to go out of bounds.
        return State(self.stock, self.pos), -2

    def _successor(self, part, new_pos):
        stock = np.copy(self.stock)
        stock[self.pos] = 0.0
        stock[new_pos] = 0.0
        status = None
        if part[new_pos] == 1.0:
            status = -1
        else:
            if self.stock[new_pos] == 1.0:
                status = 1
            else:
                status = 0
        # This is indicates the current position.
        stock[new_pos] = 2.0
        stock.flags.writeable = False
        return State(stock, new_pos), status


class StepInfo:
    def __init__(self, status, deja_vu):
        # A status code as returned by State.step
        self.status = status
        # True if we've been in this state before
        self.deja_vu = deja_vu


class Environment:
    def __init__(self):
        # Note: the next two properties aren't yet used.
        self.action_space = [0, 1, 2, 3] # right, up, left, down
        self.observation_space = None # will be an OpenAI Box
        # Maps a status code from State.step to a reward
        self.reward_map = {
            0: 0.0,
            1: 1.0,
            -1: 0.0,
            -2: 0.0,
            -3: 0.0
        }
        self.done_map = {
            0: False,
            1: False,
            -1: True,
            -2: True,
            -3: True
        }

    def reset(self):
        start = random_point_on_rectangle((0, 0), W, H)
        self.part = generate_part()
        self.state = generate_stock(start)
        self.history = []
        self.transitions = {}
        return np.array([self.part, self.state.stock])

    # Returns an observation (NumPy array), a reward (float), done (bool), and a StepInfo object
    def step(self, action):
        # Record this state and action
        self.history.append(self.state)
        if self.state not in self.transitions:
            self.transitions[self.state] = set()
        self.transitions[self.state].add(action)

        # Perform the action
        self.state, status = self.state.step(self.part, action)
        deja_vu = self.state in self.transitions
        info = StepInfo(status, deja_vu)
        return np.array([self.part, self.state.stock]), self.reward_map[status], self.done_map[status], info

    def remaining_stock_blocks(self):
        # If we're not done, self.part[i, j] == 1.0 => self.state.stock[i, j] == 1.0.
        # Thus self.state.stock - self.part only contains positive entries. The sum of these entries
        # is the number of blocks that must still be milled, plus 2.0 from the endmill position.
        return np.sum(self.state.stock - self.part) - 2.0


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


def generate_part():
    part = np.zeros((W, H), dtype=np.float32)

    # Choose a seed point for the shape.
    x = random.randint(1, W - 2)
    y = random.randint(1, H - 2)
    part[x, y] = 1.0
    boundary = set()
    boundary.update(get_neighbors((x, y)))

    # Expand the shape by adding additional points on the boundary.
    n = random.randint(0, 2 * W * H / 3)
    for _ in range(1, n):
        p = random.choice(list(boundary))
        part[p] = 1.0
        boundary.update(neighbor for neighbor in get_neighbors(p) if part[neighbor] == 0.0)

    # Partition the empty spaces into connected components.
    component_sets = []
    point_to_component = dict()
    for i in range(0, W):
        for j in range(0, H):
            if part[i, j] == 0.0:
                neighbors = [nbr for nbr in get_half_neighbors((i, j)) if part[nbr] == 0.0]
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
                part[p] = 1.0

    # If we filled in the entire millable area, we need to remove a block of material.
    if np.sum(part) == (W - 2) * (H - 2):
        # Accessible blocks are in a rectangle with corners (1, 1), (W - 2, 1), (W - 2, H - 2),
        # and (1, H - 2).
        part[random_point_on_rectangle((1, 1), W - 2, H - 2)] = 0.0

    # Finally done
    part.flags.writeable = False
    return part


def generate_stock(pos = (0, 0)):
    stock = np.ones((W, H), dtype=np.float32)
    for i in range(0, W):
        stock[i, 0] = 0.0
        stock[i, H - 1] = 0.0
    for j in range(1, H - 1):
        stock[0, j] = 0.0
        stock[W - 1, j] = 0.0
    stock[pos] = 2.0
    stock.flags.writeable = False
    return State(stock, pos)


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
        grid = generate_part()
        display_ascii(grid)
        print('\n')

