from states import *
import tensorflow as tf
tf.set_random_seed(2016)

# Notes
# - What do I really want to display/record?
#   - Definitely need a stats class
#   - probably want a separate test cycle at the end of each epoch (run with no random component but where repetition is failure)
#   - max_a(q(s, a)) / s.max_cumulative_score() i.e. how many of the remaining points we think we can get
# - Is penalization of non-milling moves necessary?
#   - it looks like not storing repetitive behaviors is enough of a fix
# - How to best set up intuitive evaluation framework
#   - I want to save my parameters, and feed the network test cases
# - Why did the recent parameter changes make things worse. Of course while epsilon is held high
#   it performs poorly, but this seems to have persisted even after letting epsilon back down again.
#   May need to make epsilon depend on what percentage of the blocks we've milled so far, so that
#   the exploit/explore tradeoff depends on if we're near the opening or the endgame.

# Network Hyperparameters
L = 50 # neurons in first layer
M = 30 # neurons in second layer
N = 15 # neurons in third layer

# Training hyperparameters
training_cycles = 100000
print_interval = 1000
memory_capacity = 50000 # samples in experience memory
memory_initial = 10000
actions_per_optimization = 20
minibatch_size = 64
target_network_decay = 0.99
learning_rate = 0.001

# Q learning hyperparameters
gamma = 0.99
epsilon_0 = 1.0
epsilon_1 = 0.1
epsilon_wait = 10000
epsilon_ramp = 80000
epsilon = epsilon_0

# Memory Bank
class MemoryBank:
    def __init__(self, capacity):
        self.capacity = capacity
        self.state_action_pairs = []
        self.index = 0

    def store(self, state, action):
        if len(self) == self.capacity:
            self.state_action_pairs[self.index] = (state, action)
            self.index = (self.index + 1) % self.capacity
        else:
            self.state_action_pairs.append((state, action))

    def sample(self, size):
        return random.sample(self.state_action_pairs, size)

    def __len__(self):
        return len(self.state_action_pairs)

# Create the memory bank and store some random state/action pairs
memory = MemoryBank(memory_capacity)
while len(memory.state_action_pairs) < memory_initial:
    s = generate_initial_state()
    while not s.terminal():
        a = random.randint(0, 3)
        r, s1 = s.perform_action(a)
        memory.store(s, a)
        s = s1

# Model parameters for Q function
W1_q = tf.Variable(tf.truncated_normal([W * H * 2, L], stddev=0.1), name="W1")
B1_q = tf.Variable(tf.ones([L])/10, name="B1")
W2_q = tf.Variable(tf.truncated_normal([L, M], stddev=0.1), name="W2")
B2_q = tf.Variable(tf.ones([M])/10, name="B2")
W3_q = tf.Variable(tf.truncated_normal([M, N], stddev=0.1), name="W3")
B3_q = tf.Variable(tf.ones([N])/10, name="B3")
W4_q = tf.Variable(tf.truncated_normal([N, S], stddev=0.1), name="W4")
B4_q = tf.Variable(tf.ones([S])/10, name="B4")

# The target network uses slow moving averages of the Q network.
ema = tf.train.ExponentialMovingAverage(decay=target_network_decay)
update_averages = ema.apply([W1_q, B1_q, W2_q, B2_q, W3_q, B3_q, W4_q, B4_q])
W1_t = ema.average(W1_q)
B1_t = ema.average(B1_q)
W2_t = ema.average(W2_q)
B2_t = ema.average(B2_q)
W3_t = ema.average(W3_q)
B3_t = ema.average(B3_q)
W4_t = ema.average(W4_q)
B4_t = ema.average(B4_q)

# Placeholders
X_q = tf.placeholder(tf.float32, [None, 2, W, H])
X_t = tf.placeholder(tf.float32, [minibatch_size, 2, W, H])
A = tf.placeholder(tf.int32, [minibatch_size])
R = tf.placeholder(tf.float32, [minibatch_size])
D = tf.placeholder(tf.float32, [minibatch_size])

# The model
XX_q = tf.reshape(X_q, [-1, 2 * W * H])
Y1_q = tf.nn.relu(tf.matmul(XX_q, W1_q) + B1_q)
Y2_q = tf.nn.relu(tf.matmul(Y1_q, W2_q) + B2_q)
Y3_q = tf.nn.relu(tf.matmul(Y2_q, W3_q) + B3_q)
Y4_q = tf.nn.relu(tf.matmul(Y3_q, W4_q) + B4_q)

XX_t = tf.reshape(X_t, [-1, 2 * W * H])
Y1_t = tf.nn.relu(tf.matmul(XX_t, W1_t) + B1_t)
Y2_t = tf.nn.relu(tf.matmul(Y1_t, W2_t) + B2_t)
Y3_t = tf.nn.relu(tf.matmul(Y2_t, W3_t) + B3_t)
Y4_t = tf.nn.relu(tf.matmul(Y3_t, W4_t) + B4_t)

# Used to select actions
best_action = tf.argmax(Y4_q, 1)

# Loss
# this calculation of reward_q does not work because gather_nd is not yet differentiable
#AA = tf.transpose(tf.pack([tf.range(minibatch_size), A]))
#reward_q = tf.gather_nd(Y4_q, AA)
reward_q = tf.reduce_sum(tf.one_hot(A, S) * Y4_q, 1)
reward_t = tf.add(R, tf.mul(D, tf.reduce_max(Y4_t, 1)))
expected_squared_error = tf.reduce_mean(tf.square(tf.sub(reward_t, reward_q)))
optimize = tf.train.AdamOptimizer(learning_rate).minimize(expected_squared_error)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def train():
    global epsilon
    episodes = 0
    total_transitions = 0
    repeated_transitions = 0
    max_reward = []
    got_reward = []
    steps = []
    episode = Episode()
    for i in range(1, training_cycles + 1):
        # Generate new state/action pairs according to the current Q function
        new_transitions = 0
        while new_transitions < actions_per_optimization:
            s = episode.current_state
            unexplored_actions = episode.unexplored_actions()
            if len(unexplored_actions) == 0:
                a = random.randint(0, 3)
            elif random.random() < epsilon:
                a = random.sample(unexplored_actions, 1)[0]
            else:
                a = sess.run(best_action, feed_dict = {X_q: np.array([s.as_numpy_array()])})[0]
                if a not in unexplored_actions:
                    a = random.sample(unexplored_actions, 1)[0]
            if a in unexplored_actions:
                new_transitions += 1
            else:
                repeated_transitions += 1
            total_transitions += 1
            memory.store(s, a)
            r, s1 = episode.perform_action(a)
            if episode.in_terminal_state():
                episodes += 1
                max_reward.append(episode.max_cumulative_reward)
                got_reward.append(episode.cumulative_reward)
                steps.append(episode.steps)
                episode = Episode()

        # Get a sample
        sample = memory.sample(minibatch_size)
        results = []
        for s, a in sample:
            results.append(s.perform_action(a))
        initial_states = [s.as_numpy_array() for s, a in sample]
        final_states = [s.as_numpy_array() for r, s in results]
        actions = [a for s, a in sample]
        rewards = [r for r, s in results]
        discounts = gamma * np.array([s.discount_factor() for r, s in results])

        # Update the q and target networks.
        sess.run(optimize, {
            X_q: initial_states,
            X_t: final_states,
            A: actions,
            R: rewards,
            D: discounts,
            #lr: learning_rate
        })
        sess.run(update_averages)

        # update epsilon
        if epsilon_wait < i <= epsilon_wait + epsilon_ramp:
            ramp_completion = float(i - epsilon_wait) / epsilon_ramp
            epsilon = (1 - ramp_completion) * epsilon_0 + ramp_completion * epsilon_1

        # Print an update if applicable
        if i % print_interval == 0:
            print("Epsilon = " + str(epsilon))
            print("Completed " + str(episodes) + " episodes with " + str(total_transitions) + " total transitions (" + str(repeated_transitions) + " repeated)")
            all_max_rewards = sum(max_reward)
            all_got_rewards = sum(got_reward)
            all_steps = sum(steps)
            print("Average reward fraction: " + str(float(all_got_rewards)/all_max_rewards) + " (of " + str(all_max_rewards) + " possible rewards)")
            print("Average steps to completion: " + str(float(all_steps)/episodes))
            episodes = 0
            total_transitions = 0
            stored_transitions = 0
            repeated_transitions = 0
            max_reward = []
            got_reward = []
            steps = []


if __name__ == "__main__":
    train()

