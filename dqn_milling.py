from states import *
import tensorflow as tf
tf.set_random_seed(2016)

# Notes
# - What do I really want to display/record?
#   - want a separate test cycle at the end of each epoch (run with no random component but where repetition is failure)
#   - max_a(q(s, a)) / s.max_cumulative_score() i.e. how many of the remaining points we think we can get
# - Is penalization of non-milling moves necessary?
#   - it looks like not storing repetitive behaviors is enough of a fix
# - How to best set up intuitive evaluation framework
#   - I want to save my parameters, and feed the network test cases

# Network Hyperparameters
L = 50 # neurons in first layer
M = 30 # neurons in second layer
N = 15 # neurons in third layer

# Training hyperparameters
training_cycles = 100000
print_interval = 1000
memory_capacity = 50000 # samples in experience memory
memory_initial = 15000
actions_per_update = 20
minibatch_size = 64
target_network_decay = 0.99
learning_rate = 0.001

# Q learning hyperparameters
gamma = 0.99
epsilon_0 = 1.0
epsilon_1 = 0.1
epsilon_ramp = 90000.0
epsilon = epsilon_0

# Evaluation hyperparameters
evaluation_episodes = 500

class TrainingAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.episode = Episode()
        self.reset_stats()

    def reset_stats(self):
        self.initial_actions = self.episode.actions()
        self.initial_reward = self.episode.cumulative_reward
        self.q_actions = 0
        self.random_actions = 0
        self.repeat_attempts = 0
        self.repeats = 0

    def advance(self, epsilon, evaluator):
        s = self.episode.current_state
        unexplored_actions = self.episode.unexplored_actions()
        new = True
        if len(unexplored_actions) == 0:
            new = False
            self.repeats += 1
            self.random_actions += 1
            a = random.randint(0, 3)
            r, s1 = self.episode.perform_action(a)
        elif random.random() < epsilon:
            self.random_actions += 1
            a = random.sample(unexplored_actions, 1)[0]
        else:
            a = evaluator(s.as_numpy_array())
            if a in unexplored_actions:
                self.q_actions += 1
            else:
                self.repeat_attempts += 1
                a = random.sample(unexplored_actions, 1)[0]
        self.episode.perform_action(a)
        return new, s, a, self.episode.in_terminal_state()

    def actions(self):
        return self.episode.actions() - self.initial_actions

    def cumulative_reward(self):
        return self.episode.cumulative_reward - self.initial_reward

    def max_cumulative_reward(self):
        return self.episode.max_cumulative_reward - self.initial_reward


class EvaluationAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.episode = Episode()
        self.q_actions = 0
        self.random_actions = 0
        self.repeat_attempts = 0
        self.repeats = 0

    def advance(self, evaluator):
        s = self.episode.current_state
        a = evaluator(s.as_numpy_array())
        self.episode.perform_action(a)
        terminal = self.episode.in_terminal_state() or self.episode.in_repeated_state()
        return s, a, terminal

    def actions(self):
        return self.episode.actions()

    def cumulative_reward(self):
        return self.episode.cumulative_reward

    def max_cumulative_reward(self):
        return self.episode.max_cumulative_reward


class StatAggregator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.episodes = 0
        self.actions = 0
        self.q_actions = 0
        self.random_actions = 0
        self.repeat_attempts = 0
        self.repeats = 0
        self.cumulative_reward = 0.0
        self.max_cumulative_reward = 0.0

    def update(self, session):
        self.episodes += 1
        self.actions += session.actions()
        self.q_actions += session.q_actions
        self.random_actions += session.random_actions
        self.repeat_attempts += session.repeat_attempts
        self.repeats += session.repeats
        self.cumulative_reward += session.cumulative_reward()
        self.max_cumulative_reward += session.max_cumulative_reward()


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
W4_q = tf.Variable(tf.truncated_normal([N, A], stddev=0.1), name="W4")
B4_q = tf.Variable(tf.ones([A])/10, name="B4")

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
action_indices = tf.placeholder(tf.int32, [minibatch_size])
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
reward_q = tf.reduce_sum(tf.one_hot(action_indices, A) * Y4_q, 1)
reward_t = tf.add(R, tf.mul(D, tf.reduce_max(Y4_t, 1)))
expected_squared_error = tf.reduce_mean(tf.square(tf.sub(reward_t, reward_q)))
optimize = tf.train.AdamOptimizer(learning_rate).minimize(expected_squared_error)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def train():
    global epsilon
    train_agent = TrainingAgent()
    eval_agent = EvaluationAgent()
    train_stats = StatAggregator()
    eval_stats = StatAggregator()
    evaluator = lambda s: sess.run(best_action, feed_dict = {X_q: np.array([s])})[0]
    for i in range(1, training_cycles + 1):
        # Generate new state/action pairs according to the current Q function
        new_transitions = 0
        while new_transitions < actions_per_update:
            new, s, a, terminal = train_agent.advance(epsilon, evaluator)
            if new:
                new_transitions += 1
                memory.store(s, a)
            if terminal:
                train_stats.update(train_agent)
                train_agent.reset()

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
            action_indices: actions,
            R: rewards,
            D: discounts,
            #lr: learning_rate
        })
        sess.run(update_averages)

        # update epsilon
        if i <= epsilon_ramp:
            ramp_completion = float(i) / epsilon_ramp
            epsilon = (1.0 - ramp_completion) * epsilon_0 + ramp_completion * epsilon_1

        # Print an update if applicable
        if i % print_interval == 0:
            for _ in range(0, evaluation_episodes):
                terminal = False
                while not terminal:
                    s, a, terminal = eval_agent.advance(evaluator)
                eval_stats.update(eval_agent)
                eval_agent.reset()

            print("================================================================================")
            print("Epsilon = " + str(epsilon))

            summary = "Completed " + str(train_stats.episodes) + " episodes with " + str(train_stats.actions) + " total transitions ("
            summary += str(train_stats.repeat_attempts) + " repeat attempts, "
            summary += str(train_stats.repeats) + " forced repeats, "
            summary += str(float(train_stats.random_actions)/train_stats.actions*100) + "% random actions)"
            print(summary)
            print("Average reward fraction: " + str(float(train_stats.cumulative_reward)/train_stats.max_cumulative_reward) + " (of " + str(train_stats.max_cumulative_reward) + " possible rewards)")
            print("Average steps to completion: " + str(float(train_stats.actions)/train_stats.episodes))
            train_stats.reset()

            summary = "Completed " + str(eval_stats.episodes) + " episodes with " + str(eval_stats.actions) + " total transitions ("
            summary += str(eval_stats.repeat_attempts) + " repeat attempts, "
            summary += str(eval_stats.repeats) + " forced repeats, "
            summary += str(float(eval_stats.random_actions)/eval_stats.actions*100) + "% random actions)"
            print(summary)
            print("Average reward fraction: " + str(float(eval_stats.cumulative_reward)/eval_stats.max_cumulative_reward) + " (of " + str(eval_stats.max_cumulative_reward) + " possible rewards)")
            print("Average steps to completion: " + str(float(eval_stats.actions)/eval_stats.episodes))
            eval_stats.reset()


if __name__ == "__main__":
    train()

