from states import *
import tensorflow as tf
tf.set_random_seed(2016)

# Network Hyperparameters
L = 50 # neurons in first layer
M = 30 # neurons in second layer
N = 15 # neurons in third layer

# Training hyperparameters
training_cycles = 15000
memory_capacity = 10000 # samples in experience memory
memory_initial = 1000
actions_per_optimization = 20
minibatch_size = 32
target_network_decay = 0.99
learning_rate = 0.001

# Q learning hyperparameters
gamma = 0.99
epsilon_0 = 1.0
epsilon_1 = 0.01
epsilon_wait = 1000
epsilon_ramp = 10000
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

# Link optimization of q network with update of target network
with tf.control_dependencies([optimize]):
    training_op = tf.group(update_averages)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def train():
    global epsilon
    episodes = 0
    # TODO encapsulate these variables in an Episode class
    s = generate_initial_state()
    max_cumulative_reward = s.max_cumulative_reward()
    cumulative_reward = 0.0
    steps = 0
    for i in range(0, training_cycles):
        # Generate new state/action pairs according to the current Q function
        for k in range(0, actions_per_optimization):
            if random.random() < epsilon:
                a = random.randint(0, 3)
            else:
                a = sess.run(best_action, feed_dict = {X_q: np.array([s.as_numpy_array()])})[0]
            memory.store(s, a)
            r, s1 = s.perform_action(a)
            cumulative_reward += r
            steps += 1
            if not s1.terminal():
                s = s1
            else:
                reward_fraction = cumulative_reward/max_cumulative_reward if max_cumulative_reward > 0.0 else 1.0
                print("Episode " + str(episodes) + " terminated with score %.2f in " % reward_fraction + str(steps) + " steps")
                episodes += 1
                s = generate_initial_state()
                max_cumulative_reward = s.max_cumulative_reward()
                cumulative_reward = 0.0
                steps = 0

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
        sess.run(training_op, {
            X_q: initial_states,
            X_t: final_states,
            A: actions,
            R: rewards,
            D: discounts,
            #lr: learning_rate
        })

        # update epsilon
        if epsilon_wait < i <= epsilon_wait + epsilon_ramp:
            ramp_completion = float(i - epsilon_wait) / epsilon_ramp
            epsilon = (1 - ramp_completion) * epsilon_0 + ramp_completion * epsilon_1

if __name__ == "__main__":
    train()

