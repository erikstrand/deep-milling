from states import *
import tensorflow as tf
tf.set_random_seed(2016)

# Network Hyperparameters
L = 50 # neurons in first layer
M = 30 # neurons in second layer
N = 15 # neurons in third layer

# Training hyperparameters
training_cycles = 100
memory_capacity = 5000 # samples in experience memory
memory_initial = 100
actions_per_sgd = 10
minibatch_size = 32
target_speed = 0.01

# Q learning hyperparameters
gamma = 0.99

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

# Model parameters for target function
W1_t = tf.Variable(tf.truncated_normal([W * H * 2, L], stddev=0.1))
B1_t = tf.Variable(tf.ones([L])/10)
W2_t = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2_t = tf.Variable(tf.ones([M])/10)
W3_t = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3_t = tf.Variable(tf.ones([N])/10)
W4_t = tf.Variable(tf.truncated_normal([N, S], stddev=0.1))
B4_t = tf.Variable(tf.ones([S])/10)

# Placeholders
X_q = tf.placeholder(tf.float32, [None, 2, W, H])
X_t = tf.placeholder(tf.float32, [minibatch_size, 2, W, H])
A = tf.placeholder(tf.int32, [minibatch_size])
R = tf.placeholder(tf.float32, [minibatch_size])
D = tf.placeholder(tf.float32, [minibatch_size])

# Learning rate and random action rate
lr = tf.placeholder(tf.float32)
epsilon = 0.1

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
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(expected_squared_error)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def train():
    episodes = 0
    s = generate_initial_state()
    max_cumulative_reward = s.max_cumulative_reward()
    cumulative_reward = 0.0
    steps = 0
    for i in range(0, training_cycles):
        # Generate new state/action pairs according to the current Q function
        for k in range(0, actions_per_sgd):
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

        # Calculate s1 for each s.
        # Shove initial states into X1. Shove final states into X2. Shove rewards into R.
        sess.run(train_step, {
            X_q: initial_states,
            X_t: final_states,
            A: actions,
            R: rewards,
            D: discounts,
            #lr: learning_rate
        })

        # every so many iterations, update target network (W_t = 0.99 * W_t + 0.01 * W_q).

if __name__ == "__main__":
    train()

