from states import *
import tensorflow as tf
tf.set_random_seed(2016)

# Training hyperparameters
training_episodes = 100000
memory_capacity = 100000 # transitions in experience memory
memory_initial = 25000
minibatch_size = 64
target_network_decay = 0.99
learning_rate = 0.001

# Q learning hyperparameters
gamma = 0.99
epsilon_0 = 1.0
epsilon_1 = 0.01
epsilon_ramp = 75000.0

# Evaluation hyperparameters
evaluation_episodes = 500


class MemoryBank:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.index = 0

    def store(self, *stuff):
        if len(self) == self.capacity:
            self.data[self.index] = stuff
            self.index = (self.index + 1) % self.capacity
        else:
            self.data.append(stuff)

    def sample(self, size):
        return random.sample(self.data, size)

    def __len__(self):
        return len(self.data)


def relu_layer_with_ema(scope_name, input1, input2, n_inputs, n_outputs, ema, linear=False):
    with tf.variable_scope(scope_name) as scope:
        weights = tf.get_variable("weights", (n_inputs, n_outputs), initializer=tf.truncated_normal_initializer(0.1))
        biases = tf.get_variable("biases", (n_outputs), initializer=tf.constant_initializer(0.01))
        ema_name = scope.name + "_ema"
        update_ema = ema.apply([weights, biases])
        weights_ema = ema.average(weights)
        biases_ema = ema.average(biases)
        if linear:
            output = tf.add(tf.matmul(input1, weights), biases, name=scope.name)
            output_ema = tf.add(tf.matmul(input2, weights_ema), biases_ema, name=ema_name)
        else:
            output = tf.nn.relu(tf.matmul(input1, weights) + biases, name=scope.name)
            output_ema = tf.nn.relu(tf.matmul(input2, weights_ema) + biases_ema, name=ema_name)
        tf.summary.histogram(scope_name + " histogram", output)
        tf.summary.histogram(ema_name + " histogram", output_ema)
        return output, output_ema, update_ema


class Model:
    def __init__(self):
        # Network Hyperparameters
        self.L = 50 # neurons in first layer
        self.M = 30 # neurons in second layer
        self.N = 15 # neurons in third layer

        # Placeholders
        self.X_q = tf.placeholder(tf.float32, [None, 2, W, H])
        self.X_t = tf.placeholder(tf.float32, [minibatch_size, 2, W, H])
        self.action = tf.placeholder(tf.int32, [minibatch_size])
        self.reward = tf.placeholder(tf.float32, [minibatch_size])
        self.not_done = tf.placeholder(tf.float32, [minibatch_size])
        
        self.score = tf.placeholder(tf.float32, [1])

        # Actual feed tensors
        XX_q = tf.reshape(self.X_q, [-1, 2 * W * H])
        XX_t = tf.reshape(self.X_t, [-1, 2 * W * H])

        # The model
        ema = tf.train.ExponentialMovingAverage(decay=target_network_decay)
        q1, t1, ema1 = relu_layer_with_ema("layer1", XX_q, XX_t, 2 * W * H, self.L, ema)
        q2, t2, ema2 = relu_layer_with_ema("layer2", q1, t1, self.L, self.M, ema)
        q3, t3, ema3 = relu_layer_with_ema("layer3", q2, t2, self.M, self.N, ema)
        q4, t4, ema4 = relu_layer_with_ema("layer4", q3, t3, self.N, A, ema, linear=True)

        # Update exponential moving averages (for the target network)
        self.update_emas = tf.group(ema1, ema2, ema3, ema4)

        # Used to select actions
        self.best_action = tf.argmax(q4, 1)

        # Expected rewards for the specified actions
        q_reward = tf.reduce_sum(tf.one_hot(self.action, A) * q4, 1)

        # Expected rewards for the specified actions and immediate rewards
        t_reward = tf.add(self.reward, tf.mul(gamma * self.not_done, tf.reduce_max(t4, 1)))

        # Loss
        expected_squared_error = tf.reduce_mean(tf.square(tf.sub(q_reward, t_reward)))
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(expected_squared_error)

        # Memory bank
        self.memory = MemoryBank(memory_capacity)

        # Summary stats
        tf.summary.scalar("expected reward (q)", tf.reduce_mean(q_reward))
        tf.summary.scalar("expected reward (t)", tf.reduce_mean(t_reward))
        tf.summary.scalar("expected_squared_error", expected_squared_error)
        self.summaries = tf.summary.merge_all()

    def train(self, sess):
        summary_writer = tf.summary.FileWriter("./train", sess.graph)

        # Seed memory bank
        env = Environment()
        while len(self.memory.data) < memory_initial:
            prior_obs = env.reset()
            done = False
            while not done:
                a = random.randint(0, 3)
                post_obs, reward, done, info = env.step(a)
                self.memory.store(prior_obs, a, reward, post_obs, done)
                prior_obs = post_obs

        epsilon = epsilon_0
        evaluator = lambda s: sess.run(self.best_action, feed_dict = {self.X_q: [s]})[0]
        for i in range(0, training_episodes):
            # Generate new state/action pairs according to the current Q function
            prior_obs = env.reset()
            done = False
            while not done:
                if random.random() < epsilon:
                    a = random.randint(0, 3)
                else:
                    a = evaluator(prior_obs)
                post_obs, reward, done, info = env.step(a)
                self.memory.store(prior_obs, a, reward, post_obs, done)
                prior_obs = post_obs

            # Get a sample
            sample = self.memory.sample(minibatch_size)
            obs1     = np.array([o1 for o1, a, r, o2, d in sample])
            actions  = np.array([a  for o1, a, r, o2, d in sample])
            rewards  = np.array([r  for o1, a, r, o2, d in sample])
            obs2     = np.array([o2 for o1, a, r, o2, d in sample])
            not_done = np.array([0.0 if d else 1.0  for o1, a, r, o2, d in sample])

            # Update the q and target networks.
            summary, _ = sess.run([self.summaries, self.optimize], {
                self.X_q: obs1,
                self.X_t: obs2,
                self.action: actions,
                self.reward: rewards,
                self.not_done: not_done,
            })
            sess.run(self.update_emas)
            summary_writer.add_summary(summary, i)

            # update epsilon
            if i <= epsilon_ramp:
                ramp_completion = float(i) / epsilon_ramp
                epsilon = (1.0 - ramp_completion) * epsilon_0 + ramp_completion * epsilon_1

            if i % 1000 == 0:
                # Perform an evaluation run (no random actions)
                test_episodes = 10
                actions = np.zeros((test_episodes))
                scores = np.zeros((test_episodes))
                possible_scores = np.zeros((test_episodes))
                for j in range(0, 10):
                    prior_obs = env.reset()
                    done = False
                    possible_scores[j] = env.remaining_stock_blocks()
                    while not done:
                        a = evaluator(prior_obs)
                        post_obs, reward, done, info = env.step(a)
                        self.memory.store(prior_obs, a, reward, post_obs, done)
                        prior_obs = post_obs
                        actions[j] += 1
                        scores[j] += reward
                total_actions = np.sum(actions)
                total_score = np.sum(scores)
                total_possible_score = np.sum(possible_scores)
                score_ratios = scores / possible_scores
                print("Episode " + str(i) + ": epsilon = " + str(epsilon))
                print("  Performed " + str(total_actions) + " total actions, average of " + str(np.mean(actions)) + " per episode")
                print("  Scored " + str(total_score) + " of " + str(total_possible_score) + " (" + str(100*total_score/total_possible_score) + "%)")
                print("  Average score per episode: " + str(np.mean(scores)) + ", average score ratio per episode: " + str(np.mean(score_ratios)))


if __name__ == "__main__":
    model = Model()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    model.train(sess)

