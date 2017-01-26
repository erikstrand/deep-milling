from states import *
import tensorflow as tf
import os
import os.path
tf.set_random_seed(2016)

# Training hyperparameters
training_episodes = 3000000
memory_capacity = 100000 # transitions in experience memory
memory_initial = 50000
minibatch_size = 64
target_network_decay = 0.99
learning_rate = 0.001

# Q learning hyperparameters
gamma = 0.99
epsilon_0 = 1.0
epsilon_1 = 0.05
epsilon_ramp = 700000.0


class MemoryBank:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
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
        self.L = 144 # neurons in first layer
        self.M = 100 # neurons in second layer
        self.N = 32 # neurons in third layer

        # Placeholders
        self.X_q = tf.placeholder(tf.float32, [None, 2, W, H])
        self.X_t = tf.placeholder(tf.float32, [None, 2, W, H])
        self.action = tf.placeholder(tf.int32, [None])
        self.reward = tf.placeholder(tf.float32, [None])
        self.not_done = tf.placeholder(tf.float32, [None])
        
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
        with tf.name_scope("net_stats"):
            tf.summary.scalar("expected reward (q)", tf.reduce_mean(q_reward))
            tf.summary.scalar("expected reward (t)", tf.reduce_mean(t_reward))
            tf.summary.scalar("expected_squared_error", expected_squared_error)

        with tf.name_scope("layer_4_stats"):
            mean = tf.reduce_mean(q4)
            tf.summary.scalar("layer_4_mean", mean)
            with tf.name_scope("std_dev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(q4 - mean)))
            tf.summary.scalar("layer_4_std_dev", stddev)
            tf.summary.scalar('max', tf.reduce_max(q4))
            tf.summary.scalar('min', tf.reduce_min(q4))

        # Performance stats
        self.test_episodes = 10
        with tf.name_scope("environment_stats"):
            self.test_actions = tf.placeholder(tf.float32, (None))
            self.test_blocks_milled = tf.placeholder(tf.float32, (None))
            self.test_max_blocks_milled = tf.placeholder(tf.float32, (None))
            self.test_total_rewards = tf.placeholder(tf.float32, (None))
            self.test_death_rewards = tf.placeholder(tf.float32, (None))
            max_rewards = self.test_max_blocks_milled + 10.0
            tf.summary.scalar("actions per episode", tf.reduce_mean(self.test_actions))
            tf.summary.scalar("blocks milled ratio", tf.reduce_mean(self.test_blocks_milled / self.test_max_blocks_milled))
            tf.summary.scalar("blocks per action", tf.reduce_mean(self.test_blocks_milled / self.test_actions))
            tf.summary.scalar("perfectness", tf.reduce_mean(self.test_total_rewards / max_rewards))
            tf.summary.scalar("death honor", tf.reduce_mean(self.test_death_rewards))

        self.summaries = tf.summary.merge_all()

    def train(self, sess):
        train_writer = tf.summary.FileWriter("./train", sess.graph)
        test_writer = tf.summary.FileWriter("./test", sess.graph)
        test_memory = MemoryBank(self.test_episodes)

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
            train_actions = 0.0
            train_max_blocks_milled = env.remaining_stock_blocks()
            train_total_reward = 0.0
            while not done:
                if random.random() < epsilon:
                    a = random.randint(0, 3)
                else:
                    a = evaluator(prior_obs)
                post_obs, reward, done, info = env.step(a)
                self.memory.store(prior_obs, a, reward, post_obs, done)
                prior_obs = post_obs
                train_actions += 1.0
                train_total_reward += reward
            train_blocks_milled = env.blocks_milled
            train_death_reward = env.death_reward

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
                self.test_actions: np.array([train_actions]),
                self.test_blocks_milled: np.array([train_blocks_milled]),
                self.test_max_blocks_milled: np.array([train_max_blocks_milled]),
                self.test_total_rewards: np.array([train_total_reward]),
                self.test_death_rewards: np.array([train_death_reward])
            })
            # TODO this is gross... let's group update_emas into optimize.
            sess.run(self.update_emas)
            if i % 100 == 0:
                train_writer.add_summary(summary, i)

            # update epsilon
            if i <= epsilon_ramp:
                ramp_completion = float(i) / epsilon_ramp
                epsilon = (1.0 - ramp_completion) * epsilon_0 + ramp_completion * epsilon_1

            if i % 1000 == 0:
                # Perform an evaluation run (no random actions)
                test_memory.reset()
                images = {}
                test_actions = np.zeros((self.test_episodes))
                test_blocks_milled = np.zeros((self.test_episodes))
                test_max_blocks_milled = np.zeros((self.test_episodes))
                test_total_rewards = np.zeros((self.test_episodes))
                test_death_rewards = np.zeros((self.test_episodes))
                for j in range(0, self.test_episodes):
                    k = 0
                    prior_obs = env.reset()
                    done = False
                    test_max_blocks_milled[j] = env.remaining_stock_blocks()
                    images[(j, k)] = env.pil_image()
                    while not done:
                        k += 1
                        a = evaluator(prior_obs)
                        post_obs, reward, done, info = env.step(a)
                        test_memory.store(prior_obs, a, reward, post_obs, done)
                        prior_obs = post_obs
                        test_actions[j] += 1.0
                        test_total_rewards[j] += reward
                        images[(j, k)] = env.pil_image()
                    test_blocks_milled[j] = env.blocks_milled
                    test_death_rewards[j] = env.death_reward

                sample = test_memory.sample(self.test_episodes)
                obs1     = np.array([o1 for o1, a, r, o2, d in sample])
                actions  = np.array([a  for o1, a, r, o2, d in sample])
                rewards  = np.array([r  for o1, a, r, o2, d in sample])
                obs2     = np.array([o2 for o1, a, r, o2, d in sample])
                not_done = np.array([0.0 if d else 1.0  for o1, a, r, o2, d in sample])
                print("Episode " + str(i) + ": epsilon = " + str(epsilon))
                summary = sess.run(self.summaries, {
                    self.X_q: obs1,
                    self.X_t: obs2,
                    self.action: actions,
                    self.reward: rewards,
                    self.not_done: not_done,
                    self.test_actions: test_actions,
                    self.test_blocks_milled: test_blocks_milled,
                    self.test_max_blocks_milled: test_max_blocks_milled,
                    self.test_total_rewards: test_total_rewards,
                    self.test_death_rewards: test_death_rewards,
                })
                test_writer.add_summary(summary, i)

                if i % 10000 == 0:
                    image_dir = "./img/" + str(i/1000) + "k/"
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    for j, k in images:
                        image_name = image_dir + "episode" + str(j) + "_step" + str(k) + ".jpg"
                        images[j, k].save(image_name)


if __name__ == "__main__":
    model = Model()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    model.train(sess)

