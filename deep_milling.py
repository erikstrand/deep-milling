from states import *
import tensorflow as tf
tf.set_random_seed(2016)

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

# Evaluation hyperparameters
evaluation_episodes = 500

class TrainingAgent:
    def __init__(self):
        self.environment = Environment()
        self.reset()

    def reset(self):
        self.observation = self.environment.reset()
        self.reset_stats()

    def reset_stats(self):
        self.actions = 0
        self.max_cumulative_reward = self.environment.remaining_stock_blocks()
        self.cumulative_reward = 0.0
        self.q_actions = 0
        self.random_actions = 0
        self.repeat_attempts = 0
        self.repeats = 0

    def advance(self, epsilon, evaluator):
        part = self.environment.part
        s = self.environment.state
        if s in self.environment.transitions:
            explored_actions = self.environment.transitions[s]
        else:
            explored_actions = set()
        unexplored_actions = set([0, 1, 2, 3]) - explored_actions
        new = True
        if len(unexplored_actions) == 0:
            new = False
            self.repeats += 1
            self.random_actions += 1
            a = random.randint(0, 3)
        elif random.random() < epsilon:
            self.random_actions += 1
            a = random.sample(unexplored_actions, 1)[0]
        else:
            a = evaluator(self.observation)
            if a in unexplored_actions:
                self.q_actions += 1
            else:
                self.repeat_attempts += 1
                a = random.sample(unexplored_actions, 1)[0]
        self.observation, r, done, info = self.environment.step(a)
        self.actions += 1
        self.cumulative_reward += r
        return a, r, new, done


class EvaluationAgent:
    def __init__(self):
        self.environment = Environment()

    def reset(self):
        self.observation = self.environment.reset()
        self.actions = 0
        self.max_cumulative_reward = self.environment.remaining_stock_blocks()
        self.cumulative_reward = 0.0
        self.q_actions = 0
        self.random_actions = 0
        self.repeat_attempts = 0
        self.repeats = 0

    def advance(self, evaluator):
        a = evaluator(self.observation)
        self.observation, r, done, info = self.environment.step(a)
        self.actions += 1
        self.cumulative_reward += r
        if info.deja_vu:
            done = True
        return done


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
        self.actions += session.actions
        self.q_actions += session.q_actions
        self.random_actions += session.random_actions
        self.repeat_attempts += session.repeat_attempts
        self.repeats += session.repeats
        self.cumulative_reward += session.cumulative_reward
        self.max_cumulative_reward += session.max_cumulative_reward


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

        # Model parameters for Q function
        W1_q = tf.Variable(tf.truncated_normal([W * H * 2, self.L], stddev=0.1), name="W1")
        B1_q = tf.Variable(tf.ones([self.L])/10, name="B1")
        W2_q = tf.Variable(tf.truncated_normal([self.L, self.M], stddev=0.1), name="W2")
        B2_q = tf.Variable(tf.ones([self.M])/10, name="B2")
        W3_q = tf.Variable(tf.truncated_normal([self.M, self.N], stddev=0.1), name="W3")
        B3_q = tf.Variable(tf.ones([self.N])/10, name="B3")
        W4_q = tf.Variable(tf.truncated_normal([self.N, A], stddev=0.1), name="W4")
        B4_q = tf.Variable(tf.ones([A])/10, name="B4")

        # The target network uses moving averages of the Q network.
        ema = tf.train.ExponentialMovingAverage(decay=target_network_decay)
        self.update_averages = ema.apply([W1_q, B1_q, W2_q, B2_q, W3_q, B3_q, W4_q, B4_q])
        W1_t = ema.average(W1_q)
        B1_t = ema.average(B1_q)
        W2_t = ema.average(W2_q)
        B2_t = ema.average(B2_q)
        W3_t = ema.average(W3_q)
        B3_t = ema.average(B3_q)
        W4_t = ema.average(W4_q)
        B4_t = ema.average(B4_q)

        # The model
        XX_q = tf.reshape(self.X_q, [-1, 2 * W * H])
        Y1_q = tf.nn.relu(tf.matmul(XX_q, W1_q) + B1_q)
        Y2_q = tf.nn.relu(tf.matmul(Y1_q, W2_q) + B2_q)
        Y3_q = tf.nn.relu(tf.matmul(Y2_q, W3_q) + B3_q)
        Y4_q = tf.nn.relu(tf.matmul(Y3_q, W4_q) + B4_q)

        XX_t = tf.reshape(self.X_t, [-1, 2 * W * H])
        Y1_t = tf.nn.relu(tf.matmul(XX_t, W1_t) + B1_t)
        Y2_t = tf.nn.relu(tf.matmul(Y1_t, W2_t) + B2_t)
        Y3_t = tf.nn.relu(tf.matmul(Y2_t, W3_t) + B3_t)
        Y4_t = tf.nn.relu(tf.matmul(Y3_t, W4_t) + B4_t)

        # Used to select actions
        self.best_action = tf.argmax(Y4_q, 1)

        # Loss
        reward_q = tf.reduce_sum(tf.one_hot(self.action, A) * Y4_q, 1)
        reward_t = tf.add(self.reward, tf.mul(gamma * self.not_done, tf.reduce_max(Y4_t, 1)))
        expected_squared_error = tf.reduce_mean(tf.square(tf.sub(reward_q, reward_t)))
        self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(expected_squared_error)

        # Memory bank
        self.memory = MemoryBank(memory_capacity)

    def train(self, sess):
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
        train_agent = TrainingAgent()
        eval_agent = EvaluationAgent()
        train_stats = StatAggregator()
        eval_stats = StatAggregator()
        evaluator = lambda s: sess.run(self.best_action, feed_dict = {self.X_q: [s]})[0]
        train_agent.reset()
        for i in range(1, training_cycles + 1):
            # Generate new state/action pairs according to the current Q function
            new_transitions = 0
            while new_transitions < actions_per_update:
                prior_observation = train_agent.observation
                a, r, new, done = train_agent.advance(epsilon, evaluator)
                post_observation = train_agent.observation
                if new:
                    new_transitions += 1
                    self.memory.store(prior_observation, a, r, post_observation, done)
                if done:
                    train_stats.update(train_agent)
                    train_agent.reset()

            # Get a sample
            sample = self.memory.sample(minibatch_size)
            obs1     = np.array([o1 for o1, a, r, o2, d in sample])
            actions  = np.array([a  for o1, a, r, o2, d in sample])
            rewards  = np.array([r  for o1, a, r, o2, d in sample])
            obs2     = np.array([o2 for o1, a, r, o2, d in sample])
            not_done = np.array([0.0 if d else 1.0  for o1, a, r, o2, d in sample])

            # Update the q and target networks.
            sess.run(self.optimize, {
                self.X_q: obs1,
                self.X_t: obs2,
                self.action: actions,
                self.reward: rewards,
                self.not_done: not_done,
            })
            sess.run(self.update_averages)

            # update epsilon
            if i <= epsilon_ramp:
                ramp_completion = float(i) / epsilon_ramp
                epsilon = (1.0 - ramp_completion) * epsilon_0 + ramp_completion * epsilon_1

            # Print an update if applicable
            if i % print_interval == 0:
                eval_agent.reset()
                for _ in range(0, evaluation_episodes):
                    done = False
                    while not done:
                        done = eval_agent.advance(evaluator)
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
    model = Model()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    model.train(sess)

