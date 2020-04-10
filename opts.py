class Opts:
    def __init__(self):
        # Expetiment options
        self.episodes = 200
        self.target_avg_score = 30
        self.target_score_episodes = 100
        self.moves_per_episode = 1000
        self.minimum_action_value = -1
        self.maximum_action_value = 1

        # Neural network hidden layers configuration 
        self.a_fc1 = 128
        self.a_fc2 = 128
        self.c_fc1 = 128
        self.c_fc2 = 128

        # Agent training options
        self.device = "cuda"

        self.buffer_size=int(1e5)
        self.batch_size=64
        self.learn_every=1
        self.update_every=1
        self.random_seed=0

        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        self.critic_weight_decay = 0
        self.gamma = 0.99
        self.tau = 1e-3

        # Output parameters
        self.output_data_path = "data/"
        self.approach_title = f"DDPG {self.a_fc1}+{self.a_fc2}-{self.c_fc1}+{self.c_fc2} bs={self.batch_size}"