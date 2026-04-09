import gymnasium as gym

class EnvConfig:
    def __init__(self, env_id, max_episode_steps=500, seed=42,
                 use_one_hot=True, terminate_on_success=False,
                 vector_strategy="async", reward_func_version="v2",
                 num_goals=10):
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.use_one_hot = use_one_hot
        # terminate_on_success: set to True for evaluation envs
        self.terminate_on_success = terminate_on_success
        self.vector_strategy = vector_strategy
        self.reward_func_version = reward_func_version
        self.num_goals = num_goals

    def spawn(self):
        return gym.make_vec(
            f"Meta-World/{self.env_id}",
            seed=self.seed,
            use_one_hot=self.use_one_hot,
            terminate_on_success=self.terminate_on_success,
            max_episode_steps=self.max_episode_steps,
            vector_strategy=self.vector_strategy,
            reward_function_version=self.reward_func_version,
            num_goals=self.num_goals,
        )
