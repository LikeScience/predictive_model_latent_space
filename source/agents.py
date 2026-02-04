import numpy as np

class random_egocentric_agent():
    def __init__(self, seed, probs):
        self.rng = np.random.default_rng(seed=seed)
        self.probs = probs
            
    def act(self):
        return self.rng.choice(range(3), p=self.probs)

class random_allocentric_agent():
    def __init__(self, seed, probs):
        self.rng = np.random.default_rng(seed=seed)
        self.probs = probs

    def act(self, agent_dir):
        x = self.rng.choice (range(4), p=self.probs)
        actions = [1 for i in range((x-agent_dir)%4)]+[2]
        return actions

        