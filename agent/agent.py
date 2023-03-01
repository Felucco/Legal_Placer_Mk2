import torch
from agent.strategy import Strategy
from agent.policy import Policy
import numpy as np

class Agent:

    def __init__(self, strategy: Strategy, policy: Policy, action_space: np.ndarray, device: torch.device) -> None:
        self._strategy=strategy
        self._policy=policy
        self._acts=action_space
        self._n_steps=0
        self._device=device
        self._gen=np.random.default_rng()

    def choose_action(self, state:torch.Tensor) -> torch.Tensor:
        strategy_exp_rate = self._strategy.get_exp_rate(self._n_steps)
        exp_val=self._gen.uniform()
        self._n_steps += 1
        #print(f"{strategy_exp_rate=}")

        if exp_val < strategy_exp_rate:
            #print(f"Agent exploring {exp_val}")
            ymax,xmax=self._acts.shape
            y=self._gen.integers(0,ymax)
            x=self._gen.integers(0,xmax)
            return torch.tensor([y,x]).to(self._device,dtype=torch.int32)
        else:
            #print("Agent exploiting")
            return self._policy.predict_best_action(state).to(self._device)