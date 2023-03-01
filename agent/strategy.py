from abc import ABC,abstractmethod
import numpy as np

class Strategy(ABC):

    def __init__(self, name:str) -> None:
        self.name=name
    
    @abstractmethod
    def get_exp_rate(self, n_steps:int) -> float:
        pass

    def reset(self) -> None:
        pass

    def next_episode(self) -> None:
        pass

class EpsGreedy(Strategy):

    def __init__(self, eps_start: float, eps_end: float, eps_decay: float) -> None:
        super().__init__("EpsilonGreedy")
        self._eps_start=eps_start
        self._eps_end=eps_end
        self._eps_decay=eps_decay
    
    def get_exp_rate(self, n_steps: int) -> float:
        return self._eps_end+(self._eps_start-self._eps_end)*np.exp(-1.*n_steps*self._eps_decay)

class LinEpsGreedy(Strategy):

    def __init__(self, eps_start: float, eps_end: float, n_episodes: int) -> None:
        super().__init__("LinearEpsilonGreedy")
        self._eps_start=eps_start
        self._eps_end=eps_end
        self._max_epi=n_episodes
        self._n_epi=0
    
    def get_exp_rate(self, n_steps: int) -> float:
        return (self._eps_start-(self._eps_start-self._eps_end)*self._n_epi/self._max_epi) if self._n_epi <= self._max_epi else self._eps_end
    
    def reset(self) -> None:
        self._n_epi=0
    
    def next_episode(self) -> None:
        self._n_epi+=1