from torch import Tensor

class Experience:

    def __init__(
        self,
        state:Tensor,
        action:Tensor,
        next_state:Tensor,
        reward:int) -> None:
        
        self.state=state
        self.action=action
        self.next_state=next_state
        self.reward=reward