from __future__ import annotations
from dataclasses import make_dataclass
import numpy as np
try:
    from rep_memory.experience import Experience
except ModuleNotFoundError:
    from experience import Experience
import torch
from torch import Tensor
import pickle

class ReplayMemory:
    def __init__(self, capacity:int, exp_sample:Experience) -> None:
        self._capacity=capacity
        self._push_count=0

        state_sample=exp_sample.state
        act_sample=exp_sample.action

        self._state_memory=torch.empty((capacity,*state_sample.shape))
        self._act_memory=torch.empty((capacity,*act_sample.shape))
        self._ns_memory=torch.empty((capacity,*state_sample.shape))
        self._rew_memory=torch.empty((capacity,))

        self._rnd_gen=np.random.default_rng()
     
    def push(self, exp: Experience) -> None:

        item_idx=self._push_count%self._capacity

        self._state_memory[item_idx]=exp.state
        self._act_memory[item_idx]=exp.action
        self._ns_memory[item_idx]=exp.next_state
        self._rew_memory[item_idx]=exp.reward

        self._push_count += 1
    
    def sample(self, batch_size:int,
            to_tensor: bool=True,
            tensor_device: torch.device=torch.device("cpu")) -> np.ndarray | tuple[Tensor,Tensor,Tensor,Tensor]:
        '''Sample a batch of experiences from the replay memory and returns them as either a numpy array
        of experiences or a tuple of 4 tensors (one for each Experience field)

        Parameters
        ----------
        batch_size: int
            How many samples to take from the replay memory
        to_tensor: bool, default False
            If true, the batch is returned as a tuple of torch Tensors
            If the state is composed of multiple tensors, both state and next_state will be tuples
        tensor_device: torch device on which to create output tensors
        '''


        max_idx=np.clip(self._push_count,0,self._capacity) #If the memory is not full we don't want to sample garbage
        all_idxs=np.arange(start=0,stop=max_idx)
        idxs=self._rnd_gen.choice(all_idxs,batch_size,replace=False)

        batch_states    = self._state_memory[idxs]
        batch_actions   = self._act_memory[idxs]
        batch_ns        = self._ns_memory[idxs]
        batch_rewards   = self._rew_memory[idxs]

        if to_tensor:
            return(
                batch_states.to(tensor_device),
                batch_actions.to(tensor_device),
                batch_ns.to(tensor_device),
                batch_rewards.to(tensor_device)
            )
        else:
            raise NotImplementedError()



    
    def can_sample(self, batch_size:int) -> bool:
        return self._push_count >= batch_size
    
    def export_env(self, out_file:str) -> None:
        end_idx=min(self._push_count,self._capacity)
        states=self._state_memory[:end_idx].cpu().numpy()
        actions=self._act_memory[:end_idx].cpu().numpy()
        next_states=self._ns_memory[:end_idx].cpu().numpy()

        out_memory=[states,actions,next_states]
        with open(out_file,"wb") as f:
            pickle.dump(out_memory,f)


def test_main():
    CAPACITY = 10
    exps = [
        Experience(
            torch.randint(0,10,(10,)),torch.randint(0,5,(2,)),
            torch.randint(0,10,(10,)),int(torch.randint(0,10,(1,)).item()))
        for _ in range(CAPACITY)
    ]

    rm=ReplayMemory(CAPACITY,exps[0])
    for e in exps:
        rm.push(e)
    
    batch=rm.sample(4)
    print("Pippo")


if __name__ == "__main__":
    test_main()