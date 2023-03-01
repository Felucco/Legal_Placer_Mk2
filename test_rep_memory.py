from env.macro import Macro
from env.env import FloorPlan_2D_Grid
from rep_memory.replay_memory import ReplayMemory
from rep_memory.experience import Experience
import torch
from torch import Tensor
from random import randrange
from agent.policy import DQN

CANVAS = (30,30)
MACROS = [
    Macro(0,5,5), Macro(1,3,2),
    Macro(2,6,7), Macro(3,10,3)
]

device=torch.device("cpu")
e=FloorPlan_2D_Grid(CANVAS,MACROS,device)
act_space=e.action_space
e.render()

m=ReplayMemory(20)

for ep in range(10):
    e.reset()
    while not e.done:
        curr_state=e.state
        print("Current cell shape:",curr_state[0].cpu().numpy())
        act=Tensor(
            (
                randrange(0,act_space.shape[0]),
                randrange(0,act_space.shape[1])
            )
        ).to(dtype=torch.int32)
        print("Placing it in",act.numpy())
        e.take_action(act)
        m.push(Experience(curr_state,act,e.state,e.reward))
        if e.done:
            print("Something happened (episode ended)")
        e.render()

exp_batch=m.sample(3)
tensor_batch=m.sample(5,to_tensor=True,tensor_device=device)
print("Pippo")