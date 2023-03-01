from agent_trainer import AgentTrainer
from env.env import FloorPlan_2D_Grid
from env.macro import Macro
from random import randint
import numpy as np
import torch

device = torch.device("cuda")

MAX_MACROS=10
MAX_SIDE=10
CANVAS_SIDE=50

canvas=(CANVAS_SIDE,CANVAS_SIDE)
canvas_size=CANVAS_SIDE*CANVAS_SIDE

macros:list[Macro]=[]

total_size=0
for i in range(MAX_MACROS):
    m=Macro(idx=i,height=randint(1,MAX_SIDE),width=randint(1,MAX_SIDE))
    total_size+=m.size
    if total_size < 0.2*canvas_size:
        macros.append(m)
    else:
        break

e=FloorPlan_2D_Grid(canvas,macros,device)

trainer=AgentTrainer(e,device)
trainer.train()
trainer.show_episode()