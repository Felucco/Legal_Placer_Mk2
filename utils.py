from matplotlib import pyplot as plt
import torch
import numpy as np
from rep_memory.replay_memory import Experience
from torch import Tensor
from agent.policy import DQN, Enc_DQN
from agent.agent import Agent
from env.env import FloorPlan_2D_Grid

def get_moving_avg(period:int, in_values:list)->np.ndarray:
    values=torch.tensor(in_values,dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0,size=period,step=1)\
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1),moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def plot(values: list, moving_avg_period:int)->None:
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)
    plt.plot(get_moving_avg(moving_avg_period,values))
    plt.pause(0.001)

def play_episode(env:FloorPlan_2D_Grid, ag:Agent):
    env.reset()
    while not env.done:
        screen:np.ndarray=env.render() # type: ignore
        plt.figure(3)
        plt.clf()
        plt.imshow(screen)
        plt.pause(0.01)
        env.take_action(ag.choose_action(env.state))
    plt.close(3)

class QValues():
    device = torch.device("cuda")

    @staticmethod
    def get_current(policy_net: DQN|Enc_DQN, states: Tensor, actions: Tensor) -> Tensor:
        canvas_size=policy_net._canvas_shape
        
        pred_fps:Tensor=policy_net(states)
        pred_fps = pred_fps.flatten(start_dim=1)
        lin_action = actions[:,1]+actions[:,0]*canvas_size[1]
        lin_action = lin_action.unsqueeze(1).to(dtype=torch.int64)

        res = pred_fps.gather(dim=1,index=lin_action).squeeze().to(QValues.device)
        return res
    
    @staticmethod
    def get_next(target_net: DQN|Enc_DQN, next_states: Tensor) -> Tensor:
        pred_fps:Tensor=target_net(next_states)
        pred_fps=pred_fps.flatten(start_dim=1)
        return pred_fps.max(dim=1)[0]