from agent.agent import Agent
from agent.policy import DQN, Enc_DQN
from agent.strategy import LinEpsGreedy
from env.env import FloorPlan_2D_Grid
import torch
from torch.nn import functional as F
from torch.optim import Adam,SGD,RMSprop
from rep_memory.replay_memory import ReplayMemory
from rep_memory.experience import Experience
from copy import deepcopy
import utils
import numpy as np
from matplotlib import pyplot as plt
import torchsummary
from torch.nn.modules.loss import _Loss as Loss

MAX_REW = np.max(np.abs(list(FloorPlan_2D_Grid.rewards.values())))

class AgentTrainer:

    N_EPISODES      = 15000
    BATCH_SIZE      = 128
    LEARNING_RATE   = 1e-4
    LR_DECAY        = 0

    LIN_FEATURES    = [512,512]
    DECODED_DEPTH   = 16

    EPS_START       = 1
    EPS_END         = 0.05
    EPS_DECAY       = 1e-4
    GAMMA           = 0.999

    MEMORY_CAP      = 10000

    TARGET_UPD      = 1000

    TARGET_VAL      = 250

    EPISODE_SHOW    = 2500
    EPISODE_STAT    = 50

    def __init__(self, env:FloorPlan_2D_Grid, device:torch.device) -> None:
        self._device = device

        self._strategy = LinEpsGreedy(
            AgentTrainer.EPS_START,
            AgentTrainer.EPS_END,
            int(AgentTrainer.N_EPISODES*0.7)
        )

        self._rm=ReplayMemory(
            AgentTrainer.MEMORY_CAP,
            Experience(env.obs_sample,torch.zeros((2,)),env.obs_sample,0))


        lin_feats=AgentTrainer.LIN_FEATURES
        dec_depth=AgentTrainer.DECODED_DEPTH
        self._policy_net=Enc_DQN(env.obs_sample,env.action_space,lin_feats,decoded_depth=dec_depth).to(device)
        self._target_net=Enc_DQN(env.obs_sample,env.action_space,lin_feats,decoded_depth=dec_depth).to(device)
        self._policy_net.load_encoder_weights("/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/enc_weights")
        self._policy_net.set_encoder_trainable(True)
        torchsummary.summary(self._policy_net,env.obs_sample.shape)
        self._target_net.eval()
        self._target_net.load_state_dict(self._policy_net.state_dict())

        self._opt = Adam(self._policy_net.trainable_params(),AgentTrainer.LEARNING_RATE,weight_decay=AgentTrainer.LR_DECAY)

        self._agent=Agent(self._strategy,self._policy_net,
            env.action_space,torch.device("cpu"))

        self._env = env

        self._loss:Loss=torch.nn.SmoothL1Loss()
    
    def show_episode(self):
        self._env.reset()
        while not self._env.done:
            plt.subplot(2,2,1)
            plt.imshow(self._env.render("noplt"),origin="lower") #type:ignore
            plt.title("Initial Canvas")

            state=self._env.state
            act=self._policy_net.predict_best_action(state)
            with torch.no_grad():
                canvas_heatmap=self._policy_net(state.unsqueeze(0)).detach().cpu().numpy().squeeze()
            self._env.take_action(act)
            res=self._env.render("noplt")
            with torch.no_grad():
                post_heatmap=self._policy_net(self._env.state.unsqueeze(0)).detach().cpu().numpy().squeeze()
            heatmap_diff = post_heatmap-canvas_heatmap

            plt.subplot(2,2,2)
            plt.imshow(canvas_heatmap,origin="lower",cmap="RdYlGn")
            plt.title("Canvas predicted QValues")

            plt.subplot(2,2,3)
            plt.imshow(res,origin="lower") #type: ignore
            plt.title("Floor Plan")

            plt.subplot(2,2,4)
            plt.imshow(heatmap_diff,origin="lower",cmap="RdYlGn")
            plt.title("HM Post-Place diff")
            plt.show()


        self._env.reset()


    
    def train(self):
        episode_rewards=[]
        batch_losses=[]
        for episode in range(AgentTrainer.N_EPISODES):
            self._env.reset(shuffle_macros=True)
            episode_rewards.append(0)

            while not self._env.done:
                state=self._env.state
                sel_act=self._agent.choose_action(state)

                self._env.take_action(sel_act)

                rew=self._env.reward

                episode_rewards[-1] += rew

                self._rm.push(
                    Experience(state,sel_act,self._env.state,rew)
                )

                if self._rm.can_sample(AgentTrainer.BATCH_SIZE):
                    states,actions,next_states,rewards=self._rm.sample(
                        AgentTrainer.BATCH_SIZE,to_tensor=True,
                        tensor_device=self._device
                    )

                    self._opt.zero_grad()

                    pred_qvals=utils.QValues.get_current(
                        self._policy_net,states,actions
                    )
                    
                    rewards:torch.Tensor

                    final_states=rewards.abs().eq(MAX_REW)
                    non_final_states=final_states.logical_not()
                    next_qvals=utils.QValues.get_next(self._target_net,next_states[non_final_states])
                    
                    target_qvals=torch.clone(rewards).to(dtype=pred_qvals.dtype)
                    target_qvals[non_final_states] += AgentTrainer.GAMMA*next_qvals

                    loss:torch.Tensor=self._loss(pred_qvals,target_qvals)
                    loss.backward()
                    self._opt.step()
                    batch_losses.append(loss.item())
            
            self._agent._strategy.next_episode()

            if episode>0 and episode % AgentTrainer.TARGET_UPD == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())
            if episode>0 and episode % AgentTrainer.EPISODE_STAT == 0:
                ep_mean_rewards=np.mean(episode_rewards[-AgentTrainer.EPISODE_STAT:])
                loss_100_batch=np.mean(batch_losses[-100:])
                print(f"Episode {episode}\t-> 100 batches loss = {loss_100_batch:.3f}\t{AgentTrainer.EPISODE_STAT} ep duration: {ep_mean_rewards:.2f}")
                if episode%AgentTrainer.EPISODE_SHOW == 0:
                    self.show_episode()
                    #self._rm.export_env("/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/env_dump.dat")
