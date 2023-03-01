import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
import numpy as np
import pickle

ENCODED_SHAPE = (16,7,7)

class Policy(ABC):
    def __init__(self,name:str) -> None:
        self.name=name
    
    @abstractmethod
    def predict_best_action(self,state:Tensor, action_space:np.ndarray|None=None) -> Tensor:
        pass

class DQN(nn.Module, Policy):
    def __init__(self, observation_sample:Tensor, action_space:np.ndarray,lin_features:list[int]=[32,64,64]) -> None:
        super().__init__()
        Policy.__init__(self,"Deep_Q_Network")

        self._canvas_shape=action_space.shape

        next_cell=observation_sample[0:2]
        canvas=observation_sample[2:]
        
        self._fc1 = nn.Linear(canvas.shape.numel(),lin_features[0])
        self._fc2 = nn.Linear(lin_features[0],lin_features[1])
        self._fc3 = nn.Linear(lin_features[1]+next_cell.shape.numel(),lin_features[2])
        self._output = nn.Linear(lin_features[2],canvas.shape.numel())
    
    def forward(self,t:Tensor):

        next_cell=t[:,0:2]
        canvas=t[:,2:]

        canvas_fwd:Tensor = F.relu(self._fc1(canvas))
        canvas_fwd:Tensor = F.relu(self._fc2(canvas_fwd))

        t = torch.cat([next_cell,canvas_fwd],dim=1)
        t = F.relu(self._fc3(t))
        t = F.relu(self._output(t))

        t = t.reshape((-1,*self._canvas_shape))

        return t
    
    def predict_best_action(self,state:Tensor) -> Tensor:
        with torch.no_grad():
            best_loc=torch.argmax(self.forward(state.unsqueeze(0))).item()
        idx=np.unravel_index(best_loc,self._canvas_shape) # type: ignore
        return torch.from_numpy(np.int32(idx))

class Enc_DQN(nn.Module, Policy):
    def __init__(self, observation_sample:Tensor, action_space:np.ndarray,lin_features:list[int]=[64,64],decoded_depth:int=32) -> None:
        super().__init__()
        Policy.__init__(self,"Encoded Deep Q-Network")

        self._canvas_shape=action_space.shape

        next_cell=observation_sample[0:2]
        canvas=observation_sample[2:].reshape(action_space.shape)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.Conv2d(32, ENCODED_SHAPE[0], kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 8, 7, 7)
            nn.ReLU(),
        )

        with torch.no_grad():
            enc_canvas:Tensor=self.encoder(canvas.unsqueeze(0))
        
        fwd_size=next_cell.shape.numel()+enc_canvas.shape.numel()

        self._fc=nn.Sequential(
            nn.Linear(in_features=fwd_size,out_features=lin_features[0]),
            nn.ReLU())

        for idx,feat in list(enumerate(lin_features))[1:]:
            self._fc.add_module(
                f"Linear {idx+1}",
                nn.Linear(lin_features[idx-1],feat)
            )
            self._fc.add_module(
                f"ReLU {idx+1}",
                nn.ReLU()
            )
        
        self._fc.add_module(
            "Output",
            nn.Linear(lin_features[-1],decoded_depth*ENCODED_SHAPE[1]*ENCODED_SHAPE[2])
        )
        self._fc.add_module(
            "Output Act",
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoded_depth, 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0, output_padding=0)  # Output size: (batch_size, 1, 50, 50)
            #,nn.Tanh(),
        )
    
    def forward(self,t:Tensor)->Tensor:
        next_cell=t[:,0:2]
        canvas=t[:,2:].reshape((-1,1,*self._canvas_shape))
        batch_size=t.shape[0]

        enc_canvas:Tensor = self.encoder(canvas)
        
        t=torch.cat([
            next_cell,
            enc_canvas.flatten(start_dim=1)
        ],dim=1)

        t = self._fc(t)

        t = t.reshape((batch_size,-1,*ENCODED_SHAPE[1:]))

        t = self.decoder(t)

        return t.squeeze()

    def load_encoder_weights(self,w_file:str)->None:
        in_file=open(w_file,"rb")

        enc_weights=pickle.load(in_file)
        self.encoder.load_state_dict(enc_weights)
    
    def set_encoder_trainable(self, trainability:bool=False)->None:
        for param in self.encoder.parameters():
            param.requires_grad=trainability
        
    def trainable_params(self):
        return filter(lambda p: p.requires_grad,self.parameters())
    
    def predict_best_action(self,state:Tensor) -> Tensor:
        with torch.no_grad():
            best_loc=torch.argmax(self.forward(state.unsqueeze(0))).item()
        idx=np.unravel_index(best_loc,self._canvas_shape) # type: ignore
        return torch.from_numpy(np.int32(idx))



