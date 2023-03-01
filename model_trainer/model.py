import torch
from torch import nn
from torch import Tensor
import numpy as np
import pickle

'''
    From a state-action pair we want to check wether the model can learn to predict the next state
'''
class NS_Model(nn.Module):
    def __init__(self, observation_sample:Tensor, action_space:Tensor,lin_features:list[int]=[512,512,392]) -> None:
        super().__init__()

        self._canvas_shape=action_space.shape

        next_cell=observation_sample[0:2]
        next_cell_loc=observation_sample[2:4]
        canvas=observation_sample[4:].reshape(action_space.shape)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 8, 7, 7)
            nn.ReLU(),
        )

        with torch.no_grad():
            enc_canvas:Tensor=self.encoder(canvas.unsqueeze(0))
        
        fwd_size=next_cell.shape.numel()+next_cell_loc.shape.numel()+enc_canvas.shape.numel()

        self.fc = nn.Sequential(
            nn.Linear(in_features=fwd_size,out_features=lin_features[0]),
            nn.ReLU(),
            nn.Linear(in_features=lin_features[0],out_features=lin_features[1]),
            nn.ReLU(),
            nn.Linear(in_features=lin_features[1],out_features=lin_features[2]),
            nn.ReLU()
        )

        #Reshape to 8 x 7 x 7

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0, output_padding=0),  # Output size: (batch_size, 1, 50, 50)
            nn.Tanh(),
        )
    
    def forward(self,t:Tensor)->Tensor:
        batch_size=t.shape[0]

        next_cell=t[:,0:2]
        next_cell_loc=t[:,2:4]
        canvas=t[:,4:].reshape((-1,1,*self._canvas_shape))

        enc_canvas:Tensor = self.encoder(canvas)
        
        t=torch.cat([
            next_cell,next_cell_loc,
            enc_canvas.flatten(start_dim=1)
        ],dim=1)

        t = self.fc(t)
        t = t.reshape(batch_size,8,7,7)

        t = self.decoder(t)

        return t.squeeze(1)

    def load_encoder_weights(self,w_file:str)->None:
        in_file=open(w_file,"rb")

        enc_weights=pickle.load(in_file)
        self.encoder.load_state_dict(enc_weights)
    
    def load_decoder_weigths(self,w_file:str)->None:
        in_file=open(w_file,"rb")

        dec_weights=pickle.load(in_file)
        self.decoder.load_state_dict(dec_weights)
    
    def set_encoder_trainable(self, trainability:bool=False)->None:
        for param in self.encoder.parameters():
            param.requires_grad=trainability
    
    def set_decoder_trainable(self, trainability:bool=False)->None:
        for param in self.decoder.parameters():
            param.requires_grad=trainability
        
    def trainable_params(self):
        return filter(lambda p: p.requires_grad,self.parameters())

class MLP_Embedder (nn.Module):

    def __init__(self,encoded_size:int,lin_features:list[int]=[512,512]) -> None:
        super().__init__()

        self.input_size=2+2+encoded_size #2 values for cell shape and 2 values for cell position

        self._fc=nn.Sequential(
            nn.Linear(in_features=self.input_size,out_features=lin_features[0]),
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
            nn.Linear(lin_features[-1],encoded_size)
        )
        self._fc.add_module(
            "Output Act",
            nn.ReLU()
        )
    
    def forward(self, x:torch.Tensor) -> Tensor:
        return self._fc(x)

    