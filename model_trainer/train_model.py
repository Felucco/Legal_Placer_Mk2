from model import NS_Model, MLP_Embedder
from next_state_eval import get_result_canvas,render
import numpy as np
import pickle
import torch
from torch.utils import data as data_utils
from matplotlib import pyplot as plt
from torch import nn
from matplotlib.colors import Normalize
import torchsummary

CANVAS_SIZE=(50,50)
LR=1e-3
EPOCHS=200
BATCH_SIZE=64
ENV_FILE="/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/env_dump.dat"
ENC_FILE="/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/enc_weights"
DEC_FILE="/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/dec_weights"
ENCODED_SHAPE=(16,7,7)
device=torch.device("cuda")

rng=np.random.default_rng()

with open(ENV_FILE,"rb") as f:
    env_mem=pickle.load(f)

states:np.ndarray=env_mem[0]
actions:np.ndarray=env_mem[1]
next_states:np.ndarray=env_mem[2]

print(f"Loaded {states.shape[0]} inputs")

inputs=torch.from_numpy(np.concatenate([states[:,:2],actions,states[:,2:]],axis=1)).to(device=device)
outputs=torch.from_numpy(next_states[:,2:].reshape((-1,*CANVAS_SIZE))).to(device=device)

train_x=inputs[:-100]
train_y=outputs[:-100]

test_x=inputs[-100:]
test_y=outputs[-100:]

train=data_utils.TensorDataset(train_x,train_y)

loader=data_utils.DataLoader(train,batch_size=BATCH_SIZE,shuffle=True)

encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.Conv2d(32, ENCODED_SHAPE[0], kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 16, 7, 7)
            nn.ReLU(),
        ).to(device)
with open(ENC_FILE,"rb") as f:
    encoder.load_state_dict(pickle.load(f))

decoder = nn.Sequential(
            nn.ConvTranspose2d(ENCODED_SHAPE[0], 32, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 32, 13, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # Output size: (batch_size, 16, 25, 25)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0, output_padding=0),  # Output size: (batch_size, 1, 50, 50)
            nn.Tanh(),
        ).to(device)

with open(DEC_FILE,"rb") as f:
    decoder.load_state_dict(pickle.load(f))

encoder.eval()
decoder.eval()

model=MLP_Embedder(ENCODED_SHAPE[0]*ENCODED_SHAPE[1]*ENCODED_SHAPE[2],[1024,1024,1024]).to(device=device)

torchsummary.summary(model,(2+2+ENCODED_SHAPE[0]*ENCODED_SHAPE[1]*ENCODED_SHAPE[2],))

opt=torch.optim.Adam(model.parameters(),lr=LR)

def train_one_epoch():
    running_loss = 0.

    for data in loader:
        inputs, targets = data

        input_canvas=inputs[:,4:].reshape(-1,1,*CANVAS_SIZE)
        enc_canvas=encoder(input_canvas).flatten(start_dim=1)
        enc_inputs=torch.cat([inputs[:,:4],enc_canvas],dim=1)
        enc_targets=encoder(targets.unsqueeze(1)).flatten(start_dim=1)

        opt.zero_grad()

        enc_outputs = model(enc_inputs)

        loss = torch.nn.functional.mse_loss(enc_outputs, enc_targets)
        loss.backward()
        opt.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss/len(loader)


def show_test(n_fp:int=5):
    norm=Normalize(vmin=-1,vmax=1)
    plt.figure(1)
    for fp in range(n_fp):
        idx=rng.integers(0,test_x.shape[0])
        x=test_x[idx]
        enc_x:torch.Tensor=encoder(x[4:].reshape(1,1,*CANVAS_SIZE)).flatten(start_dim=1)
        enc_inp=torch.cat([x[:4].unsqueeze(0),enc_x],dim=1)
        y=test_y[idx].cpu().numpy()
        with torch.no_grad():
            enc_pred:torch.Tensor=model(enc_inp)
        
        dec_pred:torch.Tensor=decoder(enc_pred.reshape(1,*ENCODED_SHAPE))
        pred = dec_pred.detach().squeeze().cpu().numpy()
        x=x[4:].reshape(CANVAS_SIZE).cpu().numpy()

        plt.subplot(n_fp,3,3*fp+1)
        plt.imshow(x,origin="lower",norm=norm,cmap="RdYlGn")
        plt.title("Initial")
        plt.subplot(n_fp,3,3*fp+2)
        plt.imshow(y,origin="lower",norm=norm,cmap="RdYlGn")
        plt.title("Real")
        plt.subplot(n_fp,3,3*fp+3)
        plt.imshow(pred,origin="lower",norm=norm,cmap="RdYlGn")
        plt.title("Predicted")
    plt.show()

for epoch in range(EPOCHS):
    epoch_mean_loss=train_one_epoch()
    print(f"Epoch {epoch} -> Loss = {epoch_mean_loss}")

show_test()


