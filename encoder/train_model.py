import pickle
from model import EncDec
import numpy as np
from torch import optim
from torch import nn
import torch
from torch.utils import data as data_utils
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

EPOCHS=20
LR = 1e-3
BATCH_SIZE = 128

rng=np.random.default_rng()

device=torch.device("cuda")

inp_images:np.ndarray
with open("/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/gen_imgs.dat","rb") as f:
    inp_images=pickle.load(f)

test_imgs=inp_images[-100:]

train_imgs=torch.from_numpy(inp_images[:-100]).unsqueeze(1).to(dtype=torch.float32,device=device)
train_dset=data_utils.TensorDataset(train_imgs)
loader=data_utils.DataLoader(train_dset,BATCH_SIZE,shuffle=True)

model = EncDec().to(device=device)
opt = optim.Adam(model.parameters(), lr=LR)

def get_test_batch(batch_size:int):
    batch=rng.choice(test_imgs,batch_size,replace=False)
    return torch.from_numpy(batch).unsqueeze(1).to(device=device,dtype=torch.float32)

def train_one_epoch():
    running_loss = 0.

    for (inputs,) in loader:
        opt.zero_grad()

        outputs = model(inputs)

        loss = torch.nn.functional.mse_loss(outputs, inputs)
        loss.backward()
        opt.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss/len(loader)

# Train the model
for epoch in range(EPOCHS):
    epoch_mean_loss=train_one_epoch()
    print(f"Epoch {epoch} -> Loss = {epoch_mean_loss}")


TEST_SIZE = 5
test_batch=get_test_batch(TEST_SIZE)
with torch.no_grad():
    test_outs:torch.Tensor=model(test_batch)

norm=Normalize(vmin=-1,vmax=1)
for t in range(TEST_SIZE):
    plt.subplot(TEST_SIZE,2,2*t+1)
    plt.imshow(test_batch[t].cpu().squeeze().numpy(),origin="lower",norm=norm,cmap="RdYlGn")
    plt.title("Input")
    plt.subplot(TEST_SIZE,2,2*t+2)
    plt.imshow(test_outs[t].cpu().squeeze().numpy(),origin="lower",norm=norm,cmap="RdYlGn")
    plt.title("Output")

plt.show()

with open("/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/enc_weights","wb") as f:
    pickle.dump(model.encoder.state_dict(),f)

with open("/home/felucco/Documents/Python/Reinforcement_Learning/Legal_Placer/dec_weights","wb") as f:
    pickle.dump(model.decoder.state_dict(),f)

