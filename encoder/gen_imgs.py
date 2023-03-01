import numpy as np
import random
from matplotlib import pyplot as plt
import pickle

CANVAS_SIZE = (50,50)
IMAGE_N     = 50000
MAX_SIDE    = 15
MAX_N       = 15
MACRO_P     = 0.7

imgs=np.ones((IMAGE_N,*CANVAS_SIZE))

for img in range(IMAGE_N):
    for macro in range(MAX_N):
        if random.random() > MACRO_P:
            continue
        macro_h=random.randrange(1,MAX_SIDE+1)
        macro_w=random.randrange(1,MAX_SIDE+1)
        macro_y=random.randrange(0,CANVAS_SIZE[0])
        macro_x=random.randrange(0,CANVAS_SIZE[1])

        top_y=np.clip(macro_y+macro_h,0,CANVAS_SIZE[0])
        top_x=np.clip(macro_x+macro_w,0,CANVAS_SIZE[1])

        imgs[img,macro_y:top_y,macro_x:top_x]-=1

        if np.any(imgs[img]<-0.1):
            break
    ###plt.imshow(imgs[img],origin='lower')
    ###plt.title(f"Image {img}")
    ###plt.show()

with open("gen_imgs.dat","wb") as f:
    pickle.dump(imgs,f)