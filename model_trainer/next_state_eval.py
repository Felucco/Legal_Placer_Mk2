from macro import Macro
import numpy as np
from matplotlib import pyplot as plt

def get_result_canvas(inp_canvas:np.ndarray,next_macro:Macro,next_macro_loc:tuple[int,int]) -> np.ndarray:
    _, mask=next_macro.get_canvas_mask(next_macro_loc,inp_canvas.shape)
    res = inp_canvas - mask
    return res

def render(canvas:np.ndarray) -> np.ndarray:
        WHITE=(255,255,255)
        BLACK=(0,0,0)
        RED=(190,0,0)
        img=np.zeros((*(canvas.shape),3),dtype=np.uint8)
        img[canvas==1]=WHITE
        img[canvas==0]=BLACK
        img[canvas==-1]=RED

        return img