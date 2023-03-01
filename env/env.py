from copy import deepcopy
from abc import ABC,abstractmethod
from torch import Tensor
import torch
import numpy as np
try:
    from env.macro import Macro
except ModuleNotFoundError:
    from macro import Macro
from matplotlib import pyplot as plt
from random import shuffle

class Environment(ABC):

    def __init__(self,env_name:str,device:torch.device) -> None:
        self.env_name=env_name
        self._done=False
        self._reward=0
        self._device = device
    
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def take_action(self, action:Tensor) -> None:
        pass

    @abstractmethod
    def render(self, render_mode:str="plt") -> np.ndarray|None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    def reward(self) -> int:
        return self._reward

    @property
    @abstractmethod
    def state(self) -> Tensor:
        pass

    @property
    def done(self) -> bool:
        return self._done

class FloorPlan_2D_Grid(Environment):

    rewards={
        "placed":0,
        "illegal":-100,
        "completed":100
    }

    def __init__(self,canvas_size:tuple[int,int],macros:list[Macro],device: torch.device) -> None:

        super().__init__("2D Grid Floor Plan",device)
        self._unplaced_macros=sorted(macros)
        self._init_unplaced_macros=deepcopy(self._unplaced_macros) #Only to allow reset
        self._canvas_shape=canvas_size
        self._canvas = np.ones(canvas_size,dtype=np.int8).flatten()
    
    @property
    def canvas_shape(self) -> tuple[int,int]:
        return self._canvas_shape
    
    def reset(self, shuffle_macros:bool=False) -> None:
        self._unplaced_macros = deepcopy(self._init_unplaced_macros)
        if shuffle_macros:
            shuffle(self._unplaced_macros)
        self._canvas.fill(1)
        self._done=False
        self._reward=0
    
    def close(self) -> None:
        del self._unplaced_macros
        del self._init_unplaced_macros
        del self._canvas
        self._done=True
    
    def _end_episode(self,good:bool=False) -> None:
        self._done=True
        rew_type="completed" if good else "illegal"
        self._reward=FloorPlan_2D_Grid.rewards[rew_type]
    
    def _check_placement(self) -> bool:
        return not bool((self._canvas<0).any())
    
    def take_action(self, action:Tensor) -> None:
        '''Place the current macro on the given location

        Parameters
        ----------
        action: (y,x) as pytorch Tensor
            Location of the bottom left macro corner
        '''


        current_macro:Macro=self._unplaced_macros.pop()
        macro_pos=tuple(action.cpu().numpy())
        mask_ok, mask=current_macro.get_canvas_mask(macro_pos,self._canvas_shape)

        self._canvas -= mask

        if not (mask_ok and self._check_placement()):

            return self._end_episode(False)

        if len(self._unplaced_macros) > 0:
            self._reward=FloorPlan_2D_Grid.rewards["placed"]
        else:
            return self._end_episode(True)
    
    def render(self, render_mode:str="plt") -> np.ndarray | None:
        WHITE=(255,255,255)
        BLACK=(0,0,0)
        RED=(190,0,0)
        img=np.zeros((*(self._canvas_shape),3),dtype=np.uint8)

        lin_img=img.reshape((-1,3))
        lin_img[self._canvas==1]=WHITE
        lin_img[self._canvas==0]=BLACK
        lin_img[self._canvas==-1]=RED

        if render_mode == "plt":
            plt.imshow(img, origin="lower")
            plt.title("Current Floor Plan")
            plt.show()
            return None
        else:
            return img
    
    @property
    def state(self) -> Tensor:
        if len(self._unplaced_macros) > 0:
            curr_cell=self._unplaced_macros[-1].shape
        else:
            curr_cell=np.array((0,0))
        
        res=np.concatenate([curr_cell,self._canvas])
        t_res=torch.from_numpy(res).to(device=self._device,dtype=torch.float32)
        return t_res
    
    @property
    def obs_sample(self) -> Tensor:
        return self.state.cpu()
    
    @property
    def action_space(self) -> np.ndarray:
        return np.zeros(self._canvas_shape,dtype=np.int8)

def test_main():

    from random import randrange

    CANVAS = (30,30)
    MACROS = [
        Macro(0,5,5), Macro(1,3,2),
        Macro(2,6,7), Macro(3,10,3)
    ]

    device=torch.device("cpu")
    e=FloorPlan_2D_Grid(CANVAS,MACROS,device)
    act_space=e.action_space
    e.render()

    for _ in range(10):
        e.reset()
        while not e.done:
            curr_state=e.state
            print("Current cell shape:",curr_state[0:2].cpu().numpy())
            act=Tensor(
                (
                    randrange(0,act_space.shape[0]),
                    randrange(0,act_space.shape[1])
                )
            ).to(dtype=torch.int32)
            print("Placing it in",act.numpy())
            e.take_action(act)
            if e.done:
                print("Something happened (episode ended)")
            e.render()

if __name__ == "__main__":
    test_main()