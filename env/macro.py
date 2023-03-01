from __future__ import annotations
import numpy as np
from functools import total_ordering

@total_ordering
class Macro:

    def __init__(self, idx:int=0, height:int=0, width:int=0) -> None:
        self._idx=idx
        self._w=width
        self._h=height
    
    def __eq__(self, __o: Macro) -> bool:
        return __o._idx==self._idx and __o._w==self._w and __o._h==self._h
    
    def __le__(self, __o: Macro) -> bool:
        return self.size <= __o.size
    
    @property
    def idx(self) -> int:
        return self._idx
    @property
    def h(self) -> int:
        return self._h
    @property
    def w(self) -> int:
        return self._w
    @property
    def shape(self) -> np.ndarray:
        return np.array((self._h,self._w))
    @property
    def size(self) -> int:
        return self._h*self._w
    
    def get_canvas_mask(self, pos:tuple[int,int], canvas_size:tuple[int,int]) -> tuple[bool,np.ndarray]:
        ''' Get canvas mask to evaluate legal placement of this macro.
        
        Boolean contains a first legality evaluation based on position on empty canvas
        
        Parameters
        ----------
        pos : (y,x) as tuple[int,int]
            Where you want to place the macro within the canvas
        
        canvas_size : (height, width) as tuple[int,int]
            Size of the floorplan canvas in terms of fixed size units
        '''

        res=np.zeros(canvas_size,dtype=np.int8)
        legal = False

        if pos[0] < 0 or pos[0]+self._h > canvas_size[0] or pos[1] < 0 or pos[1]+self._w > canvas_size[1]:
            clipd_yl, clipd_yh=np.clip([pos[0],pos[0]+self._h],0,canvas_size[0])
            clipd_xl, clipd_xh=np.clip([pos[1],pos[1]+self._w],0,canvas_size[1])
            res[clipd_yl:clipd_yh,clipd_xl:clipd_xh]=1
        else:
            res[pos[0]:pos[0]+self._h,pos[1]:pos[1]+self._w]=1
            legal = True

        return (legal,res.flatten())


def test_main():
    from matplotlib import pyplot as plt

    canvas=(15,10)
    m0=Macro(idx=0,height=3,width=2)
    m1=Macro(idx=1,height=1,width=4)
    m2=Macro(idx=2,height=5,width=3)

    print("m0 is","larger" if m0>m2 else "smaller","than m2")

    m0_valid,m0_mask=m0.get_canvas_mask((0,0),canvas)
    m1_valid,m1_mask=m1.get_canvas_mask((7,5),canvas)
    m2_valid,m2_mask=m2.get_canvas_mask((3,7),canvas)

    if m0_valid:
        m0_mask.resize(canvas)
        print("M0 Valid Mask:")
        plt.figure(1)
        plt.imshow(m0_mask)
        plt.title("M0 Placement")
        plt.show()
    else:
        print("M0 Invalid Mask")
    
    if m1_valid:
        m1_mask.resize(canvas)
        print("M1 Valid Mask:")
        plt.figure(2)
        plt.imshow(m1_mask)
        plt.title("M1 Placement")
        plt.show()
    else:
        print("M1 Invalid Mask")
    
    if m2_valid:
        m2_mask.resize(canvas)
        print("M2 Valid Mask:")
        plt.figure(3)
        plt.imshow(m2_mask)
        plt.title("M2 Placement")
        plt.show()
    else:
        print("M2 Invalid Mask")

if __name__ == "__main__":
    test_main()