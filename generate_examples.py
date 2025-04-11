from PauschBaseEffects import *
from PauschGenerator import *
import numpy as np
import cv2 as cv
import random




def main():
    PBL = AVIGenerator('test_video', frame_rate=30)
    # Add a solid color effect
    PBL.add_effect(SolidColor(rgb=(180, 0, 0), start_time=0, end_time=5))
    
    PBL.add_effect(SlowColorTransition((180, 0, 0), (0, 0, 180), start_time=5, end_time=10))
    
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=0, 
        end_time=10
    ))
    
    PBL.add_effect(MovingWall(
        color=(0, 0, 180),
        start_time=0,
        end_time=10,
        x1=0,
        y1=0,
        x2=BRIDGE_WIDTH,
        y2=BRIDGE_HEIGHT,
    ))
    
    PBL.save_video()
        
    return

if __name__ == '__main__':
    main()