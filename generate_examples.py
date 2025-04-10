from PauschBaseEffects import *
from PauschGenerator import *
import numpy as np
import cv2 as cv
import random




def main():
    PBL = AVIGenerator('test_video', frame_rate=30)
    # Add a solid color effect
    PBL.add_effect(SolidColor(rgb=(255, 0, 0), start_time=0, end_time=5))
    
    PBL.add_effect(SlowColorTransition((255, 0, 0), (0, 0, 255), start_time=5, end_time=10))
    
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=0, 
        end_time=10
    ))
    
    PBL.save_video()
        
    return

if __name__ == '__main__':
    main()