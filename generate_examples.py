from PauschBaseEffects import *
from PauschGenerator import *
import numpy as np
import cv2 as cv
import random
import math




def main():
    PBL = AVIGenerator('test_video', frame_rate=30)
    np.random.seed(42)
    
    PBL.add_effect(SlowColorTransition((220, 180, 0), (220, 120, 0), start_time=0, end_time=20))
    
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=0, 
        end_time=20
    ))
    
    PBL.add_effect(SolidColor(rgb=(220, 120, 0), start_time=20, end_time=40))
    
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=20, 
        end_time=40
    ))
    
    p1_wave_total_time = 20
    def p1_wave_fn(t, amplitude):
        theta = (t / p1_wave_total_time) * (math.pi / 2)
        return (1.0-math.sin(theta)) * amplitude
    
    PBL.add_effect(MovingWall(
        color=(0, 0, 180),
        pos_fn=p1_wave_fn,
        start_time=20,
        end_time=40,
        from_left=False
    ))

    PBL.add_effect(SolidColor(rgb=(0,0,180), start_time=40, end_time=60))
    PBL.add_effect(MovingCars(
            start_rgb=(200,200,200),
            end_rgb=(200,200,0), 
            start_pos=0, 
            end_pos=BRIDGE_WIDTH//2, 
            start_time=40, 
            end_time=60, 
            width=20, 
            height=BRIDGE_HEIGHT//2, 
            y1=BRIDGE_HEIGHT//2, 
            y2=BRIDGE_HEIGHT
        ))
    
    PBL.save_video()
        
    return

if __name__ == '__main__':
    main()