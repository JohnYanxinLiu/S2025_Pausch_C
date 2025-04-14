from PauschBaseEffects import *
from PauschGenerator import *
import numpy as np
import cv2 as cv
import random
import math




def main():
    PBL = AVIGenerator('test_video', frame_rate=30)
    np.random.seed(42)
    
    #phase 1: sunset
    P1_START_TIME = 0
    P2_START_TIME = 40
    P3_START_TIME = 115
    P4_START_TIME = 170
    P5_START_TIME = 210
    
    
    PBL.add_effect(SlowColorTransition((220, 180, 0), (220, 120, 0), start_time=P1_START_TIME, end_time=P1_START_TIME+20))

    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=P1_START_TIME,
        end_time=P1_START_TIME+20
    ))
    
    PBL.add_effect(SolidColor(rgb=(220, 120, 0), start_time=20, end_time=40))
    
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=P1_START_TIME+20,
        end_time=P2_START_TIME
    ))
    
    p1_wave_total_time = 20
    def p1_wave_fn(t, amplitude):
        theta = (t / p1_wave_total_time) * (math.pi / 2)
        return (1.0-math.sin(theta)) * amplitude
    
    PBL.add_effect(MovingWall(
        color=(0, 0, 180),
        pos_fn=p1_wave_fn,
        start_time=P1_START_TIME + 20,
        end_time=P2_START_TIME,
        from_left=False
    ))


#phase 1.2: cars at an intersection
    PBL.add_effect(SolidColor(rgb=(0,0,180), start_time=P2_START_TIME, end_time=P3_START_TIME)) #background
    PBL.add_effect(MovingCars( #gates-side car, moving first half of bridge
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=0,
        end_pos=228//2,
        height=BRIDGE_HEIGHT//2,
        width=10,
        start_time=P2_START_TIME + 10,
        end_time=P2_START_TIME + 30,
        y1=BRIDGE_HEIGHT//2,
        y2=BRIDGE_HEIGHT
    ))
    PBL.add_effect(MovingCars( #purnell-side car, moving first fourth of bridge
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228,
        end_pos=228*3//4,
        height=BRIDGE_HEIGHT//2,
        width=10,
        start_time=P2_START_TIME + 20,
        end_time=P2_START_TIME + 30,
        y1=BRIDGE_HEIGHT//2,
        y2=BRIDGE_HEIGHT
    ))
    PBL.add_effect(MovingCars( #gates car, standing still for pause
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228//2,
        end_pos=228//2,
        height=BRIDGE_HEIGHT//2,
        width=10,
        start_time=P2_START_TIME + 30,
        end_time=P2_START_TIME + 40,
        y1=BRIDGE_HEIGHT//2,
        y2=BRIDGE_HEIGHT
    ))
    PBL.add_effect(MovingCars( #purnell car, standing still for pause
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228*3//4,
        end_pos=228*3//4,
        height=BRIDGE_HEIGHT//2,
        width=10,
        start_time=P2_START_TIME + 30,
        end_time=P2_START_TIME + 45,
        y1=BRIDGE_HEIGHT//2,
        y2=BRIDGE_HEIGHT
    ))
    PBL.add_effect(MovingCars( #gates car continues moving
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228//2,
        end_pos=228,
        height=BRIDGE_HEIGHT//2,
        width=10,
        start_time=P2_START_TIME + 40,
        end_time=P2_START_TIME + 60,
        y1=BRIDGE_HEIGHT//2,
        y2=BRIDGE_HEIGHT
    ))
    PBL.add_effect(MovingCars( #purnell car continues moving
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228*3//4,
        end_pos=0,
        height=BRIDGE_HEIGHT//2,
        width=10,
        start_time=P2_START_TIME + 45,
        end_time=P3_START_TIME,
        y1=BRIDGE_HEIGHT//2,
        y2=BRIDGE_HEIGHT
    ))

    #phase 1.3: modified sunrise; colors can be altered as desired
    PBL.add_effect(SolidColor(rgb=(0, 0, 0), start_time=P3_START_TIME, end_time=P4_START_TIME-10))
    centerspot = (228//2) - 9
    PBL.add_effect(MovingCars(start_rgb=(0,0,0),end_rgb=(245,151,44),start_pos=centerspot, end_pos=centerspot, height=BRIDGE_HEIGHT, width=19, start_time=P3_START_TIME, end_time=P3_START_TIME+5 ))
    PBL.add_effect(MovingCars(start_rgb=(245,151,44),end_rgb=(245,151,44),start_pos=centerspot, end_pos=centerspot, height=BRIDGE_HEIGHT, width=19, start_time=P3_START_TIME+5, end_time=P3_START_TIME+40))

    for i in range(1, 7):
        PBL.add_effect(MovingCars(start_rgb=(0,0,0),end_rgb=(245,151,44),start_pos=centerspot+(19*i), end_pos=centerspot+(19*i), height=BRIDGE_HEIGHT, width=19, start_time=P3_START_TIME+(5*i), end_time=P3_START_TIME+5+(5*i) ))
        PBL.add_effect(MovingCars(start_rgb=(245,151,44),end_rgb=(245,151,44),start_pos=centerspot+(19*i), end_pos=centerspot+(19*i), height=BRIDGE_HEIGHT, width=19, start_time=P3_START_TIME+5+(5*i), end_time=P3_START_TIME+40))
        PBL.add_effect(MovingCars(start_rgb=(0,0,0),end_rgb=(245,151,44),start_pos=centerspot-(19*i), end_pos=centerspot-(19*i), height=BRIDGE_HEIGHT, width=19, start_time=P3_START_TIME+(5*i), end_time=P3_START_TIME+5+(5*i) ))
        PBL.add_effect(MovingCars(start_rgb=(245,151,44),end_rgb=(245,151,44),start_pos=centerspot-(19*i), end_pos=centerspot-(19*i), height=BRIDGE_HEIGHT, width=19, start_time=P3_START_TIME+5+(5*i), end_time=P3_START_TIME+40))

    PBL.add_effect(SlowColorTransition(start_rgb=(245,151,44), end_rgb=(252,244,81), start_time=P3_START_TIME+40, end_time=P3_START_TIME+45))
    PBL.add_effect(SlowColorTransition(start_rgb=(252,244,81), end_rgb=(255, 191, 0), start_time=P3_START_TIME+45, end_time=P3_START_TIME+50))
    PBL.add_effect(SlowColorTransition(start_rgb=(255,191,0), end_rgb=(255, 255, 255), start_time=P3_START_TIME+50, end_time=P3_START_TIME+55))


    #phase 1.4: oscillating/fighting for power
    #john, can you add the code you produce for this here? note that phase 1.4 should start at start_time=170

    def make_p4_fighting_fn(transition_velocity=0.1, fade_out_start_time=24.0, fade_duration=3.0):
        """
        Returns a function p4_fighting_fn(t, amplitude) that simulates
        a wall struggling and shifting between oscillation centers with smooth transitions,
        and fades out to 0 near the end.
        
        transition_velocity: rate at which the center moves between phases (0.1 = slow, 1.0 = fast)
        fade_out_start_time: when to begin fading out to zero (in seconds)
        fade_duration: how long the fade-out lasts (in seconds)
        """
        # Persistent state for smooth transitions
        last_phase = [-1]     # mutable container for closure
        center_start = [0.5]  # start center for current transition
        center_end = [0.5]    # target center
        transition_start_time = [0.0]

        # Predefined oscillation centers (in fractional amplitude units)
        centers = [0.5, 1/3, 2/3, 1/4, 3/4]
        cycle_time = 6.0  # seconds per phase

        def p4_fighting_fn(t, amplitude):
            nonlocal transition_velocity

            # Determine the current phase and local time within the phase
            phase = int(t // cycle_time)
            local_t = t % cycle_time

            # If we've just entered a new phase, set up a new transition
            if phase != last_phase[0]:
                last_phase[0] = phase
                center_start[0] = center_end[0]
                center_end[0] = centers[phase % len(centers)]
                transition_start_time[0] = t

            # Compute time since transition started
            dt = t - transition_start_time[0]

            # Smooth transition factor (from 0 to 1)
            center_interp = center_start[0]
            center_diff = center_end[0] - center_start[0]
            if center_diff != 0:
                travel_time = abs(center_diff) / transition_velocity
                alpha = np.clip(dt / travel_time, 0, 1)
                center_interp = (1 - alpha) * center_start[0] + alpha * center_end[0]

            # Oscillate around current interpolated center
            freq = 2.0
            wobble = 0.03 * math.sin(t * 7.0)
            offset = 0.025 * math.sin(freq * local_t) + wobble

            position = (center_interp + offset) * amplitude

            # Fade-out effect near the end
            if t >= fade_out_start_time:
                fade_alpha = 1.0 - np.clip((t - fade_out_start_time) / fade_duration, 0.0, 1.0)
                position *= fade_alpha

            return int(np.clip(position, 0, amplitude))

        return p4_fighting_fn



    # Set the speed of center transitions (higher = faster)
    transition_velocity = 0.2

    # Create the position function
    fighting_fn = make_p4_fighting_fn(transition_velocity)

    PBL.add_effect(SlowColorTransition(start_rgb=(255, 255, 255), end_rgb=(0, 0, 200), start_time=P4_START_TIME, end_time=P4_START_TIME+5, x1=0, x2=BRIDGE_WIDTH//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(255, 255, 255), end_rgb=(180, 180, 180), start_time=P4_START_TIME, end_time=P4_START_TIME+5, x1=BRIDGE_WIDTH//2, x2=BRIDGE_WIDTH))
    PBL.add_effect(SolidColor(rgb=(180, 180, 180), start_time=P4_START_TIME+5, end_time=P5_START_TIME))
    # Add to your moving wall effect
    PBL.add_effect(MovingWall(
        color=(0, 0, 200),
        pos_fn=fighting_fn,
        start_time=P4_START_TIME+5,
        end_time=P5_START_TIME-5,
    ))
    PBL.add_effect(SlowColorTransition(start_rgb=(180, 180, 180), end_rgb=(0, 0, 0), start_time=P5_START_TIME-5, end_time=P5_START_TIME))


    #phase 1.5: pseudo rainbow (WIP, it goes backward? im tired to fix but take a crack if u want, also i realize i made this at start_time=170)
    PBL.add_effect(SolidColor(rgb=(0, 0, 0), start_time=P5_START_TIME, end_time=P5_START_TIME))
    #red fading
    PBL.add_effect(SlowColorTransition(start_rgb=(0, 0, 0), end_rgb=(125, 10, 4), start_time=P5_START_TIME, end_time=P5_START_TIME+5, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SolidColor(rgb=(125, 10, 4), start_time=P5_START_TIME+5, end_time=P5_START_TIME+10, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(0, 0, 0), end_rgb=(125, 10, 4), start_time=P5_START_TIME+5, end_time=P5_START_TIME+10, y2=BRIDGE_HEIGHT//2))
    # orange fading
    PBL.add_effect(SolidColor(rgb=(125, 10, 4), start_time=P5_START_TIME+10, end_time=P5_START_TIME+15, y2=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(125, 10, 4), end_rgb=(209, 84, 21), start_time=P5_START_TIME+10, end_time=P5_START_TIME+15, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SolidColor(rgb=(209, 84, 21), start_time=P5_START_TIME+15, end_time=P5_START_TIME+20, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(125, 10, 4), end_rgb=(209, 84, 21), start_time=P5_START_TIME+15, end_time=P5_START_TIME+20, y2=BRIDGE_HEIGHT//2))
    # yellow fading
    PBL.add_effect(SolidColor(rgb=(209, 84, 21), start_time=P5_START_TIME+20, end_time=P5_START_TIME+25, y2=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(209, 84, 21), end_rgb=(235, 227, 19), start_time=P5_START_TIME+20, end_time=P5_START_TIME+25, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SolidColor(rgb=(235, 227, 19), start_time=P5_START_TIME+25, end_time=P5_START_TIME+30, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(209, 84, 21), end_rgb=(235, 227, 19), start_time=P5_START_TIME+25, end_time=P5_START_TIME+30, y2=BRIDGE_HEIGHT//2))
    # green fading
    PBL.add_effect(SolidColor(rgb=(235, 227, 19), start_time=P5_START_TIME+30, end_time=P5_START_TIME+35, y2=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(235, 227, 19), end_rgb=(77, 235, 19), start_time=P5_START_TIME+30, end_time=P5_START_TIME+35, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SolidColor(rgb=(77, 235, 19), start_time=P5_START_TIME+35, end_time=P5_START_TIME+40, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(235, 227, 19), end_rgb=(77, 235, 19), start_time=P5_START_TIME+35, end_time=P5_START_TIME+40, y2=BRIDGE_HEIGHT//2))
    # blue fading
    PBL.add_effect(SolidColor(rgb=(77, 235, 19), start_time=P5_START_TIME+40, end_time=P5_START_TIME+45, y2=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(77, 235, 19), end_rgb=(19, 102, 235), start_time=P5_START_TIME+40, end_time=P5_START_TIME+45, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SolidColor(rgb=(19, 102, 235), start_time=P5_START_TIME+45, end_time=P5_START_TIME+50, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(77, 235, 19), end_rgb=(19, 102, 235), start_time=P5_START_TIME+45, end_time=P5_START_TIME+50, y2=BRIDGE_HEIGHT//2))
    # indigo fading
    PBL.add_effect(SolidColor(rgb=(19, 102, 235), start_time=P5_START_TIME+50, end_time=P5_START_TIME+55, y2=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(19, 102, 235), end_rgb=(64, 22, 217), start_time=P5_START_TIME+50, end_time=P5_START_TIME+55, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SolidColor(rgb=(64, 22, 217), start_time=P5_START_TIME+55, end_time=P5_START_TIME+60, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(19, 102, 235), end_rgb=(64, 22, 217), start_time=P5_START_TIME+55, end_time=P5_START_TIME+60, y2=BRIDGE_HEIGHT//2))
    # purple fading
    PBL.add_effect(SolidColor(rgb=(64, 22, 217), start_time=P5_START_TIME+60, end_time=P5_START_TIME+65, y2=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(64, 22, 217), end_rgb=(131, 19, 212), start_time=P5_START_TIME+60, end_time=P5_START_TIME+65, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SolidColor(rgb=(131, 19, 212), start_time=P5_START_TIME+65, end_time=P5_START_TIME+70, y1=BRIDGE_HEIGHT//2))
    PBL.add_effect(SlowColorTransition(start_rgb=(64, 22, 217), end_rgb=(131, 19, 212), start_time=P5_START_TIME+65, end_time=P5_START_TIME+60, y2=BRIDGE_HEIGHT//2))

            

    #phase 1.6: flowers blooming
    #im tired rn, so not going to do this rn. also the randomization im a little confused abt ngl

    

    PBL.save_video()
        
    return

if __name__ == '__main__':
    main()