from PauschBaseEffects import *
from PauschGenerator import *
import numpy as np
import cv2 as cv
import random
import math

FRAME_RATE = 30

def make_phase(PBL: AVIGenerator, bridge_width_start, bridge_height_start, bridge_width_end, bridge_height_end, start_time):
    """
    This function creates a sequence of visual effects for the bridge.
    It includes a sunset, a car intersection, and a sunrise.
    """
    # Set the offsets for each phase
    bridge_width_start = bridge_width_start 
    bridge_height_start = bridge_height_start
    bridge_width_end = bridge_width_end
    bridge_height_end = bridge_height_end
    #phase 1: sunset
    p1_offset = 0
    p2_offset = 40
    p3_offset = 115
    p4_offset = 145
    p5_offset = 185
    
    
    PBL.add_effect(SlowColorTransition(
        start_rgb=(220, 180, 0), end_rgb=(220, 120, 0), 
        start_time=p1_offset, end_time=p1_offset+20,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))

    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=p1_offset,
        end_time=p1_offset+20,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    
    PBL.add_effect(SolidColor(rgb=(220, 120, 0), start_time=20, end_time=40))
    
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=p1_offset+20,
        end_time=p2_offset,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    
    p1_wave_total_time = 20
    def p1_wave_fn(t, amplitude):
        theta = (t / p1_wave_total_time) * (math.pi / 2)
        return (1.0-math.sin(theta)) * amplitude
    
    PBL.add_effect(MovingWall(
        color=(0, 0, 180),
        pos_fn=p1_wave_fn,
        start_time=p1_offset + 20,
        end_time=p2_offset,
        from_left=False,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))


#phase 1.2: cars at an intersection
    PBL.add_effect(SolidColor(rgb=(0,0,180), start_time=p2_offset, end_time=p3_offset)) #background
    PBL.add_effect(MovingCars( #gates-side car, moving first half of bridge
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=0,
        end_pos=228//2,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=p2_offset + 10,
        end_time=p2_offset + 30,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2,
        y2=bridge_height_end
    ))
    PBL.add_effect(MovingCars( #purnell-side car, moving first fourth of bridge
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228,
        end_pos=228*3//4,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=p2_offset + 20,
        end_time=p2_offset + 30,
        y1=(bridge_height_end - bridge_height_start)//2,
        y2=bridge_height_end
    ))
    PBL.add_effect(MovingCars( #gates car, standing still for pause
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228//2,
        end_pos=228//2,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=p2_offset + 30,
        end_time=p2_offset + 40,
        y1=(bridge_height_end - bridge_height_start)//2,
        y2=bridge_height_end
    ))
    PBL.add_effect(MovingCars( #purnell car, standing still for pause
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228*3//4,
        end_pos=228*3//4,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=p2_offset + 30,
        end_time=p2_offset + 45,
        y1=(bridge_height_end - bridge_height_start)//2,
        y2=bridge_height_end
    ))
    PBL.add_effect(MovingCars( #gates car continues moving
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228//2,
        end_pos=228,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=p2_offset + 40,
        end_time=p2_offset + 60,
        y1=(bridge_height_end - bridge_height_start)//2,
        y2=bridge_height_end
    ))
    PBL.add_effect(MovingCars( #purnell car continues moving
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228*3//4,
        end_pos=0,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=p2_offset + 45,
        end_time=p3_offset,
        y1=(bridge_height_end - bridge_height_start)//2,
        y2=bridge_height_end
    ))

    #phase 1.3: modified sunrise; colors can be altered as desired
    PBL.add_effect(SolidColor(
        rgb=(0, 0, 0), 
        start_time=p3_offset, 
        end_time=p4_offset-10,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    
    
    # Parameters
    num_segments = 10  # Segments from center to edge (on each side)
    base_rgb = (245, 151, 44)        # Normal color
    center_rgb = (255, 180, 60)      # Brighter center color
    base_duration = 15               # Max duration (outermost)
    min_duration = 8                 # Shortest duration (center-most)

    center = (bridge_width_start + bridge_width_end) // 2
    segment_width = (bridge_width_end - bridge_width_start) // (2 * num_segments)

    for i in range(num_segments):
        # Coordinates for left and right segments
        left_x1 = center - (i + 1) * segment_width
        left_x2 = center - i * segment_width
        right_x1 = center + i * segment_width
        right_x2 = center + (i + 1) * segment_width

        # Duration interpolation (shorter in center)
        duration = min_duration + i
        end_time = p3_offset + duration

        # Brighter color only for center-most segments
        color = center_rgb if i < 2 else base_rgb

        # Add both left and right segments
        PBL.add_effect(SlowColorTransition(
            start_rgb=(0, 0, 0), end_rgb=color,
            start_time=p3_offset, end_time=end_time,
            x1=left_x1, x2=left_x2,
            y1=bridge_height_start, y2=bridge_height_end,
        ))

        PBL.add_effect(SolidColor(
            rgb=color,
            start_time=end_time,
            end_time=end_time+10,
            x1=left_x1, x2=left_x2,
            y1=bridge_height_start, y2=bridge_height_end
        ))
        
        PBL.add_effect(SlowColorTransition(
            start_rgb=(0, 0, 0), end_rgb=color,
            start_time=p3_offset, end_time=end_time,
            x1=right_x1, x2=right_x2,
            y1=bridge_height_start, y2=bridge_height_end,
        ))
        
        PBL.add_effect(SolidColor(
            rgb=color,
            start_time=end_time,
            end_time=end_time+10,
            x1=right_x1, x2=right_x2,
            y1=bridge_height_start, y2=bridge_height_end
        ))

    #phase 1.4: oscillating/fighting for power

    def make_p4_fighting_fn(transition_velocity=0.1, fade_out_start_time=24.0, fade_duration=3.0):
        """
        Returns a function struggling_fight_fn(t, amplitude) that simulates
        a struggling wall being pulled from both sides with unpredictable jitter and rebound.

        This version uses random target centers, spring-like overshoot, and noise.
        """

        # Mutable state for closure
        last_phase = [-1]
        center = [0.5]  # current center
        target_center = [0.5]
        velocity = [0.0]
        transition_start_time = [0.0]

        # Time between shifts in target
        cycle_time = 2.5

        # Spring dynamics parameters
        stiffness = 3.0      # How strongly it pulls toward the target
        damping = 0.5        # How much it resists motion (lower = more bounce)
        noise_amplitude = 0.03  # Jitter to add a struggling/twitching effect

        def struggling_fight_fn(t, amplitude):
            phase = int(t // cycle_time)
            dt = t - transition_start_time[0]

            # On a new phase, pick a new random target center (biased slightly toward edges)
            if phase != last_phase[0]:
                last_phase[0] = phase
                transition_start_time[0] = t
                bias = random.choice([-1, 1]) * random.uniform(0.1, 0.3)
                target_center[0] = np.clip(0.5 + bias, 0.1, 0.9)

            # Spring dynamics: update velocity and position
            force = (target_center[0] - center[0]) * stiffness
            velocity[0] += force
            velocity[0] *= damping
            center[0] += velocity[0] * 0.1  # timestep scaling factor

            # Add noise for chaotic feel
            noise = noise_amplitude * math.sin(t * 10.0 + math.sin(t * 5.0))  # chaotic jitter
            position = (center[0] + noise) * amplitude

            # Fade out near end
            if t >= fade_out_start_time:
                fade_alpha = 1.0 - np.clip((t - fade_out_start_time) / fade_duration, 0.0, 1.0)
                position *= fade_alpha

            return int(np.clip(position, 0, amplitude))

        return struggling_fight_fn



    # Set the speed of center transitions (higher = faster)
    transition_velocity = 0.2

    # Create the position function
    fighting_fn = make_p4_fighting_fn(transition_velocity)

    PBL.add_effect(SlowColorTransition(
        start_rgb=(0, 0, 0), end_rgb=(0, 0, 200),
        start_time=p4_offset, end_time=p4_offset+5,
        x1=bridge_width_start, x2=(bridge_width_end-bridge_width_start)//2,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(0, 0, 0), end_rgb=(180, 180, 180), 
        start_time=p4_offset, end_time=p4_offset+5, 
        x1=(bridge_width_end-bridge_width_start)//2, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end))
    PBL.add_effect(SolidColor(
        rgb=(180, 180, 180), 
        start_time=p4_offset+5, end_time=p5_offset,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
        ))
    # Add to your moving wall effect
    PBL.add_effect(MovingWall(
        color=(0, 0, 200),
        pos_fn=fighting_fn,
        start_time=p4_offset+5,
        end_time=p5_offset-5,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(180, 180, 180), end_rgb=(0, 0, 0), 
        start_time=p5_offset-5, end_time=p5_offset,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))


    #phase 1.5: pseudo rainbow (WIP, it goes backward? im tired to fix but take a crack if u want, also i realize i made this at start_time=170)
    PBL.add_effect(SolidColor(
        rgb=(0, 0, 0), 
        start_time=p5_offset, end_time=p5_offset,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    #red fading
    PBL.add_effect(SlowColorTransition(
        start_rgb=(0, 0, 0), end_rgb=(125, 10, 4), 
        start_time=p5_offset, end_time=p5_offset+5, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SolidColor(
        rgb=(125, 10, 4), 
        start_time=p5_offset+5, end_time=p5_offset+10, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(0, 0, 0), end_rgb=(125, 10, 4), 
        start_time=p5_offset+5, end_time=p5_offset+10, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    # orange fading
    PBL.add_effect(SolidColor(
        rgb=(125, 10, 4), 
        start_time=p5_offset+10, end_time=p5_offset+15, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(125, 10, 4), end_rgb=(209, 84, 21), 
        start_time=p5_offset+10, end_time=p5_offset+15, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SolidColor(
        rgb=(209, 84, 21), 
        start_time=p5_offset+15, end_time=p5_offset+20, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(125, 10, 4), end_rgb=(209, 84, 21), 
        start_time=p5_offset+15, end_time=p5_offset+20, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    # yellow fading
    PBL.add_effect(SolidColor(
        rgb=(209, 84, 21), 
        start_time=p5_offset+20, end_time=p5_offset+25, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(209, 84, 21), end_rgb=(235, 227, 19), 
        start_time=p5_offset+20, end_time=p5_offset+25, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SolidColor(
        rgb=(235, 227, 19), 
        start_time=p5_offset+25, end_time=p5_offset+30, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(209, 84, 21), end_rgb=(235, 227, 19), 
        start_time=p5_offset+25, end_time=p5_offset+30, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    # green fading
    PBL.add_effect(SolidColor(
        rgb=(235, 227, 19), 
        start_time=p5_offset+30, end_time=p5_offset+35, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(235, 227, 19), end_rgb=(77, 235, 19), 
        start_time=p5_offset+30, end_time=p5_offset+35, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SolidColor(
        rgb=(77, 235, 19), 
        start_time=p5_offset+35, end_time=p5_offset+40, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2,
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(235, 227, 19), end_rgb=(77, 235, 19), 
        start_time=p5_offset+35, end_time=p5_offset+40, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    
    # blue fading
    PBL.add_effect(SolidColor(
        rgb=(77, 235, 19), 
        start_time=p5_offset+40, end_time=p5_offset+45, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(77, 235, 19), end_rgb=(19, 102, 235), 
        start_time=p5_offset+40, end_time=p5_offset+45, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2
    ))
    PBL.add_effect(SolidColor(
        rgb=(19, 102, 235), 
        start_time=p5_offset+45, end_time=p5_offset+50, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(77, 235, 19), end_rgb=(19, 102, 235), 
        start_time=p5_offset+45, end_time=p5_offset+50, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    # indigo fading
    PBL.add_effect(SolidColor(
        rgb=(19, 102, 235), 
        start_time=p5_offset+50, end_time=p5_offset+55, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(19, 102, 235), end_rgb=(64, 22, 217), 
        start_time=p5_offset+50, end_time=p5_offset+55, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2
    ))
    PBL.add_effect(SolidColor(
        rgb=(64, 22, 217), 
        start_time=p5_offset+55, 
        end_time=p5_offset+60, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(19, 102, 235), end_rgb=(64, 22, 217), 
        start_time=p5_offset+55, end_time=p5_offset+60, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    # purple fading
    PBL.add_effect(SolidColor(
        rgb=(64, 22, 217), 
        start_time=p5_offset+60, end_time=p5_offset+65, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(64, 22, 217), end_rgb=(131, 19, 212), 
        start_time=p5_offset+60, end_time=p5_offset+65, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2
    ))
    PBL.add_effect(SolidColor(
        rgb=(131, 19, 212), 
        start_time=p5_offset+65, end_time=p5_offset+70,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=(bridge_height_end - bridge_height_start)//2
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(64, 22, 217), end_rgb=(131, 19, 212), 
        start_time=p5_offset+65, end_time=p5_offset+60, 
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2, y2=bridge_height_end
    ))


def main():
    PBL = AVIGenerator('test_video', frame_rate=30)
    np.random.seed(42)
    
    make_phase(
        PBL, 
        bridge_width_start=0,
        bridge_width_end=BRIDGE_WIDTH,
        bridge_height_start=0,
        bridge_height_end=BRIDGE_HEIGHT,
        start_time=0
    )

    PBL.save_video()
    
    return

if __name__ == '__main__':
    main()