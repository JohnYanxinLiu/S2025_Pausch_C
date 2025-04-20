from PauschBaseEffects import *
from PauschGenerator import *
import numpy as np
import cv2 as cv
import random
import math

FRAME_RATE = 30



def subphase_1(PBL, start_time, end_time, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end):
    
    PBL.add_effect(SlowColorTransition(
        start_rgb=(220, 180, 0), end_rgb=(220, 120, 0), 
        start_time=start_time, end_time=start_time+15,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))

    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=start_time,
        end_time=start_time+15,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    
    PBL.add_effect(SolidColor(
        rgb=(220, 120, 0), 
        start_time=start_time+15, 
        end_time=end_time,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.8,
        spawn_rate=0.5,
        sparkles_per_spawn=5,
        start_time=start_time+13,
        end_time=end_time,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    
    p1_wave_total_time = 15
    def p1_wave_fn(t, amplitude):
        theta = (t / p1_wave_total_time) * (math.pi / 2)
        return (1.0-math.sin(theta)) * amplitude
    
    PBL.add_effect(MovingWall(
        color=(0, 0, 180),
        pos_fn=p1_wave_fn,
        start_time=start_time + 15,
        end_time=end_time,
        from_left=False,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))


def subphase_2(PBL, start_time, end_time, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end):
    #phase 1.2: cars at an intersection
    PBL.add_effect(SolidColor(rgb=(0,0,180), start_time=start_time, end_time=end_time)) #background

    # Define bridge width
    bridge_width = bridge_width_end - bridge_width_start
    car_color = (200, 200, 0)
    car_y1 = (bridge_height_end + bridge_height_start) // 2
    car_y2 = bridge_height_end
    car_height = (bridge_height_end - bridge_height_start) // 2
    car_width = (bridge_width_end - bridge_width_start)//24

    random.seed(42)

    # 1. Spawn 5–7 random cars before the main sequence
    num_random_cars = random.randint(5, 7)
    random_start_time = start_time
    random_end_time = start_time + 10
    min_time_spacing = 1.0  # minimum seconds between car start times

    car_start_times = []

    attempts = 0
    max_attempts = 100

    while len(car_start_times) < num_random_cars and attempts < max_attempts:
        attempts += 1

        # Try generating a new start time
        candidate_time = random.uniform(random_start_time, random_end_time - 2)

        # Check if it's far enough from existing start times
        if all(abs(candidate_time - t) >= min_time_spacing for t in car_start_times):
            car_start_times.append(candidate_time)

            # Random direction
            if random.randint(0, 1) == 0:
                start_pos = bridge_width_start
                end_pos = bridge_width_end
            else:
                start_pos = bridge_width_end
                end_pos = bridge_width_start

            duration = random.uniform(4, 6)

            PBL.add_effect(MovingCars(
                start_rgb=car_color,
                end_rgb=car_color,
                start_pos=start_pos,
                end_pos=end_pos,
                height=car_height,
                width=car_width,
                start_time=candidate_time,
                end_time=candidate_time + duration,
                x1=bridge_width_start, x2=bridge_width_end,
                y1=car_y1, y2=car_y2
            ))


    # 2. Main car movement sequence (now with general bridge width)


    PBL.add_effect(MovingCars(  # gates-side car (shorter travel)
        start_rgb=car_color,
        end_rgb=car_color,
        start_pos=bridge_width_start,
        end_pos=bridge_width_start + bridge_width // 3,
        height=car_height,
        width=car_width,
        start_time=start_time + 10,
        end_time=start_time + 15,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=car_y1, y2=car_y2
    ))

    PBL.add_effect(MovingCars(  # purnell-side car (shorter travel)
        start_rgb=car_color,
        end_rgb=car_color,
        start_pos=bridge_width_end,
        end_pos=bridge_width_start + bridge_width * 2 // 3,
        height=car_height,
        width=car_width,
        start_time=start_time + 13,
        end_time=start_time + 17,
        y1=car_y1, y2=car_y2
    ))

    PBL.add_effect(MovingCars(  # gates car pause
        start_rgb=car_color,
        end_rgb=car_color,
        start_pos=bridge_width_start + bridge_width // 3,
        end_pos=bridge_width_start + bridge_width // 3,
        height=car_height,
        width=car_width,
        start_time=start_time + 15,
        end_time=start_time + 19,
        y1=car_y1, y2=car_y2
    ))

    PBL.add_effect(MovingCars(  # purnell car pause
        start_rgb=car_color,
        end_rgb=car_color,
        start_pos=bridge_width_start + bridge_width * 2 // 3,
        end_pos=bridge_width_start + bridge_width * 2 // 3,
        height=car_height,
        width=car_width,
        start_time=start_time + 17,
        end_time=start_time + 20,
        y1=car_y1, y2=car_y2
    ))

    PBL.add_effect(MovingCars(  # gates car continues
        start_rgb=car_color,
        end_rgb=car_color,
        start_pos=bridge_width_start + bridge_width // 3,
        end_pos=bridge_width_end,
        height=car_height,
        width=car_width,
        start_time=start_time + 19,
        end_time=start_time + 26,
        y1=car_y1, y2=car_y2
    ))

    PBL.add_effect(MovingCars(  # purnell car continues
        start_rgb=car_color,
        end_rgb=car_color,
        start_pos=bridge_width_start + bridge_width * 2 // 3,
        end_pos=bridge_width_start,
        height=car_height,
        width=car_width,
        start_time=start_time + 20,
        end_time=end_time,
        y1=car_y1, y2=car_y2
    ))

def subphase_3(PBL, start_time, phase_end_time, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end, start_color=(0, 0, 180)):
    #phase 1.3: modified sunrise; colors can be altered as desired
    PBL.add_effect(SolidColor(
        rgb=start_color, 
        start_time=start_time, 
        end_time=phase_end_time-15,
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
    half_width = (bridge_width_end - bridge_width_start) // 2
    segment_width = half_width // num_segments
    
    # Add the last segment to fill the entire width
    PBL.add_effect(SlowColorTransition(
        start_rgb=start_color, end_rgb=base_rgb,
        start_time=start_time, end_time=start_time + min_duration + num_segments,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end,
    ))
    PBL.add_effect(SolidColor(
        rgb=base_rgb,
        start_time=start_time + min_duration + num_segments,
        end_time=phase_end_time-15,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end,
    ))

    for i in range(num_segments):
        # Coordinates for left and right segments
        left_x1 = max(bridge_width_start, center - (i + 1) * segment_width)
        left_x2 = center - i * segment_width
        right_x1 = center + i * segment_width
        right_x2 = min(bridge_width_end, center + (i + 1) * segment_width)

        # Duration interpolation (shorter in center)
        duration = min_duration + i
        end_time = start_time + duration

        # Brighter color only for center-most segments
        color = center_rgb if i < 2 else base_rgb

        # Add both left and right segments
        for x1, x2 in [(left_x1, left_x2), (right_x1, right_x2)]:
            PBL.add_effect(SlowColorTransition(
                start_rgb=start_color, end_rgb=color,
                start_time=start_time, end_time=end_time,
                x1=x1, x2=x2,
                y1=bridge_height_start, y2=bridge_height_end,
            ))
            PBL.add_effect(SolidColor(
                rgb=color,
                start_time=end_time,
                end_time=end_time + 10,
                x1=x1, x2=x2,
                y1=bridge_height_start, y2=bridge_height_end,
            ))

def subphase_4(PBL, start_time, end_time, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end):
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
        start_time=start_time, end_time=start_time+5,
        x1=bridge_width_start, x2=(bridge_width_end+bridge_width_start)//2,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    PBL.add_effect(SlowColorTransition(
        start_rgb=(0, 0, 0), end_rgb=(180, 180, 180), 
        start_time=start_time, end_time=start_time+5, 
        x1=(bridge_width_end+bridge_width_start)//2, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end))
    PBL.add_effect(SolidColor(
        rgb=(180, 180, 180), 
        start_time=start_time+5, end_time=end_time,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
        ))
    # Add to your moving wall effect
    PBL.add_effect(MovingWall(
        color=(0, 0, 200),
        pos_fn=fighting_fn,
        start_time=start_time+5,
        end_time=end_time,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))

def subphase_5(PBL, start_time, end_time, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end):
    # Rainbow colors
    rainbow_colors = [
        (125, 10, 4),     # Red
        (209, 84, 21),    # Orange
        (235, 227, 19),   # Yellow
        (77, 235, 19),    # Green
        (19, 102, 235),   # Blue
        (64, 22, 217),    # Indigo
        (131, 19, 212)    # Violet
    ]

    # Start with a base wash
    PBL.add_effect(SolidColor(
        rgb=(180, 180, 180),
        start_time=start_time,
        end_time=start_time + 0.1 * (end_time - start_time),  # Quick base flash
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))

    full_height = bridge_height_end - bridge_height_start
    mid_y = (bridge_height_start + bridge_height_end) // 2

    total_colors = len(rainbow_colors)
    chunk_duration = (end_time - start_time) / total_colors

    for i in range(total_colors):
        phase_start = start_time + i * chunk_duration
        phase_mid = phase_start + chunk_duration / 2
        phase_end = phase_start + chunk_duration

        current_color = rainbow_colors[i]
        prev_color = rainbow_colors[i - 1] if i > 0 else (0, 0, 0)

        # Top half fade in
        PBL.add_effect(SlowColorTransition(
            start_rgb=prev_color,
            end_rgb=current_color,
            start_time=phase_start,
            end_time=phase_mid,
            x1=bridge_width_start, x2=bridge_width_end,
            y1=bridge_height_start, y2=mid_y
        ))
        # Top half hold
        PBL.add_effect(SolidColor(
            rgb=current_color,
            start_time=phase_mid,
            end_time=phase_end,
            x1=bridge_width_start, x2=bridge_width_end,
            y1=bridge_height_start, y2=mid_y
        ))

        # Bottom half fade in
        PBL.add_effect(SlowColorTransition(
            start_rgb=prev_color,
            end_rgb=current_color,
            start_time=phase_mid,
            end_time=phase_end,
            x1=bridge_width_start, x2=bridge_width_end,
            y1=mid_y, y2=bridge_height_end
        ))

        # Bottom half hold (optional, could skip to save layering)
        if i < total_colors - 1:
            PBL.add_effect(SolidColor(
                rgb=current_color,
                start_time=phase_end,
                end_time=start_time + (i + 2) * chunk_duration,  # short overlap
                x1=bridge_width_start, x2=bridge_width_end,
                y1=mid_y, y2=bridge_height_end
            ))
        else:
            # Final color fade out to black
            PBL.add_effect(SlowColorTransition(
                start_rgb=current_color,
                end_rgb=(0, 0, 0),
                start_time=phase_end,
                end_time=end_time,
                x1=bridge_width_start, x2=bridge_width_end,
                y1=bridge_height_start, y2=bridge_height_end
            ))
            
    PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.6,
        spawn_rate=0.45,
        sparkles_per_spawn=3,
        start_time=start_time,
        end_time=end_time,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))


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
    p2_offset = 30
    p3_offset = 60
    p4_offset = 90
    p5_offset = 125
    p5_end_offset = 170
    
    
    subphase_1(PBL, start_time+p1_offset, start_time+p2_offset, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end)

    subphase_2(PBL, start_time+p2_offset, start_time+p3_offset, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end)

    subphase_3(PBL, start_time+p3_offset, start_time+p4_offset, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end)

    subphase_4(PBL, start_time+p4_offset, start_time+p5_offset, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end)
    
    subphase_5(PBL, start_time+p5_offset, start_time+p5_end_offset, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end)


def combustion_subphase(PBL, start_time, end_time, bridge_width_start, bridge_width_end, bridge_height_start, bridge_height_end):
    PBL.add_effect(SolidColor(rgb=(0,0,180), start_time=start_time, end_time=start_time+30)) #background
    PBL.add_effect(MovingCars( #gates-side car, moving pre-"glitch"
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=0,
        end_pos=228//4,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=start_time,
        end_time=start_time+5,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start)//2,
        y2=bridge_height_end
    ))
    PBL.add_effect(MovingCars( #gates-side car, "glitch"; im trying to have it move along the top rail but not working; john can u take a look
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228//4,
        end_pos=228//2,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=start_time+5,
        end_time=start_time+10,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=(bridge_height_end-bridge_height_start),
        y2=(bridge_height_end-bridge_height_start)//2
    ))
    PBL.add_effect(MovingCars( #gates-side car, post-glitch
        start_rgb=(200, 200, 0),
        end_rgb=(200, 200, 0),
        start_pos=228//2,
        end_pos=228*3//4,
        height=(bridge_height_end - bridge_height_start)//2,
        width=10,
        start_time=start_time+10,
        end_time=start_time+15,
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
        start_time=start_time + 5,
        end_time=start_time + 15,
        y1=(bridge_height_end - bridge_height_start)//2,
        y2=bridge_height_end
    ))
    for i in range(0,17):
        if i%2==0:
            if i<5:
                PBL.add_effect(MovingCars(start_rgb=(230, 46, 14), #r
                                        end_rgb=(252, 186, 3),
                                        start_pos=171+i*10,
                                        end_pos=171+i*10,
                                        height=BRIDGE_HEIGHT,
                                        width=10,
                                        start_time=start_time+15+0.5*i,
                                        end_time=start_time+15+9
                                ))
            PBL.add_effect(MovingCars(start_rgb=(230, 46, 14),
                                      end_rgb=(252, 186, 3),
                                      start_pos=171-i*10,
                                      end_pos=171-i*10,
                                      height=BRIDGE_HEIGHT,
                                      width=10,
                                      start_time=start_time+15+0.5*i,
                                      end_time=start_time+15+9
            ))
        if i%2!=0:
            if i<5:
                PBL.add_effect(MovingCars(start_rgb=(194, 39, 12), #r
                                        end_rgb=(252, 186, 3),
                                        start_pos=171+i*10,
                                        end_pos=171+i*10,
                                        height=BRIDGE_HEIGHT,
                                        width=10,
                                        start_time=start_time+15+0.5*i,
                                        end_time=start_time+15+9
                                        ))
            PBL.add_effect(MovingCars(start_rgb=(194, 39, 12),
                                      end_rgb=(252, 186, 3),
                                      start_pos=171-i*10,
                                      end_pos=171-i*10,
                                      height=BRIDGE_HEIGHT,
                                      width=10,
                                      start_time=start_time+15+0.5*i,
                                      end_time=start_time+15+9
                                    ))
        PBL.add_effect(SlowColorTransition(start_rgb=(252, 186, 3),
            end_rgb=(255,255,255),
            start_time=start_time+15+9,
            end_time=start_time+15+16,
        ))
        PBL.add_effect(SparkleJitter(
        jitter_val=50,
        radius=2,
        time_to_live=0.5,
        spawn_rate=0.8,
        sparkles_per_spawn=5,
        start_time=start_time+15+11,
        end_time=start_time+15+20,
        x1=bridge_width_start, x2=bridge_width_end,
        y1=bridge_height_start, y2=bridge_height_end
    ))
    PBL.add_effect(SolidColor(rgb=(0,0,0), start_time=start_time+15+20, end_time=start_time+15+30))

def main():
    PBL = AVIGenerator('test_video', frame_rate=30)
    np.random.seed(42)
    
    PHASE1_START_TIME=0
    PHASE2_START_TIME=175
    PHASE3_START_TIME=350
    PHASE4_START_TIME=525
    PHASE5_START_TIME=700
    
    # Phase 1: Order
    make_phase(
        PBL, 
        bridge_width_start=0,
        bridge_width_end=BRIDGE_WIDTH,
        bridge_height_start=0,
        bridge_height_end=BRIDGE_HEIGHT,
        start_time=PHASE1_START_TIME
    )
    
    #Phase 2: Erosion
    make_phase(
        PBL, 
        bridge_width_start=0,
        bridge_width_end=BRIDGE_WIDTH,
        bridge_height_start=0,
        bridge_height_end=BRIDGE_HEIGHT,
        start_time=PHASE2_START_TIME
    )
    PBL.add_effect(FlickerOutEffect(
        radius=4, 
        time_to_live=0.5,
        flicker_duration=0.25,
        spawn_rate=0.1,
        sparkles_per_spawn=2,
        base_rgb=None, 
        start_time=PHASE2_START_TIME,
        end_time=PHASE2_START_TIME+170,
        x1=0, y1=0, 
        x2=BRIDGE_WIDTH, 
        y2=BRIDGE_HEIGHT, 
        frame_rate=FRAME_RATE
    ))
    
    #Phase 3: Fracture
    make_phase(
        PBL, 
        bridge_width_start=0,
        bridge_width_end=BRIDGE_WIDTH,
        bridge_height_start=0,
        bridge_height_end=BRIDGE_HEIGHT,
        start_time=PHASE3_START_TIME
    )
    

    PBL.add_effect(FlickerInvertEffect(
        flicker_prob=0.02,
        min_spacing_sec=2,
        flicker_blinks=4,
        flicker_speed=2,  # frames between blinks
        ttl=0.5,  # seconds to stay inverted
        start_time=PHASE3_START_TIME,
        end_time=PHASE3_START_TIME+170,
    ))
    
    PBL.add_effect(FlickerOutEffect(
        radius=4, 
        time_to_live=0.5,
        flicker_duration=0.25,
        spawn_rate=0.1,
        sparkles_per_spawn=2,
        base_rgb=None, 
        start_time=PHASE3_START_TIME,
        end_time=PHASE3_START_TIME+170,
        x1=0, y1=0, 
        x2=BRIDGE_WIDTH, 
        y2=BRIDGE_HEIGHT, 
        frame_rate=FRAME_RATE
    ))
    
    #Phase 5: Combustion
    
    PHASE4_SUB2_START_TIME=PHASE4_START_TIME+30
    subphase_1(PBL, PHASE4_START_TIME, PHASE4_SUB2_START_TIME, 
               bridge_width_start=0, bridge_width_end=BRIDGE_WIDTH,
               bridge_height_start=0, bridge_height_end=BRIDGE_HEIGHT)

    combustion_subphase(PBL, PHASE4_SUB2_START_TIME, PHASE4_SUB2_START_TIME+45, 0, BRIDGE_WIDTH, 0, BRIDGE_HEIGHT)
    subphase_3(PBL, PHASE4_SUB2_START_TIME+40, PHASE4_SUB2_START_TIME+55,
               bridge_width_start=0, bridge_width_end=BRIDGE_WIDTH,
               bridge_height_start=0, bridge_height_end=BRIDGE_HEIGHT, start_color=(0, 0, 0))
    
    PBL.save_video()
    
    return

if __name__ == '__main__':
    main()