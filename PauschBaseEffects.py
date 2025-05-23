
import numpy as np
import cv2 as cv

BRIDGE_WIDTH = 228
BRIDGE_HEIGHT = 8
FRAME_RATE = 30

# x1 is defined as the leftmost pixel of the region
# y1 is defined as the topmost pixel of the region
# x2 is defined as the rightmost pixel of the region
# y2 is defined as the bottommost pixel of the region

# Each effect will receive an input of "frames." These frames will be the frames that have already been generated.
# The effect will either write over the frames, or if the frames are not long enough to accommodate the effect, it will
# generate new frames and append them to the end of the list.

class Effect:
    def __init__(self, start_time=0, end_time=0, x1=0, y1=0, x2=BRIDGE_WIDTH, y2=BRIDGE_HEIGHT, frame_rate=FRAME_RATE):
        self.start_time = start_time
        self.start_frame = int(start_time * frame_rate)
        self.end_time = end_time
        self.end_frame = int(end_time * frame_rate)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_rate = frame_rate

    def generate_frames(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def extend_frames_(self, frames=np.array([])):
        curr_len = frames.shape[0]
        if curr_len < self.end_frame:
            padding = np.zeros((self.end_frame - curr_len, BRIDGE_HEIGHT, BRIDGE_WIDTH, 3), dtype=np.uint8)
            frames = np.concatenate((frames, padding), axis=0)
        return frames


class SlowColorTransition(Effect):
    def __init__(self, start_rgb, end_rgb, start_time=0, end_time=0, x1=0, y1=0, x2=BRIDGE_WIDTH, y2=BRIDGE_HEIGHT, frame_rate=FRAME_RATE):
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.start_rgb = start_rgb
        self.end_rgb = end_rgb

    def lerp_color(start_color, end_color, t):
        return tuple(int(start + (end - start) * t) for start, end in zip(start_color, end_color))
    
    def generate_frames(self, frames=np.array([]), interpolate_fn=lerp_color):
        frames = self.extend_frames_(frames)

        for i in range(self.start_frame, self.end_frame):
            t = (i - self.start_frame) / (self.end_frame - self.start_frame)
            color = interpolate_fn(self.start_rgb, self.end_rgb, t)
            frames[i][self.y1:self.y2, self.x1:self.x2] = color
        
        return frames


class SolidColor(Effect):
    def __init__(self, rgb, start_time=0, end_time=0, x1=0, y1=0, x2=BRIDGE_WIDTH, y2=BRIDGE_HEIGHT, frame_rate=FRAME_RATE):
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.rgb = rgb


    def generate_frames(self, frames=np.array([])):
        
        frames = self.extend_frames_(frames)
        
        for i in range(self.start_frame, self.end_frame):
            frames[i][self.y1:self.y2, self.x1:self.x2] = self.rgb
        return frames



class SparkleJitter(Effect):
    def __init__(
            self, 
            jitter_val=10, 
            radius=4, 
            time_to_live=0.5,
            spawn_rate=0.1,
            sparkles_per_spawn=2,
            base_rgb=None, 
            start_time=0, 
            end_time=0, 
            x1=0, y1=0, 
            x2=BRIDGE_WIDTH, 
            y2=BRIDGE_HEIGHT, 
            frame_rate=FRAME_RATE
        ):
        
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.base_rgb = base_rgb
        self.jitter_val = jitter_val
        self.radius = radius
        self.time_to_live = time_to_live
        self.frames_to_live = int(time_to_live * frame_rate)
        self.spawn_rate = spawn_rate
        self.sparkles_per_spawn = sparkles_per_spawn
        self.sparkles = []

    min_distance_radius = 4

    def _generate_sparkles(self, frames):
        # Generate sparkles
        sparkle_positions_by_frame = {}

        def _generate_sparkle(self, t):
            # Ensure we only make a sparkle far enough away from other sparkles
            num_attempts = 0
            while num_attempts < 100:
                
                x = np.random.randint(self.x1, self.x2)
                y = np.random.randint(self.y1, self.y2)
                too_close = any(
                    abs(x - sx) < self.min_distance_radius and abs(y - sy) < self.min_distance_radius
                    for sx, sy in sparkle_positions_by_frame.get(t, [])
                )
                if too_close:
                    num_attempts += 1
                    continue
                
                self.sparkles.append((x, y, t))
                
                for i in range(t, t + self.frames_to_live):
                    if i not in sparkle_positions_by_frame:
                        sparkle_positions_by_frame[i] = []
                    sparkle_positions_by_frame[i].append((x, y))
                break
        
        
        for t in range(self.start_frame, self.end_frame - self.frames_to_live):
            if np.random.rand() < self.spawn_rate:
                for _ in range(self.sparkles_per_spawn):
                    _generate_sparkle(self, t)

    def generate_frames(self, frames=np.array([])):
        
        frames = self.extend_frames_(frames)
        
        # If we put a solid color behind, then we need to add the solid color first
        if self.base_rgb is not None:
            for i in range(self.start_frame, self.end_frame):
                frames[i][self.y1:self.y2, self.x1:self.x2] = self.base_rgb
                
        # Generate sparkles
        self._generate_sparkles(frames)
        
        def get_faded_color(start_color, end_color, current_time, ttl):
            # Normalize time and compute peak-based fade intensity
            peak = ttl // 2
            distance_from_peak = abs(current_time - peak)
            fade_intensity = np.sqrt(1.0 - (distance_from_peak / peak))
            fade_intensity = np.clip(fade_intensity, 0.0, 1.0)

            # Interpolate between start and end colors
            interpolated = [
                int((1 - fade_intensity) * s + fade_intensity * e)
                for s, e in zip(start_color, end_color)
            ]

            return tuple(interpolated)
        
        # Draw sparkles
        for x, y, t in self.sparkles:
            color = np.clip(tuple(np.random.randint(-self.jitter_val, self.jitter_val) + int(frames[t][y, x, i]) for i in range(3)), 0, 255).astype(np.uint8)
            starting_color = frames[t][y, x]
            
            for i in range(t, t + self.frames_to_live):
                current_color = get_faded_color(starting_color, color, i - t, self.frames_to_live)
                
                for dx in range(-self.radius, self.radius + 1):
                    for dy in range(-self.radius, self.radius + 1):
                        if dx**2 + dy**2 <= self.radius**2:
                            if 0 <= y + dy < BRIDGE_HEIGHT and 0 <= x + dx < BRIDGE_WIDTH:
                                frames[i][y + dy][x + dx] = current_color
        return frames
    
    


class MovingWall(Effect):
    
    def sin_pos_fn(t, amplitude=10):
        return int(amplitude * np.sin(t / 10) + amplitude)
        
    def __init__(self, color, pos_fn=sin_pos_fn, from_left=True, start_time=0, end_time=0, x1=0, y1=0, x2=BRIDGE_WIDTH, y2=BRIDGE_HEIGHT, frame_rate=FRAME_RATE):
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.color = color
        self.from_left = from_left
        self.pos_fn = pos_fn


    def generate_frames(self, frames=np.array([])):
        frames = self.extend_frames_(frames)
        
        for i in range(self.start_frame, self.end_frame):
            t = (i - self.start_frame)/self.frame_rate
            pos = self.pos_fn(t=t, amplitude=self.x2 - self.x1)
            pos = int(np.clip(pos, self.x1, self.x2))
            
            if self.from_left:
                frames[i][self.y1:self.y2, self.x1:pos] = self.color
            else:
                frames[i][self.y1:self.y2, pos:self.x2] = self.color
        return frames
    

class MovingCars(Effect):
    def __init__(self, start_rgb=(), end_rgb=(), start_pos=0, end_pos=0, height=0, width=0,
                 start_time=0, end_time=0, x1=0, y1=0, x2=BRIDGE_WIDTH, y2=BRIDGE_HEIGHT, frame_rate=FRAME_RATE):
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.start_rgb = np.array(start_rgb, dtype=float)
        self.end_rgb = np.array(end_rgb, dtype=float)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.height = height
        self.width = width

    def generate_frames(self, frames=np.array([])):
        frames = self.extend_frames_(frames)

        total_frames = self.end_frame - self.start_frame
        for i in range(self.start_frame, self.end_frame):
            # import pdb; pdb.set_trace()
            t = (i - self.start_frame) / total_frames

            # Linearly interpolate position
            current_pos = int(self.start_pos + t * (self.end_pos - self.start_pos))

            # Linearly interpolate color
            current_rgb = (1 - t) * self.start_rgb + t * self.end_rgb
            current_rgb = np.clip(current_rgb, 0, 255).astype(np.uint8)

            # Draw the "car" as a rectangle at the current position
            x_start = int(np.clip(current_pos, self.x1, self.x2 - self.width))
            x_end = x_start + self.width
            y_start = int(np.clip(self.y1, 0, BRIDGE_HEIGHT - self.height))
            y_end = y_start + self.height

            frames[i][y_start:y_end, x_start:x_end] = current_rgb

        return frames

class FlickerOutEffect(Effect):
    def __init__(
        self, 
        radius=4, 
        time_to_live=0.5,
        flicker_duration=0.25,
        spawn_rate=0.1,
        sparkles_per_spawn=2,
        base_rgb=None, 
        start_time=0, 
        end_time=0, 
        x1=0, y1=0, 
        x2=BRIDGE_WIDTH, 
        y2=BRIDGE_HEIGHT, 
        frame_rate=FRAME_RATE
    ):
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.base_rgb = base_rgb
        self.radius = radius
        self.time_to_live = time_to_live
        self.frames_to_live = int(time_to_live * frame_rate)
        self.flicker_duration = flicker_duration
        self.frames_to_flicker = int(flicker_duration * frame_rate)
        self.spawn_rate = spawn_rate
        self.sparkles_per_spawn = sparkles_per_spawn
        self.sparkles = []

    min_distance_radius = 4

    def _generate_sparkles(self, frames):
        sparkle_positions_by_frame = {}

        def _generate_sparkle(t):
            num_attempts = 0
            while num_attempts < 100:
                x = np.random.randint(self.x1, self.x2)
                y = np.random.randint(self.y1, self.y2)
                too_close = any(
                    abs(x - sx) < self.min_distance_radius and abs(y - sy) < self.min_distance_radius
                    for sx, sy in sparkle_positions_by_frame.get(t, [])
                )
                if too_close:
                    num_attempts += 1
                    continue
                
                self.sparkles.append((x, y, t))
                
                for i in range(t, t + self.frames_to_live):
                    if i not in sparkle_positions_by_frame:
                        sparkle_positions_by_frame[i] = []
                    sparkle_positions_by_frame[i].append((x, y))
                break
        
        for t in range(self.start_frame, self.end_frame - self.frames_to_live):
            if np.random.rand() < self.spawn_rate:
                for _ in range(self.sparkles_per_spawn):
                    _generate_sparkle(t)

    def generate_frames(self, frames=np.array([])):
        frames = self.extend_frames_(frames)
        
        if self.base_rgb is not None:
            for i in range(self.start_frame, self.end_frame):
                frames[i][self.y1:self.y2, self.x1:self.x2] = self.base_rgb

        self._generate_sparkles(frames)

        for x, y, t in self.sparkles:
            base_color = frames[t][y, x].astype(np.int16)

            for i in range(t, t + self.frames_to_live):
                age = i - t

                if age < self.frames_to_flicker:
                    # Flicker by randomly dimming the base color
                    flicker_factor = np.random.uniform(0.3, 1.0)
                    current_color = tuple((base_color * flicker_factor).astype(np.uint8))
                else:
                    # Smooth linear fade to black
                    decay_progress = (age - self.frames_to_flicker) / (self.frames_to_live - self.frames_to_flicker)
                    current_color = tuple((base_color * (1 - decay_progress)).astype(np.uint8))

                for dx in range(-self.radius, self.radius + 1):
                    for dy in range(-self.radius, self.radius + 1):
                        if dx**2 + dy**2 <= self.radius**2:
                            px, py = x + dx, y + dy
                            if 0 <= py < BRIDGE_HEIGHT and 0 <= px < BRIDGE_WIDTH:
                                frames[i][py][px] = current_color
        return frames

class RainbowTransition(Effect):
    def __init__(
        self, 
        period=1.0, 
        start_time=0, 
        end_time=0, 
        x1=0, y1=0, 
        x2=BRIDGE_WIDTH, y2=BRIDGE_HEIGHT, 
        frame_rate=FRAME_RATE
    ):
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.period = period
        self.frame_rate = frame_rate

        # Define the rainbow color sequence
        self.colors = [
            (255, 0, 0),      # Red
            (255, 165, 0),    # Orange
            (255, 255, 0),    # Yellow
            (0, 255, 0),      # Green
            (0, 255, 255),    # Cyan
            (0, 0, 255),      # Blue
            (255, 0, 255),    # Magenta
            (255, 0, 0),      # Back to Red
        ]

    @staticmethod
    def lerp_color(start_color, end_color, t):
        return tuple(int(start + (end - start) * t) for start, end in zip(start_color, end_color))

    def generate_frames(self, frames=np.array([])):
        frames = self.extend_frames_(frames)

        frames_per_transition = int(self.period * self.frame_rate)
        total_frames = self.end_frame - self.start_frame
        num_transitions = len(self.colors) - 1  # since last goes back to first

        for i in range(total_frames):
            global_frame = self.start_frame + i

            # Where we are in the color cycle
            transition_index = (i // frames_per_transition) % num_transitions
            local_frame = i % frames_per_transition
            t = local_frame / frames_per_transition

            start_color = self.colors[transition_index]
            end_color = self.colors[transition_index + 1]
            color = self.lerp_color(start_color, end_color, t)

            frames[global_frame][self.y1:self.y2, self.x1:self.x2] = color

        return frames

class FlickerInvertEffect(Effect):
    def __init__(
        self,
        flicker_prob=0.01,
        min_spacing_sec=3,
        flicker_blinks=4,
        flicker_speed=2,  # frames between blinks
        ttl=0.5,  # seconds to stay inverted
        start_time=0,
        end_time=0,
        x1=0,
        y1=0,
        x2=BRIDGE_WIDTH,
        y2=BRIDGE_HEIGHT,
        frame_rate=FRAME_RATE,
        seed=None
    ):
        super().__init__(start_time, end_time, x1, y1, x2, y2, frame_rate)
        self.flicker_prob = flicker_prob
        self.min_spacing_frames = int(min_spacing_sec * frame_rate)
        self.flicker_blinks = flicker_blinks
        self.flicker_speed = flicker_speed
        self.ttl_frames = int(ttl * frame_rate)
        self.seed = seed

    def generate_frames(self, frames=np.array([])):
        frames = self.extend_frames_(frames)
        if self.seed is not None:
            np.random.seed(self.seed)

        last_flicker_frame = -self.min_spacing_frames
        i = self.start_frame

        while i < self.end_frame:
            if i - last_flicker_frame >= self.min_spacing_frames and np.random.rand() < self.flicker_prob:
                # Flicker in/out
                for b in range(self.flicker_blinks):
                    flicker_frame = i + b * self.flicker_speed
                    if flicker_frame >= self.end_frame:
                        break
                    if b % 2 == 0:
                        frames[flicker_frame][self.y1:self.y2, self.x1:self.x2] = 255 - frames[flicker_frame][self.y1:self.y2, self.x1:self.x2]
                    else:
                        # revert to original (assume frames were initialized with base color or effect layers before)
                        pass  # do nothing, leaves the previous color untouched

                i += self.flicker_blinks * self.flicker_speed

                # Full invert for TTL
                for j in range(i, min(i + self.ttl_frames, self.end_frame)):
                    frames[j][self.y1:self.y2, self.x1:self.x2] = 255 - frames[j][self.y1:self.y2, self.x1:self.x2]

                last_flicker_frame = i
                i += self.ttl_frames
            else:
                i += 1

        return frames
