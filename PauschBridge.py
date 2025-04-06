from email.mime import base
import cv2 as cv
import numpy as np
import os
import pandas as pd
import random as rd
import yaml

from typing import NewType

bridge_width = 228
bridge_height = 8
frame_rate = 30
codec_code = cv.VideoWriter.fourcc(*'png ')
dtype = 'int16'  # used so we can mask with -1, converted to uint8 that opencv expects before writing

RGB = NewType('RGB', tuple[int, int, int])

Indices = NewType(
    'Indices', tuple[tuple[int, int], tuple[int, int], tuple[int, int]])


def read_palette(filename):
    return [parse_tuple(color) for color in pd.read_csv(filename).colors]


def parse_tuple(s, dtype=int):
    s = s.replace('(', '').replace(')', '')
    return tuple(dtype(num) for num in s.split(','))


def parse_field(data, field, optional=False, default=(0, 0), dtype=int):
    ''' parse yaml field into appropriate tuple values
        :param data:        data dictionary
        :param field:       field to access data dictionary from
        :param optional:    [optional] if True, return default value if field not in data
        :param default:     [optional] value to return if optional flag is true
        :param dtype:       [optional] what to cast tuple vals into (default is integer) '''
    if optional and field not in data:
        return default
    return parse_tuple(data[field], dtype)


def parse_sprite_yaml(data, curr_time):
    ''' parses color, position, etc from sprite '''
    params = {}
    params['base_rgb'] = parse_field(data, 'bg_color', True, (-1, -1, -1), int)
    params['highlight_rgb'] = parse_field(data, 'sprite_color')

    for entry in data['positions']:
        params['pos'] = parse_field(entry, 'start')
        params['velocity'] = parse_field(entry, 'velocity', True, dtype=float)
        params['acceleration'] = parse_field(
            entry, 'acceleration', True, dtype=float)
        params['start_time'] = curr_time
        params['end_time'] = params['start_time'] + int(entry['duration'])

        yield params


class PauschFrame:
    def __init__(self):
        self.frame = np.zeros((bridge_height, bridge_width, 3), dtype=dtype)

    def get_base_indices(self):
        return [(0, bridge_height), (0, bridge_width), (0, 3)]

    def get_top(self, indices: Indices = None) -> Indices:
        indices = indices if indices is not None else self.get_base_indices()
        (height_start, height_stop), width, color = indices
        return (height_start, int(height_stop / 2)), width, color

    def get_bottom(self, indices: Indices = None) -> Indices:
        indices = indices if indices is not None else self.get_base_indices()
        (height_start, height_stop), width, color = indices
        return ((height_start - height_stop) / 2, height_stop), width, color

    def get_region(self, start, end, indices: Indices = None) -> Indices:
        indices = indices if indices is not None else self.get_base_indices()
        height, _, color = indices
        return height, (start, end), color

    def set_values(self, indices: Indices, subframe: np.matrix):
        height, width, rgb = [slice(start, stop) for start, stop in indices]

        mask_data = subframe != -1

        self.frame[height, width, rgb] = np.where(
            mask_data > 0, subframe, self.frame[height, width, rgb])


class PauschBridge:
    def __init__(self, num_frames: int = 0):
        self.frames = [PauschFrame() for _ in range(num_frames)]

    def __add__(self, other):
        pbl = PauschBridge()
        pbl.frames = self.frames + other.frames
        return pbl

    def _effect_params(self, start_time: int, end_time: int, slices: list[Indices]):
        ''' boilerplate parameters often needed for any effect methods
            :param start_time:  time (sec) of effect start
            :param end_time:    time (sec) of effect end
            :param slices:      [optional] subset of frame on which the effect takes place
            :return             tuple of start_frame index, end_frame index, and slices '''

        self.add_missing_frames(end_time)
        start_frame = start_time * frame_rate
        end_frame = end_time * frame_rate

        slices = slices if slices is not None else [
            frame.get_base_indices() for frame in self.frames[start_frame:end_frame]]

        return start_frame, end_frame, slices

    def add_missing_frames(self, end_time: int):
        ''' if self.frames is not large enough to incorporate end_time, pad it
            :param end_time: time (sec) to fill self.frames up to'''

        end_index = end_time * frame_rate
        # add missing frames if needed
        if len(self.frames) < end_index:
            self.frames += [PauschFrame()
                            for _ in range(len(self.frames), end_index)]

    def set_values(self, indices: list[Indices], frames: list[np.matrix], start_time, end_time):
        ''' set frame values within the specified timeframe
            :param indices:     subset of frame on which the effect takes place
            :param frames:      frame list to update self.frames, should match size specified by indices
            :param start_time:  time (sec) of effect start
            :param end_time:    time (sec) of effect end '''

        start_frame = start_time * frame_rate
        end_frame = end_time * frame_rate
        for inds, mat, frame in zip(indices, frames, range(start_frame, end_frame)):
            self.frames[frame].set_values(inds, mat)

    def get_top(self, duration, start_time=0):
        ''' gets list of indices specifying the top half of Pausch Bridge only
            :param duration:    time (sec) of effect end 
            :param start_time:  [optional] time (sec) of effect start'''

        self.add_missing_frames(duration - start_time)
        # calculate frame indices
        start_index = start_time * frame_rate
        end_index = duration * frame_rate

        return [frame.get_top() for frame in self.frames[start_index:end_index]]

    def get_bottom(self, duration, start_time=0):
        ''' gets list of indices specifying the bottom half of Pausch Bridge only
            :param duration:    time (sec) of effect end 
            :param start_time:  [optional] time (sec) of effect start'''

        self.add_missing_frames(duration - start_time)
        # calculate frame indices
        start_index = start_time * frame_rate
        end_index = duration * frame_rate

        return [frame.get_bottom() for frame in self.frames[start_index:end_index]]

    def get_region(self, duration, region_start, region_end, start_time=0):
        ''' gets list of indices specifying the bottom half of Pausch Bridge only
            :param duration:    time (sec) of effect end 
            :param start_time:  [optional] time (sec) of effect start'''

        self.add_missing_frames(duration - start_time)
        # calculate frame indices
        start_index = start_time * frame_rate
        end_index = duration * frame_rate

        return [frame.get_region(region_start, region_end) for frame in self.frames[start_index:end_index]]

    def solid_color(self, rgb: RGB, end_time: int, start_time: int = 0, slices: list[Indices] = None):
        ''' effect that displays a solid color on the bridge
            :param rgb:         RGB values of the desired color
            :param end_time:    time (sec) of effect end
            :param start_time:  [optional] time (sec) of effect start, defaults to 0
            :param slices:      [optional] list of the subset of the frame to display effect on, defaults to whole frame'''

        _, _, slices = self._effect_params(start_time, end_time, slices)

        self.set_values(slices, [rgb for _ in slices], start_time, end_time)
        return self

    def hue_shift(self, start_rgb: RGB, end_rgb: RGB, end_time: int, start_time: int = 0, slices: list[Indices] = None):
        ''' effect that displays a gradual (linear) shift from one color to another
            :param start_rgb:   RGB values of the desired starting color
            :param end_rgb:     RGB values of the desired ending color
            :param end_time:    time (sec) of effect end
            :param start_time:  [optional] time (sec) of effect start
            :param slices:      [optional] list of the subset of the frame to display effect on, defaults to whole frame'''
        def rgb_ranges(start_rgb: RGB, end_rgb: RGB, num_frames: int):
            ''' generator for hue shift'''
            ranges = [np.linspace(start, end, num_frames)
                      for start, end in zip(start_rgb, end_rgb)]

            for tup in zip(*ranges):
                yield tup

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        start_frame = start_time * frame_rate
        end_frame = end_time * frame_rate
        num_frames = end_frame - start_frame

        self.set_values(slices, rgb_ranges(
            start_rgb, end_rgb, num_frames), start_time, end_time)

        return self

    def gradient(self, start_rgb: RGB, end_rgb: RGB, end_time: int, start_time: int = 0, slices: list[Indices] = None):
        ''' effect that displays a gradient between two colors (linear) from one side of the bridge to another
            :param start_rgb:   RGB values of the desired starting color
            :param end_rgb:     RGB values of the desired ending color
            :param end_time:    time (sec) of effect end
            :param start_time:  [optional] time (sec) of effect start
            :param slices:      [optional] list of the subset of the frame to display effect on, defaults to whole frame'''
        def rgb_ranges(start_rgb: RGB, end_rgb: RGB, num_frames: int):
            region_start, region_end = slices[0][1]
            gradient_width = region_end - region_start if slices else bridge_width
            ''' generator for hue shift'''
            ranges = [np.linspace(start, end, gradient_width)
                      for start, end in zip(start_rgb, end_rgb)]

            frame = np.zeros([tup[1] - tup[0] for tup in slices[0]])

            for i, tup in enumerate(zip(*ranges)):
                frame[:, i] = tup

            for _ in range(num_frames):
                yield frame

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        start_frame = start_time * frame_rate
        end_frame = end_time * frame_rate
        num_frames = end_frame - start_frame

        self.set_values(slices, rgb_ranges(
            start_rgb, end_rgb, num_frames), start_time, end_time)

        return self

    def sprite_from_file(self, filename: str, end_time: int, start_time: int = 0):
        ''' effect that moves a sprite based on data given from filename
            :param filename:    path to file
            :param end_time:        time (sec) of effect end
            :param start_time:      time (sec) of effect start'''

        # check that file exists
        if not os.path.exists(filename):
            print('filename {} does not exist!'.format(filename))

        # parse actual data
        with open('sprite_data.yaml', 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # each separate entry represents a different sprite
        for sprite_data in data:
            for params in parse_sprite_yaml(sprite_data, start_time):
                self.sprite(**params)
                start_time = params['end_time']

        return self

    def sprite(self, highlight_rgb: RGB, start_time: int, end_time: int, pos: tuple[int, int], velocity: tuple[int, int], acceleration: tuple[int, int], base_rgb: RGB, slices: list[Indices] = None):
        ''' effect that displays a small sprite moving linearly
            :param highlight_rgb:   RGB values of the desired sparkle color
            :param start_time:      time (sec) of effect start
            :param end_time:        time (sec) of effect end
            :param pos:             starting position of small sprite
            :param velocity:        velocity of small sprite (2-d tuple)
            :param base_rgb:        [optional] RGB values of the desired base color
            :param slices:          [optional] list of the subset of the frame to display effect on, defaults to whole frame'''

        def gen_slice(pos: tuple[int, int], size: int = 3, limit: tuple[int, int] = (8, 228)):
            x, y = map(round, pos)
            half = size // 2
            min_x = x - half if x - half >= 0 else 0
            min_y = y - half if y - half >= 0 else 0
            max_x = x + half + 1 if x + half + 1 < limit[0] else limit[0]
            max_y = y + half + 1 if y + half + 1 < limit[1] else limit[1]

            # check if any are outside the frame bounds
            if max_x < 0 or max_y < 0:
                return None, None
            return slice(min_x, max_x), slice(min_y, max_y)

        def gen_sprite_movement(num_frames):
            curr_pos = pos
            curr_vel = velocity
            for _ in range(num_frames):
                frame = np.full((bridge_height, bridge_width, 3),
                                base_rgb, dtype=dtype)

                x, y = gen_slice(curr_pos)
                if x is not None:
                    frame[x, y] = highlight_rgb

                curr_vel = [v + a for v, a in zip(curr_vel, acceleration)]
                curr_pos = [p + v for p, v in zip(curr_pos, curr_vel)]

                yield frame

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        self.set_values(slices, gen_sprite_movement(
            end_frame - start_frame), start_time, end_time)

        return self

    def sparkle(self, highlight_rgb: RGB, end_time: int, start_time: int = 0, base_rgb: RGB = (-1, -1, -1), slices: list[Indices] = None):
        ''' effect that displays sparkles of a desired color on a solid background color
            :param highlight_rgb:   RGB values of the desired sparkle color
            :param end_time:        time (sec) of effect end
            :param start_time:      [optional] time (sec) of effect start
            :param base_rgb:        [optional] RGB values of the desired base color. If not specified, will not overwrite base color
            :param slices:          [optional] list of the subset of the frame to display effect on, defaults to whole frame'''

        def gen_sparkles(num_frames):
            ''' generator frame function for the sparkles'''
            sparkles = {}
            for frame_i in range(num_frames):
                # gen 15 sparkle every 3 frames
                if not frame_i % 3:
                    for _ in range(15):
                        inds = (rd.randrange(bridge_height),
                                rd.randrange(bridge_width))
                        sparkles[inds] = rd.randrange(3, 7)

                frame = np.full((bridge_height, bridge_width, 3),
                                base_rgb, dtype=dtype)
                for (row, col), value in sparkles.items():
                    if not value:
                        continue

                    sparkles[row, col] -= 1
                    frame[row, col, :] = highlight_rgb

                yield frame

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        self.set_values(slices, gen_sparkles(
            end_frame - start_frame), start_time, end_time)

        return self
    
    def faded_jitter_background_gradient(
        self,
        end_time: int,
        start_time: int = 0,
        base_rgb: RGB = (0, 0, 0),
        final_rgb: RGB = (0, 0, 0),
        slices: list[Indices] = None,
        jitter: int = 0,
        sparkle_frame_duration: int = 50,
        num_sparkles_per_frame: int = 15,
        sparkle_spawn_interval: int = 3 # NEW
    ):
        ''' Subtle sparkle effect with a base background color that fades into a final color.

            :param end_time:                 time (sec) when the effect ends
            :param start_time:              [optional] time (sec) when the effect starts
            :param base_rgb:                [optional] starting background color
            :param final_rgb:               final background color to fade into
            :param slices:                  [optional] subset of frame to apply effect on
            :param jitter:                  [optional] max jitter (±value) per RGB channel
            :param sparkle_frame_duration:  [optional] lifespan (frames) of each sparkle
            :param num_sparkles_per_frame:  [optional] new sparkles to generate every sparkle cycle
        '''

        def lerp_rgb(rgb_start, rgb_end, t: float):
            return tuple(int(rgb_start[i] + (rgb_end[i] - rgb_start[i]) * t) for i in range(3))


    

        def gen_jitter_sparkles(num_frames):
            active_sparkles = {}

            for frame_i in range(num_frames):
                t = frame_i / (num_frames - 1)
                current_bg = lerp_rgb(base_rgb, final_rgb, t)

                # Add new sparkles every sparkle_frame_duration frames
                if frame_i % sparkle_spawn_interval == 0:
                    for _ in range(num_sparkles_per_frame):
                        row = rd.randrange(bridge_height)
                        col = rd.randrange(bridge_width)
                        jitter_rgb = tuple(current_bg[i] + rd.randint(-jitter, jitter) for i in range(3))
                        # Ensure the RGB values are within valid range
                        jitter_rgb = tuple(max(0, min(255, val)) for val in jitter_rgb)
                        # Add the sparkle with a TTL
                        active_sparkles[(row, col)] = {
                            "rgb": jitter_rgb,
                            "ttl": sparkle_frame_duration
                        }

                # Build frame
                frame = np.full((bridge_height, bridge_width, 3),
                                current_bg, dtype=dtype)

                expired_keys = []
                for (row, col), sparkle in active_sparkles.items():
                    if sparkle["ttl"] <= 0:
                        expired_keys.append((row, col))
                        continue

                    sparkle["ttl"] -= 1
                    frame[row, col, :] = sparkle["rgb"]

                for key in expired_keys:
                    del active_sparkles[key]

                yield frame

        start_frame, end_frame, slices = self._effect_params(start_time, end_time, slices)

        self.set_values(
            slices,
            gen_jitter_sparkles(end_frame - start_frame),
            start_time,
            end_time
        )

        return self
    def wave_paint(self, highlight_rgb: RGB, end_time: int, start_time: int = 0,
               base_rgb: RGB = (-1, -1, -1), slices: list[Indices] = None,
               width: float = 0.1, speed: int = 30, start_pos=bridge_width):
        ''' effect that paints the bridge with a wave that moves right-to-left and leaves behind color
        :param highlight_rgb: RGB values of the desired wave color
        :param end_time:      time (sec) of effect end
        :param start_time:    [optional] time (sec) of effect start
        :param base_rgb:      [optional] RGB values of the background
        :param slices:        [optional] subset of the bridge to apply effect to
        :param width:         fraction of the bridge width as wave width (e.g., 0.1 for 10%)
        :param speed:         speed of the wave in pixels/sec
        :param start_pos:     [optional] start pos (default: far right)
        '''

        def gen_wave_paint(num_frames, wave_width):
            dims = tuple([end - start for start, end in slices[0]])
            wave_pos = start_pos
            min_wave_reach = bridge_width  # How far left the wave has reached

            for _ in range(num_frames):
                wave_pos -= speed / frame_rate
                wave_index = round(wave_pos)
                wave_start = wave_index
                wave_end = min(wave_index + wave_width, bridge_width)
                min_wave_reach = min(min_wave_reach, wave_start)

                frame = np.full(dims, base_rgb, dtype=dtype)
                frame[:, min_wave_reach:, :] = highlight_rgb
                yield frame

        start_frame, end_frame, slices = self._effect_params(start_time, end_time, slices)
        wave_width = int(width * bridge_width)
        num_frames = end_frame - start_frame

        self.set_values(slices, gen_wave_paint(num_frames, wave_width), start_time, end_time)

        return self

    def wave_function_fill(self, 
                       highlight_rgb: RGB, 
                       end_time: int, 
                       start_time: int = 0,
                       base_rgb: RGB = (-1, -1, -1), 
                       slices: list[Indices] = None,
                       function=None):
        ''' Draws a wave whose front follows a custom function. All LEDs to the right of that front are filled.
        :param highlight_rgb:  RGB value of the wave fill
        :param end_time:       end time of effect (seconds)
        :param start_time:     start time of effect (seconds)
        :param base_rgb:       base/background color, or (-1,-1,-1) to leave unchanged
        :param slices:         optional region of the bridge
        :param function:       a function f(t: float) -> float returning wave x-position at time t (in seconds)
    '''
        def gen_function_wave(num_frames, duration):
            dims = tuple([end - start for start, end in slices[0]])
            times = np.linspace(start_time, end_time, num_frames)

            for t in times:
                x_pos = int(function(t))  # where the wave front is now
                frame = np.full(dims, base_rgb, dtype=dtype)
                if x_pos < bridge_width:
                    frame[:, x_pos:, :] = highlight_rgb
                yield frame

        start_frame, end_frame, slices = self._effect_params(start_time, end_time, slices)
        num_frames = end_frame - start_frame

        # default cosine wave function: oscillates back and forth over full width
        if function is None:
            A = bridge_width//2
            frequency = 0.025  # 1 cycle every 40 seconds
            function=lambda t: A + A* np.cos(2 * np.pi * (4 / 40) * t)

        self.set_values(slices, gen_function_wave(num_frames, end_time - start_time), start_time, end_time)
        return self

    def color_block(self, palette: list[RGB], end_time: int, start_time: int = 0, slices: list[Indices] = None, width: int = 4, speed: int = 30):
        ''' effect that displays a wave of desired color & width on a base color
            :param palette:     list of RGB values to randomly pick from
            :param end_time:    time (sec) of effect end
            :param start_time:  [optional] time (sec) of effect start
            :param base_rgb:    [optional] RGB values of the desired base color. If not specified, will overlay wave on top of existing color in frames
            :param slices:      [optional] list of the subset of the frame to display effect on, defaults to whole frame
            :param width:       desired width of wave in relation to bridge width, i.e. 0.5 means half the bridge width
            :param speed:       desired speed of wave in pixels / second '''

        def gen_color_block(start_frame, end_frame):
            dims = tuple([end - start for start, end in slices[0]])
            # generate the starting frame first
            frame = np.zeros(dims, dtype=dtype)
            prev_color = None
            for pos in range(0, dims[1], width):
                # randomly choose a color and add it to the bridge, ensure it's not the previously generated color
                if prev_color:
                    curr_palette = [p for p in palette if p != prev_color]
                else:
                    curr_palette = palette

                prev_color = rd.choice(curr_palette)
                frame[:, pos:pos+width] = prev_color

            for frame_index in range(end_frame - start_frame):
                if frame_index % speed == 0:  # time to move colors down
                    frame[:, :-width, :] = frame[:, width:, :]
                    prev_color = tuple(frame[-1, -1, :])
                    frame[:, -width:,
                          :] = rd.choice([p for p in palette if p != prev_color])
                yield frame

        start_frame, end_frame, slices = self._effect_params(
            start_time, end_time, slices)

        self.set_values(slices, gen_color_block(
            start_frame, end_frame), start_time, end_time)

        return self
    
    def wave_move_block(self,
                    highlight_rgb: RGB,
                    background_rgb: RGB,
                    start_time: int,
                    end_time: int,
                    initial_pos: int,
                    final_pos: int,
                    block_width: int = 5,
                    block_height: int = bridge_height,
                    vertical_offset: int = 0,
                    slices: list[Indices] = None):
        ''' Moves a solid wave block (no trail) from initial to final x-position over time.
            :param highlight_rgb:    RGB color of the block
            :param background_rgb:   RGB color of background
            :param start_time:       effect start time in seconds
            :param end_time:         effect end time in seconds
            :param initial_pos:      starting x-position of the block
            :param final_pos:        ending x-position of the block
            :param block_width:      width of the block in pixels (x direction)
            :param block_height:     height of the block in pixels (y direction)
            :param vertical_offset:  top offset to start the block vertically
            :param slices:           optional region of the bridge
        '''

        def gen_block_motion(num_frames):
            dims = tuple([end - start for start, end in slices[0]])
            x_positions = np.linspace(initial_pos, final_pos, num_frames)

            for x in x_positions:
                frame = np.full(dims, background_rgb, dtype=dtype)

                x_start = int(max(0, round(x)))
                x_end = int(min(bridge_width, x_start + block_width))

                y_start = vertical_offset
                y_end = min(y_start + block_height, bridge_height)

                frame[y_start:y_end, x_start:x_end, :] = highlight_rgb
                yield frame

        start_frame, end_frame, slices = self._effect_params(start_time, end_time, slices)
        num_frames = end_frame - start_frame

        self.set_values(slices, gen_block_motion(num_frames), start_time, end_time)

        return self

    



    def save(self, basename):
        ''' save frame output to .avi file
            :param basename: base filename (without extension) '''
        filename = basename + '.avi'
        out = cv.VideoWriter(filename, codec_code,
                             frame_rate, (bridge_width, bridge_height))

        for frame in self.frames:
            # Convert from RGB to BGR before writing
            bgr_frame = cv.cvtColor(np.uint8(frame.frame), cv.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()


def full_day_simulation():
    black = (0, 0, 0)
    dark_red = (14, 1, 134)
    yellow = (0, 228, 236)
    sky_blue = (255, 208, 65)
    cloud_grey = (237, 237, 237)
    white = (255, 255, 255)

    pbl = PauschBridge().hue_shift(black, dark_red, 30)
    pbl += PauschBridge().hue_shift(dark_red, yellow, 28)
    pbl += PauschBridge().hue_shift(yellow, sky_blue, 2)
    pbl += PauschBridge().solid_color(sky_blue, 60).wave(cloud_grey,
                                                         60, slices=pbl.get_top(60))
    pbl += PauschBridge().hue_shift(sky_blue, yellow, 2)
    pbl += PauschBridge().hue_shift(yellow, dark_red, 28)
    pbl += PauschBridge().hue_shift(dark_red, black, 30)
    pbl += PauschBridge().sparkle(white, 60, base_rgb=black)
    pbl.save('full_day_simulation')


if __name__ == '__main__':
    full_day_simulation()
