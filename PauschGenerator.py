from PauschBaseEffects import *
import numpy as np
import cv2 as cv



class AVIGenerator:
    def __init__(self, filename, frame_rate=FRAME_RATE):
        self.frame_rate = frame_rate
        self.filename = filename + '.avi'
        
        self.effects = []
    
    
    def add_effect(self, effect: Effect):
        self.effects.append(effect)
        
        
    def _generate_frames(self, frame_count=0):
        frames = np.zeros((frame_count, BRIDGE_HEIGHT, BRIDGE_WIDTH, 3), dtype=np.uint8)
        
        for effect in self.effects:
            frames = effect.generate_frames(frames)
        
        return frames
    
    
    def save_video(self):
        codec_code = cv.VideoWriter.fourcc(*'png ')
        out = cv.VideoWriter(self.filename, codec_code, self.frame_rate, (BRIDGE_WIDTH, BRIDGE_HEIGHT))
        
        frames = self._generate_frames()
        for frame in frames:
            out.write(cv.cvtColor(np.uint8(frame), cv.COLOR_RGB2BGR))
            
        out.release()