from PauschBridge import PauschBridge, read_palette


def test_wave():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 10)
    pbl.wave((255, 255, 255), 10, speed=40, start_pos=100)
    pbl.save('test_wave')


def test_sparkle():
    pbl = PauschBridge()
    pbl.solid_color((186, 102, 50), 10)
    pbl.wave((23, 9, 16), 10)
    pbl.sparkle((255, 255, 255), 10)
    pbl.save('test_sparkle')



def test_sprite():
    pbl = PauschBridge()
    pbl.solid_color((255, 0, 0), 10)
    pbl.wave((0, 255, 0), 10)
    pbl.sparkle((255, 255, 255), 10)
    pbl.sprite_from_file('sprite_data.yaml', 5)
    pbl.save('test_sprite')


def test_wave_top():
    sky_blue = (255, 208, 65)
    cloud_grey = (237, 237, 237)
    pbl = PauschBridge()
    pbl.solid_color(sky_blue, 5).wave(cloud_grey, 5, slices=pbl.get_top(5))
    pbl.save('top_wave')


def simple_test():
    blue = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)
    cyan = (255, 255, 0)
    pbl = PauschBridge().solid_color(blue, 5)
    pbl += PauschBridge().hue_shift(blue, cyan, 5)
    pbl += PauschBridge().solid_color(blue, 5).wave(white, 5)
    pbl += PauschBridge().sparkle(white, 5, base_rgb=black)
    pbl = PauschBridge().sprite_from_file('sprite_data.yaml', 5)

    pbl.save('test')


def colorblock_test():
    pbl = PauschBridge()
    palette = read_palette('ColorPallate_2022-03-02_09-39-54.csv')
    pbl.color_block(palette, 10)
    pbl.save('test_colorblock')


def region_select_test():
    sky_blue = (255, 208, 65)
    black = (0, 0, 0)
    white = (255, 255, 255)
    pbl = PauschBridge()
    pbl.solid_color(sky_blue, 5).hue_shift(
        black, white, 5, slices=pbl.get_region(5, 40, 80))
    pbl.save('test_region')


def test_gradient():
    yellow = (0, 228, 236)
    sky_blue = (255, 208, 65)
    cloud_grey = (237, 237, 237)

    pbl = PauschBridge()
    pbl.solid_color(cloud_grey, 10)
    pbl.gradient(sky_blue, yellow, 10, slices=pbl.get_region(10, 30, 120))
    pbl.save('test_gradient')

def test_fade_to_color():
    pbl = PauschBridge()
    pbl.solid_color((255, 120, 0), end_time=20)
    pbl.wave_paint(
    highlight_rgb=(28, 74, 148),
    end_time=20,
    start_time=0,
    base_rgb=(-1, -1, -1),
    speed=10,               
    width=0.05            
    )
    pbl.save("wavertol_20s")

def test_fighting_colors():
    pbl = PauschBridge()
    pbl.solid_color((255,120,0), end_time=20)
    pbl.wave_function_fill(
        highlight_rgb=(0, 0, 255),
        start_time=0,
        end_time=20,
        base_rgb=(-1, -1, -1),  # don't overwrite the background
        )
    pbl.save("cosine_wave_fill")

def test_movingblock():
    pbl = PauschBridge()
    pbl.wave_move_block(
        highlight_rgb=(0, 255, 200),
        background_rgb=(-1, -1, -1), #no bg overlay
        start_time=0,
        end_time=10,
        initial_pos=0,
        final_pos=228//4,
        block_width=10,
        block_height= 8 // 2,                # half-height
        vertical_offset= 8//2                # bottom half
    )
    pbl.save("wave_move_block_example")








def test_fading_jitter():
    pbl = PauschBridge()

    # Set a solid amber background (warm, earthy tone)
    amber_rgb = (255, 191, 0)
    # pbl.solid_color(amber_rgb, 10)

    # Add a faded jitter sparkle effect on top with slight variation

    end_time = 10
    start_time = 0
    base_rgb = (255, 191, 0)  # amber
    final_rgb = (255, 120, 0)  # orange-red
    # slices = pbl.get_region(0, 0, 100)  # Apply to the entire region
    jitter = 100
    sparkle_frame_duration = 3
    num_sparkles_per_frame = 15

    pbl.faded_jitter_background_gradient(
            start_time=start_time,
            end_time=end_time,
            base_rgb=base_rgb,
            final_rgb=final_rgb,
            # slices=slices,
            jitter=jitter,
            sparkle_frame_duration=sparkle_frame_duration,
            num_sparkles_per_frame=num_sparkles_per_frame,            
        )

    # Save the result
    pbl.save('p1-test_fading_jitter')
    
def showpreview():
    pbl = PauschBridge()
    #phase 1
    pbl.faded_jitter_background_gradient(
            start_time=0,
            end_time=20,
            base_rgb=(255,191,0),
            final_rgb=(255, 120, 0),
            jitter=100,
            sparkle_frame_duration=3,
            num_sparkles_per_frame=15,            
        )
    pbl.solid_color((255, 120, 0), start_time=20, end_time=40)
    pbl.wave_paint(
            highlight_rgb=(28, 74, 148),
            end_time=40,
            start_time=20,
            base_rgb=(-1, -1, -1),
            speed=5.7,               
            width=0.05            
        )
    #phase 2
    pbl.solid_color((28,74,148), start_time=40, end_time=50)
    pbl.wave_move_block( #car from gates, traveling the first 1/4 of the bridge
            highlight_rgb=(237, 237, 26),
            background_rgb=(-1, -1, -1), #no bg overlay
            start_time=40,
            end_time=50,
            initial_pos=0,
            final_pos=228//4,
            block_width=10,
            block_height= 8,                
            vertical_offset= 8//2                
        )
    pbl.wave_move_block( #car from gates, traveling the second 1/4 of the bridge
            highlight_rgb=(237, 237, 26),
            background_rgb=(-1, -1, -1), #no bg overlay
            start_time=50,
            end_time=60,
            initial_pos=228//4,
            final_pos=228//2,
            block_width=10,
            block_height= 8//2,               
            vertical_offset= 8//2                
        ) 
    pbl.wave_move_block( #car from purnell, traveling the first 1/4 of the bridge
            highlight_rgb=(237, 237, 26),
            background_rgb=(-1, -1, -1), #no bg overlay
            start_time=50,
            end_time=60,
            initial_pos=228,
            final_pos=228*3//4,
            block_width=10,
            block_height= 8//2,               
            vertical_offset= 8//2                
        )
    pbl.wave_move_block( #gates car, staying where it is for those 6 seconds
            highlight_rgb=(237, 237, 26),
            background_rgb=(-1, -1, -1), #no bg overlay
            start_time=60,
            end_time=66,
            initial_pos=228//2,
            final_pos=228//2,
            block_width=10,
            block_height= 8//2,                
            vertical_offset= 8//2               
        )
    pbl.wave_move_block( #purnell car, staying where it is for those 10? seconds
            highlight_rgb=(237, 237, 26),
            background_rgb=(-1, -1, -1), #no bg overlay
            start_time=60,
            end_time=70,
            initial_pos=228*3//4,
            final_pos=228*3//4,
            block_width=10,
            block_height= 8//2,                
            vertical_offset= 8//2                
        )
    pbl.wave_move_block( #this is the gates car, starting to move again
            highlight_rgb=(237, 237, 26),
            background_rgb=(-1, -1, -1), #no bg overlay
            start_time=66,
            end_time=76,
            initial_pos=228//2,
            final_pos=228,
            block_width=10,
            block_height= 8//2,                
            vertical_offset= 8//2                
        )
    pbl.wave_move_block( #this is the purnell car, staying where it is for those 8 seconds
            highlight_rgb=(237, 237, 26),
            background_rgb=(-1, -1, -1), #no bg overlay
            start_time=68,
            end_time=93,
            initial_pos=228*3//4,
            final_pos=0,
            block_width=10,
            block_height= 8//2,                
            vertical_offset= 8//2             
        )
    #phase 3:
    pbl.solid_color((0,0,0),start_time=93,end_time=100)
    pbl.block_move_fade(
                            start_rgb=(0,0,0)
                            end_rgb=(235, 157, 12)
                            background_rgb=(-1,-1,-1)
                            start_time=93
                            end_time=100
                            initial_pos=228//2
                            final_pos=228//2
                            block_width=10
                            block_height=8
                            vertical_offset=0
                        )
    #phase 4 (sort of:)
    pbl.solid_color((255,255,255), start_time=100, end_time=120)
    pbl.wave_function_fill(
        highlight_rgb=(28, 74, 148),
        start_time=100,
        end_time=120,
        base_rgb=(-1, -1, -1),  #no bg overlay
        )
    
    #phase 5:

    
    pbl.save("previewvideo")

def testblockmovefade(): 
    pbl = PauschBridge()
    pbl.block_move_fade(
    start_rgb=(255, 0, 0),          # red
    end_rgb=(0, 0, 255),            # blue
    background_rgb=(0, 0, 0),       # black background
    start_time=0,
    end_time=10,
    initial_pos=228,
    final_pos=0,
    block_width=10,
    block_height=8//2,
    vertical_offset=8//2  # bottom two rows
)
    pbl.save("fade_block_motion")




if __name__ == '__main__':
    # test_wave()
    # test_sparkle()
    # test_sprite()
    # test_wave_top()
    # simple_test()
    # colorblock_test()
    # region_select_test()
    #test_fading_jitter()
    #test_fade_to_color()
    #test_fighting_colors()
    # test_gradient()
    showpreview()
    testblockmovefade()
