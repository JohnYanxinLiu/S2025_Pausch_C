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
    pbl.faded_jitter_background_gradient(
            start_time=0,
            end_time=10,
            base_rgb=(255,191,0),
            final_rgb=(255, 120, 0),
            jitter=100,
            sparkle_frame_duration=3,
            num_sparkles_per_frame=15,            
        )
    pbl.solid_color((255, 120, 0), start_time=10, end_time=30)
    pbl.wave_paint(
            highlight_rgb=(28, 74, 148),
            end_time=30,
            start_time=10,
            base_rgb=(-1, -1, -1),
            speed=10,               
            width=0.05            
        )
    pbl.solid_color((255,120,0), start_time=30, end_time=50)
    pbl.wave_function_fill(
        highlight_rgb=(0, 0, 255),
        start_time=30,
        end_time=50,
        base_rgb=(-1, -1, -1),  # don't overwrite the background
        )
    pbl.save("previewvideo")





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
