# python -m  tests.test_rendering 
from envisionhgdetector import render_pose_video

def main():
    video_path = "C:/Users/babaj/Desktop/zipped/ted_kid.mp4"
    output_path = "C:/Users/babaj/Desktop/zipped/envisionhgdetector/tests/test.mp4"
    logo_path = "C:/Users/babaj/Desktop/zipped/envisionbox_logo.png" # Optional: Path to a logo image
    render_pose_video(
        video_path, output_path, 
        logo_path, logo_position= "top-right",
        only_skeleton= False,
        make_gif= True, gif_start= 0, gif_end= 5, gif_fps= 10,
    )

if __name__ == "__main__":
    main()