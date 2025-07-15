import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tyro
from moviepy.editor import ColorClip, VideoFileClip, clips_array, concatenate_videoclips


def list_mp4_files(directory: Path, wildcard: str) -> List[str]:
    return list(directory.glob(wildcard))


def get_video_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


@dataclass
class Config:
    input_dir: str = "/juno/u/thankyou/autom/diffusion_policy/data/robomimic/datasets/augmented/demo_025trials/2023-10-01/"
    output_path: str = ""
    wildcard: str = "*/*playback*.mp4"
    n_videos_per_mp4: int = 50
    trim_to_shortest: bool = (
        True  # whether to trim all videos (from the front) to the shortest video
    )
    loop: bool = False  # whether to loop the videos to the max duration

    def __post_init__(self):
        if self.output_path == "":
            self.output_path = Path(self.input_dir).joinpath("merged_playback_mp4s.mp4")


def make_video_collage(cfg: Config) -> None:
    """
    Collage videos into a single video containing at most n_videos_per_mp4.
    """
    all_video_paths = list_mp4_files(Path(cfg.input_dir), cfg.wildcard)
    all_video_paths = sorted(all_video_paths)

    print(f"Found {len(all_video_paths)} videos")
    for start_idx in range(0, len(all_video_paths), cfg.n_videos_per_mp4):
        video_paths = all_video_paths[start_idx : start_idx + cfg.n_videos_per_mp4]
        video_clips = [VideoFileClip(str(path)) for path in video_paths]

        # Get the duration of the shortest video
        min_duration = min([video.duration for video in video_clips])

        if cfg.trim_to_shortest:
            # Trim each video to only play the last K frames equivalent to the duration of the shortest video
            video_clips = [
                video.subclip(video.duration - min_duration, video.duration)
                for video in video_clips
            ]

        if cfg.loop:
            max_duration = max([video.duration for video in video_clips])
            final_durations = [max_duration - video.duration for video in video_clips]
            black_clips = [
                ColorClip(size=video_clips[0].size, color=(0, 0, 0)).set_duration(
                    final_duration
                )
                for final_duration in final_durations
            ]
            video_clips = [
                concatenate_videoclips([clip, black_clip])
                for clip, black_clip in zip(video_clips, black_clips)
            ]

        # You can then proceed to use `looped_video_clips` for your collage
        black_clip = ColorClip(size=video_clips[0].size, color=(0, 0, 0)).set_duration(
            min_duration
        )

        # 2. Determine collage dimensions
        num_videos = len(video_clips)
        rows = math.floor(math.sqrt(num_videos))
        cols = math.ceil(num_videos / rows)

        grid = []
        for r in range(rows):
            row_clips = video_clips[r * cols : (r + 1) * cols]
            # If a row is not full, fill with the last frame of the last video
            while len(row_clips) < cols:
                row_clips.append(black_clip)
            grid.append(row_clips)

        # name save path by appending start_idx to just before the extension
        save_path = Path(cfg.output_path)
        save_path = save_path.parent.joinpath(
            f"{save_path.stem}_{start_idx}{save_path.suffix}"
        )

        # Create the video grid and write it to file
        final_clip = clips_array(grid)
        final_clip.write_videofile(str(save_path), codec="libx264", audio_codec="aac")
        print(f"Collage video saved to {save_path}")


if __name__ == "__main__":
    tyro.cli(make_video_collage)
