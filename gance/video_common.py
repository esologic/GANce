"""
Common functionality for dealing with video files
"""

import itertools
import math
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Tuple

import ffmpeg
import numpy as np
from cv2 import cv2
from PIL import Image

from gance.gance_types import RGBInt8ImageType
from gance.logger_common import LOGGER

PNG = "png"


def add_wav_to_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """
    Adds an audio file to a video file. Copied from: https://stackoverflow.com/a/65547166
    :param video_path: Path to the video file.
    :param audio_path: Path to the audio file.
    :param output_path: The path to write the new file to.
    :return: None
    """
    ffmpeg.run(
        ffmpeg.output(
            ffmpeg.input(str(video_path)).video,  # get only video channel
            ffmpeg.input(str(audio_path)).audio,  # get only audio channel
            str(output_path),
            vcodec="copy",
            acodec="aac",
            strict="experimental",
        ),
        quiet=True,
        overwrite_output=True,
    )


def create_video_writer(
    video_path: Path, video_fps: float, video_height: int, num_squares: int
) -> cv2.VideoWriter:
    """
    Helper function to configure the VideoWriter which writes frames to a video file.
    :param video_path: Path to the file on disk.
    :param video_fps: Desired FPS of the video.
    :param video_height: Height of the video in pixels.
    :param num_squares: Since each section of the video is a `video_height` x `video_height` square
    this parameter sets the width for the video in pixels, with the number of these squares that
    will be written in each frame.
    :return: The openCV `VideoWriter` object.
    """

    return cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (video_height * num_squares, video_height),
    )


class VideoFrames(NamedTuple):
    """
    Contains metadata about the video, and an iterator that produces the frames.
    """

    original_fps: float
    total_frame_count: int
    original_resolution: Tuple[int, int]
    frames: Iterator[RGBInt8ImageType]


def reduce_fps_take_every(original_fps: float, new_fps: float) -> Optional[int]:
    """
    When reducing a video from a high FPS, to a lower FPS, you should take every nth (output)
    frame of the input video.
    TODO: Wording very rough here.
    :param original_fps: Must be higher than `new_fps`.
    :param new_fps: Must go evenly into `original_fps`.
    :return: Take every n frames to get an evenly reduced output video.
    """

    if new_fps is not None:
        frac, whole = math.modf(original_fps / new_fps)
        if frac != 0:
            raise ValueError(f"Cannot evenly get {new_fps} out of {original_fps}.")

        if whole != 1:
            return int(whole)

    return None


def frames_in_video(
    video_path: Path,
    video_fps: Optional[float] = None,
    reduce_fps_to: Optional[float] = None,
    width_height: Optional[Tuple[int, int]] = None,
) -> VideoFrames:
    """
    Creates an interface to read each frame from a video into local memory for
    analysis + manipulation.
    :param video_path: The path to the video file on disk.
    :param video_fps: Can be used to override the actual FPS of the video.
    :param reduce_fps_to: Discards frames such that the frames that are returned are at this
    FPS.
    :param width_height: If given, the output frames will be resized to this resolution.
    :return: An NT containing metadata about the video, and an iterator that produces the frames.
    Frames are in RGB color order.
    :raises: ValueError if the video can't be opened, or the given `reduce_fps_to` is impossible.
    """

    vid_capture = cv2.VideoCapture(str(video_path))

    file_fps = float(vid_capture.get(cv2.CAP_PROP_FPS))

    if video_fps:
        if video_fps != file_fps:
            LOGGER.warning(
                f"Override FPS of: {video_fps} fps "
                f"did not match the fps from the file of: {file_fps} fps. "
                f"Projected frames will not line up exactly."
            )
        fps = video_fps
    else:
        fps = file_fps

    take_every = reduce_fps_take_every(original_fps=fps, new_fps=reduce_fps_to)

    original_width_height = (
        int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    resize = (original_width_height != width_height) if width_height is not None else False

    if not vid_capture.isOpened():
        raise ValueError(f"Couldn't open video file: {video_path}")

    def frames() -> Iterator[RGBInt8ImageType]:
        """
        Read frames off of the video capture until there none left or pulling a frame fails.
        :return: An iterator of frames.
        """
        while vid_capture.isOpened():
            ret, frame = vid_capture.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output = image if not resize else cv2.resize(image, width_height)
                yield output
            else:
                break

    return VideoFrames(
        original_fps=vid_capture.get(cv2.CAP_PROP_FPS),
        original_resolution=original_width_height,
        total_frame_count=int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        frames=itertools.islice(frames(), None, None, take_every),
    )


def read_image(image_path: Path) -> RGBInt8ImageType:
    """
    Read an image from disk into the canonical, in-memory format.
    :param image_path: Path to the image file on disk.
    :return: The image
    """
    # Verified by hand that this cast is valid
    return RGBInt8ImageType(np.asarray(Image.open(str(image_path))))


def write_image(image: RGBInt8ImageType, path: Path) -> None:
    """
    Writes a given image to the path.
    Uses PNG by default.
    :param image: Image in memory.
    :param path: Destination.
    :return: None
    """

    Image.fromarray(image).save(fp=str(path), format=PNG.upper())
