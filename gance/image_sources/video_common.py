"""
Common functionality for dealing with video files.
"""

import itertools
import math
from pathlib import Path
from typing import Iterable, Iterator, List, NamedTuple, Optional, Tuple

import ffmpeg
import numpy as np
from cv2 import cv2
from ffmpeg.nodes import FilterableStream

from gance.gance_types import ImageSourceType, OptionalImageSourceType, RGBInt8ImageType
from gance.image_sources.image_sources_common import ImageResolution, image_resolution
from gance.logger_common import LOGGER


def _write_video(video_path: Path, audio: FilterableStream, output_path: Path) -> None:
    """
    Adds an audio file to a video file. Copied from: https://stackoverflow.com/a/65547166
    :param video_path: Path to the video file.
    :param audio: The ffmpeg representation of the audio.
    :param output_path: The path to write the new file to.
    :return: None
    """
    ffmpeg.run(
        ffmpeg.output(
            ffmpeg.input(str(video_path)).video,  # get only video channel
            audio,  # get only audio channel
            str(output_path),
            vcodec="copy",
            acodec="aac",
            strict="experimental",
        ),
        quiet=True,
        overwrite_output=True,
    )


def _read_wav(audio_path: Path) -> FilterableStream:
    """
    Read an audio file as an ffmpeg stream.
    :param audio_path: Path to the audio file.
    :return: The ffmpeg stream of the audio.
    """
    return ffmpeg.input(str(audio_path)).audio


def add_wav_to_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """
    Adds an audio file to a video file. Copied from: https://stackoverflow.com/a/65547166
    :param video_path: Path to the video file.
    :param audio_path: Path to the audio file.
    :param output_path: The path to write the new file to.
    :return: None
    """
    _write_video(video_path=video_path, audio=_read_wav(audio_path), output_path=output_path)


def add_wavs_to_video(video_path: Path, audio_paths: List[Path], output_path: Path) -> None:
    """
    Adds an audio file to a video file. Copied from: https://stackoverflow.com/a/65547166
    :param video_path: Path to the video file.
    :param audio_paths: Paths to multiple audio files.
    :param output_path: The path to write the new file to.
    :return: None
    """
    _write_video(
        video_path=video_path,
        audio=ffmpeg.concat(*[_read_wav(audio_path) for audio_path in audio_paths], v=0, a=1),
        output_path=output_path,
    )


def _create_video_writer_resolution(
    video_path: Path, video_fps: float, resolution: ImageResolution
) -> cv2.VideoWriter:
    """
    Create a video writer of a given FPS and resolution.
    :param video_path: Resulting file path.
    :param video_fps: FPS of the video.
    :param resolution: Size of the resulting video.
    :return: The writer.
    """

    return cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (resolution.width, resolution.height),
    )


def create_video_writer(
    video_path: Path,
    video_fps: float,
    video_height: int,
    num_squares_width: int,
    num_squares_height: int = 1,
) -> cv2.VideoWriter:
    """
    Helper function to configure the VideoWriter which writes frames to a video file.
    :param video_path: Path to the file on disk.
    :param video_fps: Desired FPS of the video.
    :param video_height: Height of the video in pixels.
    :param num_squares_width: Since each section of the video is a `video_height` x `video_height`
    square this parameter sets the width for the video in pixels, with the number of these squares
    that will be written in each frame.
    :param num_squares_height: Like `num_squares_width`, but for height. Sets the height of
    the video in units of `video_height`.
    :return: The openCV `VideoWriter` object.
    """

    return _create_video_writer_resolution(
        video_path=video_path,
        video_fps=video_fps,
        image_resolution=ImageResolution(
            width=video_height * num_squares_width, height=video_height * num_squares_height
        ),
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


def write_source_to_disk(source: ImageSourceType, video_path: Path, video_fps: float) -> None:
    """
    Consume an image source, write it out to disk.
    :param source: To write to disk.
    :param video_path: Output video path.
    :param video_fps: FPS of the output video.
    :return: None
    """

    first_frame = next(source)

    writer = _create_video_writer_resolution(
        video_path=video_path, video_fps=video_fps, image_resolution=image_resolution(first_frame)
    )

    def write_frame(frame: RGBInt8ImageType) -> None:
        """
        Write the given frame to the file.
        :param frame: To write.
        :return: None
        """
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

    write_frame(first_frame)

    for image in source:
        write_frame(image)

    writer.release()


def horizontal_concat_optional_sources(
    sources: Iterable[OptionalImageSourceType],
) -> ImageSourceType:
    """
    For each frame in each frame source in `sources`, concatenate frames at the same
    index, and emit the result.
    :param sources: An iterable of frame sources.
    :return: An iterator of the newly combined frames.
    """

    yield from (
        cv2.hconcat(list(filter(lambda frame: frame is not None, frames)))
        for frames in zip(*sources)
    )
