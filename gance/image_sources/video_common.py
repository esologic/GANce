"""
Common functionality for dealing with video files.
"""

import itertools
import pprint
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterator, List, NamedTuple, Optional, cast

import ffmpeg
import more_itertools
import numpy as np
from cv2 import cv2
from ffmpeg.nodes import FilterableStream
from vidgear.gears import WriteGear

from gance import divisor
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.image_sources.image_sources_common import ImageResolution, image_resolution
from gance.iterator_common import first_item_from_iterator
from gance.logger_common import LOGGER

_DEFAULT_WINDOW_NAME = "Frames"


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
            acodec="flac",
            audio_bitrate=200,
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


class VideoOutputController(NamedTuple):
    """
    Interface for something to write videos with.
    Mimics `cv2.VideoWriter`.
    """

    # Adds a frame to the video.
    write: Callable[[RGBInt8ImageType], None]

    # Finalizes the video, you should be able to open and read the video after calling this.
    release: Callable[[], None]


def _create_video_writer_resolution(
    video_path: Path, video_fps: float, resolution: ImageResolution, high_quality: bool
) -> VideoOutputController:
    """
    Create a video writer of a given FPS and resolution.
    :param video_path: Resulting file path.
    :param video_fps: FPS of the video.
    :param resolution: Size of the resulting video.
    :param high_quality: If true, `ffmpeg` will be invoked directly via the VidGears library,
    this will create a much larger, much higher quality output.
    :return: The writer.
    """

    if high_quality:
        # Tries to best match YouTube's compression.
        # See: https://video.stackexchange.com/a/24481
        output_params = {
            "-input_framerate": video_fps,
            "-vf": f"yadif,scale={resolution.width}:{resolution.height}",
            "-vcodec": "libx264",
            "-crf": 18,
            "-bf": 2,
            "-use_editlist": 0,
            "-movflags": "+faststart",
            "-pix_fmt": "yuv422p",
        }
        ffmpeg_writer = WriteGear(
            output_filename=str(video_path), compression_mode=True, **output_params
        )

        def write_frame(image: RGBInt8ImageType) -> None:
            """
            Helper function to satisfy mypy.
            :param image: To write.
            :return: None
            """
            try:
                ffmpeg_writer.write(image)
            except Exception as e:
                LOGGER.exception(f"Ran into problem writing frame: {image}")
                raise e

        return VideoOutputController(
            write=write_frame,
            release=ffmpeg_writer.close,
        )

    else:
        opencv_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (resolution.width, resolution.height),
        )

        def frame_writer(image: RGBInt8ImageType) -> None:
            """
            Helper function to satisfy mypy.
            Adds a check to make sure the input image matches the expected
            resolution of the output. If this doesn't happen, openCV will
            silently create a video without frames.
            :param image: To write.
            :return: None
            """
            if image_resolution(image) != resolution:
                raise ValueError("Incoming frame did not match output resolution!")
            opencv_writer.write(image)

        return VideoOutputController(write=frame_writer, release=opencv_writer.release)


def create_video_writer(
    video_path: Path,
    video_fps: float,
    video_height: int,
    num_squares_width: int,
    num_squares_height: int = 1,
    high_quality: bool = False,
) -> VideoOutputController:
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
    :param high_quality: Will be forwarded to library function.
    :return: The openCV `VideoWriter` object.
    """

    return _create_video_writer_resolution(
        video_path=video_path,
        video_fps=video_fps,
        resolution=ImageResolution(
            width=video_height * num_squares_width, height=video_height * num_squares_height
        ),
        high_quality=high_quality,
    )


class VideoFrames(NamedTuple):
    """
    Contains metadata about the video, and an iterator that produces the frames.
    """

    original_fps: float
    total_frame_count: int
    original_resolution: ImageResolution
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

        whole = divisor.divide_no_remainder(numerator=original_fps, denominator=new_fps)

        if whole != 1:
            return int(whole)

    return None


def frames_in_video(
    video_path: Path,
    video_fps: Optional[float] = None,
    reduce_fps_to: Optional[float] = None,
    width_height: Optional[ImageResolution] = None,
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

    original_width_height = ImageResolution(
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
                output = (
                    image
                    if not resize
                    else cv2.resize(image, (width_height.width, width_height.height))
                )
                yield output
            else:
                break

    return VideoFrames(
        original_fps=vid_capture.get(cv2.CAP_PROP_FPS),
        original_resolution=original_width_height,
        total_frame_count=int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        frames=itertools.islice(frames(), None, None, take_every),
    )


def write_source_to_disk_forward(
    source: ImageSourceType,
    video_path: Path,
    video_fps: float,
    audio_paths: Optional[List[Path]] = None,
    high_quality: bool = False,
) -> ImageSourceType:
    """
    Consume an image source, write it out to disk.
    :param source: To write to disk.
    :param video_path: Output video path.
    :param video_fps: Frames/Second of the output video.
    :param audio_paths: If given, the audio files will be written to the output video.
    :param high_quality: Flag will be forwarded to library function.
    :return: None
    """

    def setup_iteration(output_path: Path) -> ImageSourceType:
        """
        Helper function to set up the output and forwarding operation.
        :param output_path: Intermediate video path.
        :return: The frames to yield.
        """

        first_frame, frame_source = first_item_from_iterator(source)

        writer = _create_video_writer_resolution(
            video_path=output_path,
            video_fps=video_fps,
            resolution=image_resolution(first_frame),
            high_quality=high_quality,
        )

        def write_frame(frame: RGBInt8ImageType) -> None:
            """
            Write the given frame to the file.
            :param frame: To write.
            :return: None
            """
            writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

        for index, image in enumerate(frame_source):
            LOGGER.info(f"Writing frame #{index} to file: {video_path}")
            write_frame(image)
            yield image

        writer.release()

    try:
        if audio_paths is None:
            yield from setup_iteration(video_path)
        else:
            # Don't write the video-only file to disk at the output path, instead
            with NamedTemporaryFile(suffix=video_path.suffix) as file:
                temp_video_path = Path(file.name)
                yield from setup_iteration(temp_video_path)
                file.flush()
                LOGGER.info(f"Finalizing {video_path}")
                add_wavs_to_video(
                    video_path=temp_video_path, audio_paths=audio_paths, output_path=video_path
                )
    except Exception as e:
        LOGGER.exception(f"Ran into an exception writing out a video: {pprint.pformat(e)}")
        raise e


def write_source_to_disk_consume(
    source: ImageSourceType,
    video_path: Path,
    video_fps: float,
    audio_paths: Optional[List[Path]] = None,
    high_quality: bool = False,
) -> None:
    """
    Consume an image source, write it out to disk.
    :param source: To write to disk.
    :param video_path: Output video path.
    :param video_fps: FPS of the output video.
    :param audio_paths: If given, the audio file at this path will be written to the output video.
    :param high_quality: Flag will be forwarded to library function.
    :return: None
    """

    more_itertools.consume(
        write_source_to_disk_forward(
            source=source,
            video_path=video_path,
            video_fps=video_fps,
            audio_paths=audio_paths,
            high_quality=high_quality,
        )
    )


def resize_image(
    image: RGBInt8ImageType, resolution: ImageResolution, delete: bool = False
) -> RGBInt8ImageType:
    """
    Resizes an image to the input resolution.
    Uses, `cv2.INTER_CUBIC`, which is visually good-looking but somewhat slow.
    May want to be able to pass this in.
    :param image: To scale.
    :param resolution: Output resolution.
    :param delete: If true, `del` will be used on `image` to force it's memory to be released.
    :return: Scaled image.
    """

    output: RGBInt8ImageType = cv2.resize(
        image, (resolution.height, resolution.width), interpolation=cv2.INTER_CUBIC
    )

    # The image in `source` has now been 'consumed', and can't be used again.
    # We delete this frame here to avoid memory leaks.
    # Not really sure if this is needed, but it shouldn't cause harm.
    if delete:
        del image

    # The scaled image.
    return output


def resize_source(source: ImageSourceType, resolution: ImageResolution) -> ImageSourceType:
    """
    For each item in the input source, scale it to the given resolution.
    :param source: Contains Images.
    :param resolution: Desired resolution.
    :return: A new source of scaled images.
    """

    yield from (
        resize_image(image, resolution, delete=True)
        if image_resolution(image) != resolution
        else image
        for image in source
    )


def scale_square_source_duplicate(
    source: ImageSourceType, output_side_length: int, frame_multiplier: int = 1
) -> ImageSourceType:
    """
    Scale the resolution and number of frames in a given source.
    :param source: To scale.
    :param output_side_length: Square frames will be resized to this side length.
    :param frame_multiplier: Every frame will be duplicated this many times.
    :return: Scaled source.
    """

    resized = resize_source(source, ImageResolution(output_side_length, output_side_length))

    return (
        cast(
            ImageSourceType,
            more_itertools.repeat_each(
                resized,
                frame_multiplier,
            ),
        )
        if frame_multiplier != 1
        else resized
    )


def display_frame_forward_opencv(
    source: ImageSourceType,
    window_name: str = _DEFAULT_WINDOW_NAME,
    display_resolution: Optional[ImageResolution] = None,
    full_screen: bool = False,
) -> ImageSourceType:
    """
    Displays the the images in `source`, and forwards the image so they can be consumed again.
    Uses an openCV `imshow` to reveal the image.
    TODO: better docs.
    :param source: To display.
    :param window_name: Name of the window.
    :param display_resolution: Change the input frames to this resolution before displaying. Doesn't
    modify the output frames.
    :param full_screen: If True, images will be displayed fullscreen, windowed if otherwise.
    :return: Forwarded iterator, `source`.
    """

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_GUI_NORMAL if full_screen else cv2.WINDOW_AUTOSIZE,
    )

    if full_screen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def display_frame(frame: RGBInt8ImageType) -> RGBInt8ImageType:
        """
        Helper function, actually show the image.
        :param frame: To display.
        :return: `frame`.
        """
        cv2.imshow(
            window_name,
            cv2.cvtColor(
                resize_image(frame, display_resolution)
                if display_resolution is not None
                else frame,
                cv2.COLOR_RGB2BGR,
            ),
        )
        cv2.waitKey(1)
        return frame

    # yield from is used so we can run things after the input has been exhausted to cleanup.
    yield from map(display_frame, source)

    cv2.destroyWindow(window_name)
