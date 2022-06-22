"""
Projections take up a lot of memory. For example, a 1000 step projection produces 1000 different
(1 x 18 x 512) matrices, along with 1000 (1 x 4 x 4) noises and then 1000 (1024, 1024, 3) images.

Projections also take a lot of wall time to generate. That same 1000 step projection would typically
take around 10 minutes to generate. So, to create a 5 second video at 60 fps of all projections
takes nearly 8 hours.

These two points define the purpose of this module, which is to be able to store these projections
on disk in a safe and portable way, and to be able to read them back into memory efficiently.

There's a memory leak somewhere in the Projector library code. This file allows you to spawn a
child process, only compute the important parts and then return only the important data. Then,
this process is killed and the corresponding memory freed.
"""

import itertools
import multiprocessing
import tempfile
from dataclasses import dataclass
from functools import partial
from multiprocessing import Queue  # pylint: disable=unused-import
from multiprocessing import Process
from pathlib import Path
from queue import Empty
from time import sleep
from typing import (  # pylint: disable=unused-import
    Any,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import h5py
import numpy as np
from dataclasses_json import dataclass_json
from h5py import Group
from typing_extensions import Protocol

from gance.gance_types import RGBInt8ImageType
from gance.hash_file import hash_file
from gance.image_sources.image_sources_common import ImageResolution
from gance.image_sources.video_common import frames_in_video
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import load_network_network, wrap_loaded_network
from gance.process_common import COMPLETE_SENTINEL, cleanup_worker, empty_queue_sentinel
from gance.projection.projection_types import (
    CompleteLatentsType,
    FlattenedNoisesType,
    NoisesShapesType,
    NoisesType,
    TFRecordsType,
    complete_latents_to_matrix,
)
from gance.stylegan2.dataset_tool import TFRecordExporter
from gance.stylegan2.projector import Projector
from gance.stylegan2.training import dataset, misc
from gance.stylegan2.training.dataset import TFRecordDataset

DEFAULT_EXPECTED_TIME_PER_STEP = 60.0

# Intermediate projection results are placed in a queue on their way from the worker process
# back to the main thread. This queue is how they're moved and this is the max number of objects
# that can be stored in that queue. If you start running out of memory decrease this number.
FORWARDING_QUEUE_MAX_SIZE = 75

LATEST_VERSION = 2
TARGET_IMAGES_GROUP_NAME = "target_images"
FINAL_LATENTS_GROUP_NAME = "final_latents"
FINAL_IMAGE_GROUP_NAME = "final_images"
_PER_FRAME_DATASET_GROUP_NAMES = [
    TARGET_IMAGES_GROUP_NAME,
    FINAL_LATENTS_GROUP_NAME,
    FINAL_IMAGE_GROUP_NAME,
]
LATENTS_HISTORIES_GROUP_NAME = "latents_histories"
IMAGES_HISTORIES_GROUP_NAME = "images_histories"
NOISES_HISTORIES_GROUP_NAME = "noises_histories"
_PER_FRAME_SUB_GROUP_GROUP_NAMES = [
    LATENTS_HISTORIES_GROUP_NAME,
    IMAGES_HISTORIES_GROUP_NAME,
    NOISES_HISTORIES_GROUP_NAME,
]
COMPRESSION_LEVEL = 9


@dataclass_json
@dataclass
class ProjectionAttributes:  # pylint: disable=too-many-instance-attributes
    """
    Stores metadata about the projection.
    """

    # The version number of this file. Used to maintain compatibility across versions.
    version_number: int

    # This will only be true if every frame in the input video has been projected.
    # Things can crash etc, and results are written to file as they're available so it's possible
    # to have a partially projected video. However, individual projections cannot be recovered. So
    # if the projection of a still image is interrupted, the progress made on it cannot be
    # recovered.
    complete: bool

    # Information about the projection target.

    # This is going to change! It is maintained to give a future viewer of this projection some
    # context as to how it was created.
    original_target_path: str

    # The resolution the source was captured at.
    original_width_height: Tuple[int, int]

    # The resolution the source was scaled to before projection. If no scaling was needed,
    # this will match `original_resolution`.
    projection_width_height: Tuple[int, int]

    # Hex digest of the target. This can be used to figure out exactly which input was projected.
    target_md5_hash: str

    # The following fields are related to the network used in projection.

    # This is going to change! It is maintained to give a future viewer of this projection some
    # context as to how it was created.
    original_network_path: str

    # Hex digest of the pickled network. This can be used to figure out exactly which network was
    # used for projection.
    network_md5_hash: str

    # The following fields are related to the projection itself

    # The number of times the `.step()` function is called in the projection.
    steps_in_projection: int

    # The noise images themselves have to be flattened because they are inconsistently shaped.
    # This records their shapes so they can be rebuilt.
    # THIS SAYS `np.float` BUT THE ONLY ACCEPTABLE VALUE HERE IS `np.nan`!!
    noises_shapes: Union[NoisesShapesType, float]

    # If latent histories are going to be present in the projection file.
    latents_histories_enabled: bool

    # If noises histories are going to be present in the projection file.
    # Warning! This will make the output file MASSIVE.
    noises_histories_enabled: bool

    # If image histories are going to be present in the projection file.
    # Warning! This will make the output file MASSIVE.
    images_histories_enabled: bool

    # The following fields are only relevant to video sources.

    # The fps the video was recorded at.
    original_fps: Optional[float]

    # Since it takes a long time to generate projections, input fps is often decreased to save
    # time. This records the rate at witch the source video was sampled to create the projection.
    projection_fps: Optional[float]

    # The total frame count in the raw source video.
    original_frame_count: Optional[int]

    # The number of frames in the down-sampled video that was fed to projection (if applicable).
    # This will be the lengths of the iterators in the file.
    projection_frame_count: Optional[int]


class TotalProjectionResult(NamedTuple):
    """
    Contains all of the component parts of a projection. Inputs and outputs.
    """

    # The input to the projection.
    target_image: RGBInt8ImageType

    # The final output image from the projection.
    projected_image: RGBInt8ImageType

    # Exposed for convenience, the final latent vectors of the projection. Was input into the
    # network to produce `projected_image`.
    final_latents: CompleteLatentsType

    # Noises are a complicated data structure and may need to be flattened before saving etc.
    # maintaining their shapes allows a user to rebuild.
    noises_shapes: NoisesShapesType


class IntermediateProcessorFunction(Protocol):
    """
    Defines functions that process the intermediate parts of a projection.
    Implementors could write this data to a file, visualize it, compute stats etc.
    """

    def __call__(
        self,
        step_number: int,
        latents: CompleteLatentsType,
        noises: List[NoisesType],
        images: RGBInt8ImageType,
    ) -> None:
        """
        :param step_number: The current projection step.
        :param latents: Output from `projector.get_dlatents()`. These are the latent codes that are
        fed into the network to produce images that are then evaluated and iterated on.
        :param noises: Output from `projector.get_noises()`, the noise that is used with
        input to shape vectors over time.
        :param images: Output from `projector.get_images()`, the images that are evaluated
        and iterated on.
        :return: None
        """


def _total_projection(
    network_path: Path,
    target_image: RGBInt8ImageType,
    intermediate_processor: IntermediateProcessorFunction,
    start_event: Any,
    num_projection_steps: Optional[int],
) -> TotalProjectionResult:
    """
    Project an image using a given network.
    :param network_path: Path to the network used in projection.
    :param target_image: Image to project.
    :param intermediate_processor: See protocol docs.
    :param num_projection_steps: The desired number of steps in the projection.
    :return: The component parts of the projection.
    """
    try:
        network = load_network_network(network_path, True)
        projector = Projector(num_steps=num_projection_steps)
        projector.set_network(network)  # type: ignore

        start_event.set()

        LOGGER.info("Generating Projection.")

        results: IntermediateResults = make_directory_write_records_call_function(
            image=target_image,
            tfrecords_input_callable=lambda tfrecords: project_process_intermediate(
                target_tfrecords=tfrecords,
                projector=projector,
                intermediate_processor=intermediate_processor,
            ),
        )

        LOGGER.info("Projection complete, synthesizing final image.")

        projected_image = wrap_loaded_network(network=network).create_image_matrix(
            complete_latents_to_matrix(results.final_latents)
        )

        LOGGER.info("Created final image.")

        return TotalProjectionResult(
            target_image=target_image,
            projected_image=projected_image,
            final_latents=results.final_latents,
            noises_shapes=results.noises_shapes,
        )
    except Exception as e:
        start_event.set()
        raise e


def _projector_worker(
    network_path: Path,
    image: RGBInt8ImageType,
    output_queue: "Queue[Union[TotalProjectionResult, Exception, str]]",
    intermediate_processor: IntermediateProcessorFunction,
    num_projection_steps: Optional[int],
    start_event: Any,
    stop_event: Any,
) -> None:
    """
    Meant to be called as a child process so when the underlying works is completed all resources
    are correctly freed.
    :param network_path: The path to the network to complete the projection with.
    :param image: The image to project.
    :param output_queue: Results, exceptions and sentinels are put in this queue to communicate
    with the caller.
    :return: None, results are passed through the queue.
    """

    try:
        projection = _total_projection(
            network_path=network_path,
            target_image=image,
            intermediate_processor=intermediate_processor,
            num_projection_steps=num_projection_steps,
            start_event=start_event,
        )
        LOGGER.info("Projection complete, adding to output queue.")
        output_queue.put(projection)
        LOGGER.info("Projection added to output queue.")
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.exception("Ran into problem with synthesis.")
        output_queue.put(e)

    LOGGER.info("Adding sentinel to output queue.")
    output_queue.put(COMPLETE_SENTINEL)
    stop_event.set()


QueueItem = TypeVar("QueueItem")


def _pull_from_queue(
    worker: Process, queue: "Queue[QueueItem]", error_message: str, timeout: float
) -> QueueItem:
    """
    Helper function. If a timeout occurs, kills the worker and raises an exception.
    :param worker: The process to kill if the `.get()` call times out.
    :param queue: Queue to `.get()` from.
    :param error_message: Exception message.
    :return: The value returned from the queue.
    :param timeout: Amount of time to wait before killing worker.
    :return: The item from the queue.
    """

    try:
        return queue.get(timeout=timeout)
    except Empty as e:
        worker.terminate()
        worker.join()
        raise RuntimeError(error_message) from e


def project_image_in_process(  # pylint: disable=too-many-locals
    network_path: Path,
    image: RGBInt8ImageType,
    intermediate_processor: IntermediateProcessorFunction,
    steps_per_projection: int,
    expected_time_per_step: float = DEFAULT_EXPECTED_TIME_PER_STEP,
) -> TotalProjectionResult:
    """
    Use a network to get a latent projection of a given image. Does all computation in a child
    process to get around memory leak problems.

    I incorrectly assumed that hdf5 files could be written from multiple processes. Originally,
    the function was passed in directly, but this didn't work because writes in this way were
    ignored.

    The input to `intermediate_processor` is passed from within the process through a queue, back
    to this function (the main thread) where the call to `intermediate_processor` is actually made.

    This is kind of deceptive and given more time I'd come up with a more elegant solution.

    :param network_path: Path to the network to do the projection.
    :param image: The image to project.
    :param intermediate_processor: Defines functions that process the intermediate parts of a
    projection. Implementors could write this data to a file, visualize it, compute stats etc.
    :param steps_per_projection: Number of times the projector's `.step()` function will get called.
    :param expected_time_per_step: If a projection step doesn't complete within this
    amount of time, we'll assume it has crashed and raise an exception.
    :return: An NT containing the resulting vector + final projected image.
    :raises RuntimeError: if the process responsible for rendering the projection takes too long.
    In this case we assume it has died.
    """

    stop_event = multiprocessing.Event()
    start_event = multiprocessing.Event()

    output_queue: "Queue[Union[TotalProjectionResult, Exception, str]]" = multiprocessing.Queue()

    # The worker could potentially produce projection steps faster than we can consume them.
    # Having this upper limit prevents the queue from becoming too large and consuming all of
    # system memory.
    forward: "Queue[Tuple[int, CompleteLatentsType, List[NoisesType], RGBInt8ImageType]]" = (
        multiprocessing.Queue(maxsize=FORWARDING_QUEUE_MAX_SIZE)  # pylint: disable=line-too-long
    )

    def queue_putter(
        step_number: int,
        latents: CompleteLatentsType,
        noises: List[NoisesType],
        images: RGBInt8ImageType,
    ) -> None:
        """
        Since `forward` can be shared between processes, this exposes a way to move the
        inputs one at a time between the process and this main thread. Since the function's shape
        is assured via the protocol we can use the same args.
        :param step_number: See protocol.
        :param latents: See protocol.
        :param noises: See protocol.
        :param images: See protocol.
        :return: See protocol.
        """
        LOGGER.debug("Worker put args in forwarding queue.")
        forward.put((step_number, latents, noises, images))

    worker = multiprocessing.Process(
        target=_projector_worker,
        kwargs={
            "network_path": network_path,
            "image": image,
            "output_queue": output_queue,
            "intermediate_processor": queue_putter,
            "num_projection_steps": steps_per_projection,
            "start_event": start_event,
            "stop_event": stop_event,
        },
    )

    worker.start()

    while not start_event.is_set():
        pass

    sleep(1)

    # TODO: there might be a cleaner way to do this, using sentinels etc, but since we know exactly
    # how many times `queue_putter` is going to get called inside the process, we know exactly
    # how many items to pull out of this queue.
    for _ in range(steps_per_projection):

        if stop_event.is_set():
            LOGGER.error("Found stop event earlier than expected.")
            break

        output_step_number, output_latents, output_noises, output_images = _pull_from_queue(
            worker=worker,
            queue=forward,
            timeout=expected_time_per_step,
            error_message="Timeout occurred waiting for intermediate result.",
        )

        LOGGER.debug("Main thread got args from forwarding queue.")

        # Make sure the input function is called in the main thread, not in the process.
        intermediate_processor(
            step_number=output_step_number,
            latents=output_latents,
            noises=output_noises,
            images=output_images,
        )

    LOGGER.debug("Waiting for return value from projection worker.")

    output_or_error: "Union[TotalProjectionResult, Exception, str]" = _pull_from_queue(
        worker=worker,
        queue=output_queue,
        timeout=DEFAULT_EXPECTED_TIME_PER_STEP,
        error_message="Timeout occurred waiting for return value.",
    )

    LOGGER.debug("Got return value from projection worker!")

    empty_queue_sentinel(output_queue)

    try:
        cleanup_worker(process=worker)
    except ValueError as e:
        LOGGER.error("Couldn't clean up projection worker")
        raise e

    if isinstance(output_or_error, Exception):
        raise output_or_error

    # Type was checked previously this conversion is safe.
    return cast(TotalProjectionResult, output_or_error)


def _image_to_tfrecords_directory(records_directory: Path, image: RGBInt8ImageType) -> None:
    """
    Converts a given image to a directory full of `.tfrecords` files.
    :param records_directory: The directory to write the `.tfrecords` files to.
    :param image: The image to convert
    :return: None
    """

    with TFRecordExporter(records_directory, 1) as tfr:  # type:ignore
        order = tfr.choose_shuffled_order()
        for _ in range(order.size):
            transposed = image.transpose([2, 0, 1])  # HWC => CHW
            tfr.add_image(transposed)


class TFRecordDirectoryFunction(Protocol):
    """
    Defines an interface for working with a temporary directory full of `.tfrecords` files. Since
    these files are massive, they need to be removed from disk ASAP. This function enables user
    to work with these files as input while they're still on disk.
    """

    def __call__(self, tfrecords: TFRecordsType) -> Any:
        """
        :param tfrecords: The TFRecords files loaded into memory as done in the original
        `stylegan2` implementation.
        :return: The result of a computation on a directory full of `.tfrecords` files. By the time
        this value has been returned, the directory that was the input to this function will be
        deleted!
        """


def make_directory_write_records_call_function(
    image: RGBInt8ImageType, tfrecords_input_callable: TFRecordDirectoryFunction
) -> Any:
    """
    This function is kind of convoluted because the intermediate `.tfrecords` directories are very
    large. A 10 second 1024x1024@30fps video can quickly become 12GB when converted to `.tfrecords`.
    The idea with this function is to only have one frame's worth of records on disk at a time to
    avoid running out of space.

    * Creates the temp directory to write the `.tfrecords` to.
    * Writes the `.tfrecords` to that directory.
    * Loads these resulting `.tfrecords` files back into memory.
    * Passes the loaded `.tfrecords` to the function, `tfrecords_input_callable`.
    * Deletes the directory of `.tfrecords` (it's a tempdir so it falls out of scope).
    * Returns the output of `tfrecords_input_callable`.
    :return: Returns the output of `tfrecords_input_callable`. Given the input image loaded
    as tfrecords.
    """

    def load_dataset_object(records_directory: Path) -> TFRecordDataset:
        """
        Helper function.
        :param records_directory: The directory that contain the `.tfrecords` to load.
        :return: The records loaded into memory
        """
        output: TFRecordDataset = dataset.load_dataset(  # type: ignore
            data_dir=str(records_directory.parent),
            tfrecord_dir=str(records_directory),
            max_label_size=0,
            repeat=False,
            shuffle_mb=0,
        )
        return output

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        directory = Path(tmp_dir_name)
        _image_to_tfrecords_directory(directory, image)

        # These numbers here come from the `stylegan2` implementation.
        dataset_obj = load_dataset_object(directory)
        images, _labels = dataset_obj.get_minibatch_np(1)  # type: ignore
        input_images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])  # type: ignore

        # When we return here, `tmp_dir_name` falls out of scope and is deleted.
        return tfrecords_input_callable(input_images)


class IntermediateResults(NamedTuple):
    """
    Intermediate Type (heh.)
    Stores part of the projection that would be difficult to recover using a processor function
    because they might already be "gone" (written to file, etc).
    """

    final_latents: CompleteLatentsType
    noises_shapes: NoisesShapesType


def project_process_intermediate(
    target_tfrecords: TFRecordsType,
    projector: Projector,
    intermediate_processor: IntermediateProcessorFunction,
) -> IntermediateResults:
    """
    Like `project_image` in stylegan2's `run_projector.py` but doesn't write the intermediate
    results to disk, and maintains the resulting latent vectors for use in other processes.
    Word of warning. Trying to maintain a whole set of latents, noises, and images in memory
    at once is very difficult, which is why this processor function was implemented.
    :param target_tfrecords: Contains the image we're trying to project.
    :param projector: Contains the network we're going to be projecting with, and monitors how the
    process is going.
    :param intermediate_processor: Provides a way for user to process the projection steps along
    the way, see protocol for more docs.
    :return: An NT containing the final latents, and other things that would be cumbersome
    to recover.
    """

    projector.start(target_tfrecords)  # type: ignore

    noises_shapes: Optional[NoisesShapesType] = None

    def noises_to_shapes(noises: List[NoisesType]) -> NoisesShapesType:
        """
        Helper function.
        :param noises: list of current noises.
        :return: The shapes of the noises.
        """
        return tuple(noise.shape for noise in noises)  # type: ignore

    while projector.get_cur_step() < projector.num_steps:  # type: ignore
        projector.step()  # type: ignore

        # Performs some action on the current results of the projection. Write it to file,
        # do some analysis and print it, visualize it etc.
        intermediate_processor(
            step_number=projector.get_cur_step(),  # type: ignore
            latents=projector.get_dlatents(),  # type: ignore
            noises=projector.get_noises(),  # type: ignore
            images=projector.get_images(),  # type: ignore
        )

        current_shapes = noises_to_shapes(noises=projector.get_noises())  # type: ignore

        if noises_shapes is None:
            noises_shapes = current_shapes
        else:
            if current_shapes != noises_shapes:
                LOGGER.warning(
                    f"Noises shapes have changed mid projection! "
                    f"Was: {noises_shapes}, now: {current_shapes}"
                )

    return IntermediateResults(
        final_latents=projector.get_dlatents(),  # type: ignore
        noises_shapes=noises_shapes,
    )


def project_video_to_file(  # pylint: disable=too-many-locals,too-many-arguments
    path_to_video: Path,
    path_to_network: Path,
    projection_file_path: Path,
    video_fps: Optional[float] = None,
    projection_width_height: Optional[Tuple[int, int]] = None,
    projection_fps: Optional[float] = None,
    steps_per_projection: Optional[int] = None,
    num_frames_to_project: Optional[int] = None,
    latents_histories_enabled: bool = True,
    noises_histories_enabled: bool = False,
    images_histories_enabled: bool = False,
    batch_number: Optional[int] = None,
) -> None:
    """
    Project a video to a projection file on disk.
    :param path_to_video: Path to the video to project.
    :param path_to_network: Path to the network to do the projection with.
    :param projection_file_path: Path to output file.
    :param video_fps: Can override the actual FPS of the input video.
    :param projection_width_height: Scale each frame of the video to this size before feeding it
    into projection.
    :param projection_fps: Down sample the video to be at this FPS. Note, can only be lower than the
    original FPS of the input video, and must evenly go into original FPS. If not given, projection
    will be done at the native FPS of the input video.
    :param steps_per_projection: The number of times the `.step()` function will be called for the
    projection in this run. Default is the value baked into the stylegan2 repo.
    :param num_frames_to_project: The number of frames to project. After the video has been
    resampled to the fps given by `projection_fps`, this many frames will be projected of that
    video.
    :param latents_histories_enabled: Records the intermediate latents seen during projection.
    :param noises_histories_enabled: If the noises used in each projection should be recorded.
    Warning! This will make the output file MASSIVE.
    :param images_histories_enabled: If the images over time throughout the projection should be
    recorded. Warning! This will make the output file MASSIVE.
    :param batch_number: If this video is part of a batch of videos being projected, this provides
    context to the logger.
    :return: None
    """

    try:
        video = frames_in_video(
            video_path=path_to_video,
            video_fps=video_fps,
            width_height=ImageResolution(*projection_width_height)
            if projection_width_height is not None
            else None,
            reduce_fps_to=projection_fps,
        )
    except ValueError as e:
        raise ValueError("Couldn't read input video to do projection.") from e

    original_frame_count = video.total_frame_count
    original_fps = video.original_fps
    projection_width_height = (
        projection_width_height
        if projection_width_height is not None
        else video.original_resolution
    )
    network_hash: str = hash_file(path_to_network)
    target_hash: str = hash_file(path_to_video)

    # TODO - find a better home for this `1000` magic number.
    steps_per_projection = steps_per_projection if steps_per_projection is not None else 1000

    true_projection_fps = original_fps if projection_fps is None else projection_fps

    if num_frames_to_project:
        num_projection_frames = num_frames_to_project
    else:
        if projection_fps is None:
            num_projection_frames = original_frame_count
        else:
            num_projection_frames = int(original_frame_count * true_projection_fps / original_fps)

    noises_shapes: Optional[NoisesShapesType] = None

    def make_attributes(complete: bool, shapes: Optional[NoisesShapesType]) -> ProjectionAttributes:
        """
        Helper function, lets you change the status if the projection finishes in a single run.
        TODO we actually want two classes here instead of optional or changing members.
        Think `IncompleteProjectionAttributes` and `ProjectionAttributes`.
        :param complete: Param to pass.
        :param shapes: Noises shapes.
        :return: The attributes ready to be written to the file.
        """
        output = ProjectionAttributes(
            version_number=LATEST_VERSION,
            complete=complete,
            original_target_path=str(path_to_video),
            original_width_height=video.original_resolution,
            projection_width_height=projection_width_height,
            target_md5_hash=target_hash,
            original_network_path=str(path_to_network),
            network_md5_hash=network_hash,
            steps_in_projection=steps_per_projection,
            # Can't use a `None` here, not compatible with hdf5
            noises_shapes=shapes if shapes else np.nan,
            latents_histories_enabled=latents_histories_enabled,
            noises_histories_enabled=noises_histories_enabled,
            images_histories_enabled=images_histories_enabled,
            original_fps=original_fps,
            projection_fps=true_projection_fps,
            original_frame_count=original_frame_count,
            projection_frame_count=num_projection_frames,
        )
        LOGGER.info(f"Setting projection attributes: {output}")
        return output

    with h5py.File(name=str(projection_file_path), mode="w") as f:

        f.attrs.update(
            make_attributes(  # type: ignore  # pylint: disable=no-member
                complete=False,
                shapes=noises_shapes,
            ).to_dict()
        )

        # The top level groups are always created even though they could be empty.

        # going to be full of datasets
        per_frame_dataset_groups = [f.create_group(name) for name in _PER_FRAME_DATASET_GROUP_NAMES]

        # going to be filled with groups that are going to be filled with datasets
        per_frame_groups: List[Group] = [
            f.create_group(name) for name in _PER_FRAME_SUB_GROUP_GROUP_NAMES
        ]

        for index, frame in enumerate(itertools.islice(video.frames, num_frames_to_project)):

            LOGGER.info(f"Rendering projection {index}/{num_projection_frames}")

            current_groups: List[Optional[Group]] = [
                group.create_group(f"{Path(group.name).name}_{index}") if enabled else None
                for group, enabled in zip(
                    per_frame_groups,
                    [latents_histories_enabled, images_histories_enabled, noises_histories_enabled],
                )
            ]

            result: TotalProjectionResult = project_image_in_process(
                network_path=path_to_network,
                image=frame,
                intermediate_processor=partial(
                    _write_intermediate,
                    batch_number,
                    index + 1,
                    num_projection_frames,
                    steps_per_projection,
                    current_groups,
                ),
                steps_per_projection=steps_per_projection,
            )

            if noises_shapes is None:
                noises_shapes = result.noises_shapes
            else:
                if result.noises_shapes != noises_shapes:
                    LOGGER.warning(
                        f"Noises shapes changed between projections. "
                        f"Was: {noises_shapes}, now: {result.noises_shapes}"
                    )

            final_data: List[Union[RGBInt8ImageType, CompleteLatentsType, FlattenedNoisesType]] = [
                frame,
                result.final_latents,
                result.projected_image,
            ]

            for frame_group, frame_data in zip(per_frame_dataset_groups, final_data):
                _create_dataset_wrapper(
                    group=frame_group,
                    name=f"{frame_group.name}_{index}",
                    data=frame_data,
                )

            LOGGER.info("Projection of frame written to file.")
            f.flush()

        LOGGER.info("Projection totally complete!")

        f.attrs.update(
            make_attributes(  # type: ignore  # pylint: disable=no-member
                complete=True, shapes=noises_shapes
            ).to_dict()
        )


def _flatten_noises(noises: List[NoisesType]) -> FlattenedNoisesType:
    """
    Flatten the noises array so it can be written to the file.
    :param noises: Noises to flatten.
    :return: Flattened noises.
    """
    return FlattenedNoisesType(
        np.concatenate([noise.flatten() for noise in noises])  # type: ignore[no-untyped-call]
    )


def _create_dataset_wrapper(
    group: Group, name: str, data: Union[RGBInt8ImageType, CompleteLatentsType, FlattenedNoisesType]
) -> None:
    """
    Write a given piece of data to the given group as a dataset.
    :param group: Group to write `data` to.
    :param name: Name of the dataset.
    :param data: Data to write. The type of this variable is read directly and set to the type
    of the resulting dataset.
    :return: None
    """
    data_set = group.create_dataset(
        f"/{group.name}/{name}",
        shape=data.shape,
        dtype=data.dtype,
        data=data,
        compression="gzip",
        compression_opts=COMPRESSION_LEVEL,
        shuffle=True,
    )
    LOGGER.debug(f"Wrote dataset: {data_set.name}")


def _write_intermediate(
    batch_number: Optional[int],
    frame_number: int,
    total_frames: int,
    total_steps: int,
    current_groups: List[Optional[Group]],
    step_number: int,
    latents: CompleteLatentsType,
    noises: List[NoisesType],
    images: RGBInt8ImageType,
) -> None:
    """
    Writes the input data structures to the current projection file.
    Note on latents here. This input is going to be of shape (1, 18, 512) as an example, but
    in the read operation will be converted to (18, 512). Rationale being that I don't really
    understand if there's value to that outer matrix, but the data is so expensive to compute that
    it seems stupid to drop on the floor at read time because there may be a day where it's useful.
    :param frame_number: Provided via a partial, not part of the protocol and consumed in a log.
    :param total_frames: Provided via a partial, not part of the protocol and consumed in a log.
    :param total_steps: Provided via a partial, not part of the protocol and consumed in a log.
    :param current_groups: Groups to write the data to, assumes these are in the correct order.
    :param step_number: See protocol docs.
    :param latents: See protocol docs.
    :param noises: See protocol docs.
    :param images: See protocol docs.
    :return: See protocol docs.
    """

    to_write: List[Union[RGBInt8ImageType, CompleteLatentsType, FlattenedNoisesType]] = [
        latents,
        _flatten_noises(noises),
        images,
    ]

    for intermediate_group, intermediate_data in zip(current_groups, to_write):
        if intermediate_group is not None:
            _create_dataset_wrapper(
                group=intermediate_group,
                name=f"{Path(intermediate_group.name).name}_step_{step_number}",
                data=intermediate_data,
            )

    LOGGER.info(
        "Processed Projection Step: "
        f"{f'Batch: {batch_number} - ' if batch_number is not None else ''}"
        f"Frame: {frame_number}/{total_frames} - "
        f"Step: {step_number}/{total_steps}"
    )
