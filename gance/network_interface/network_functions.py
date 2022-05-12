"""
Functions to be able to load networks from disk and interact with them.
Working with these networks can be messy and this module tries to only expose the important things.
"""
import json
import logging
import multiprocessing
import os
import pickle
import sys
import typing
from functools import partial, wraps
from multiprocessing import Queue  # pylint: disable=unused-import
from pathlib import Path
from queue import Empty
from typing import Any, Callable, List, NamedTuple, Optional, Union  # pylint: disable=unused-import

import numpy as np
import PIL
import pydantic
from pydantic import BaseModel, FilePath
from typing_extensions import Protocol

from gance import process_common
from gance.gance_types import RGBInt8ImageType
from gance.logger_common import LOGGER
from gance.process_common import COMPLETE_SENTINEL, empty_queue_sentinel
from gance.stylegan2 import dnnlib
from gance.stylegan2.dnnlib import tflib  # pylint: disable=unused-import
from gance.stylegan2.dnnlib.tflib import Network
from gance.vector_sources.vector_types import SingleMatrix, SingleVector, is_vector

sys.modules["dnnlib"] = dnnlib

LENGTH_SENTINEL = -1
STARTED_WITHOUT_ERROR_SENTINEL = "Started Without Error"

NETWORK_SUFFIX = ".pkl"


def sorted_networks_in_directory(networks_directory: Path) -> List[Path]:
    """
    Given a directory with network `.pkl` files in it, return a list of the paths in this directory
    sorted alphabetically.
    :param networks_directory: Path to the directory with networks in it.
    :return: List of Paths to networks.
    """
    return list(sorted(networks_directory.glob(f"*{NETWORK_SUFFIX}")))


class ImageFunction(Protocol):  # pylint: disable=too-few-public-methods
    """
    Defines the function that writes vectors to a network producing images.
    """

    def __call__(
        self: "ImageFunction", data: Union[SingleVector, SingleMatrix]
    ) -> RGBInt8ImageType:
        """
        Coverts some data, a vector or matrix to an image.
        :param data: The vector to be input to the network.
        :return: The image.
        """


class NetworkInterface(NamedTuple):
    """
    A common interface to only expose the necessary parts of a network to avoid confusion.
    """

    # The length of the individual vectors within the matrix.
    expected_vector_length: int

    create_image_vector: ImageFunction  # accepts a vector as input
    create_image_matrix: ImageFunction  # accepts a matrix as input

    # accepts either, then decides based on the type which fun to use.
    create_image_generic: ImageFunction


def fix_cuda_path() -> None:
    """
    For some reason, even if you have a `.bashrc` file in place, pycharm's remote interpreter
    doesn't get the correct PATH variables set and subsequently can't find `nvcc`. If they're
    missing from path they get added here.
    :return: None
    """
    for path_string in ["/usr/local/cuda/bin", "/usr/local/cuda/lib64"]:
        if path_string not in os.environ["PATH"]:
            os.environ["PATH"] += ":" + path_string


def load_network_network(
    network_path: Path, call_init_function: bool, gpu_index: Optional[int] = None
) -> Network:
    """
    Load the network from a file.
    :param network_path: Path to the network.
    :param call_init_function: If true, will init the tensorflow session.
    :param gpu_index: If given, the network will be loaded on this GPU. Allows for multiple networks
    to be loaded simultaneously across multiple GPUs. Will have no effect if `call_init_function`
    is False.
    :return: The network.
    """

    if call_init_function:
        fix_cuda_path()

        if gpu_index is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        dnnlib.tflib.init_tf()

    # Once you load this thing it does all kinds of bullshit under the hood, and it's almost
    # impossible to unload it. If you need to unload/load a new network in an application
    # look at `create_network_interface_process`.
    network: Network = pickle.load(
        open(str(network_path), "rb")  # pylint: disable=consider-using-with
    )[2]
    return network


def wrap_loaded_network(network: Network) -> NetworkInterface:
    """
    Given the network's network object produce a wrapper NT.
    :param network: The network to wrap.
    :return: The wrapper NT.
    """

    network_kwargs = dnnlib.EasyDict()
    network_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    network_kwargs.input_transform = None
    network_kwargs.truncation_psi = 1.2
    network_kwargs.randomize_noise = False

    def reshape_input_for_network(data: Union[SingleVector, SingleMatrix]) -> RGBInt8ImageType:
        """
        network expects the data reshaped to be (1, original, shape).
        :param data: Data to reshape.
        :return: Reshaped data
        """
        new_shape = (1, *data.shape)
        return RGBInt8ImageType(np.reshape(data, new_shape))

    def convert_network_output_to_image(network_output: np.ndarray) -> RGBInt8ImageType:
        """
        Takes the raw output from an inference and converts it to a more usable image type.
        :param network_output: Raw output from network.
        :return: Image
        """
        return RGBInt8ImageType(np.array(PIL.Image.fromarray(network_output[0])))

    def create_image_vector(data: SingleVector) -> RGBInt8ImageType:
        """
        Canonical way to go from vector -> image.
        :param data: Vector to synthesize.
        :return: Raw image from network.
        """
        # Verified by hand that this cast is valid.
        return RGBInt8ImageType(
            network.run(
                reshape_input_for_network(data),
                None,
                truncation_psi=1.2,
                output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            )
        )

    def create_image_matrix(data: SingleMatrix) -> RGBInt8ImageType:
        """
        This is the way images are created during the projection process.
        Latents from there need to be used here to create images.
        :param data: Latents to synthesize.
        :return: Raw image from network.
        """
        return RGBInt8ImageType(
            network.components.synthesis.run(reshape_input_for_network(data), **network_kwargs)
        )

    def create_image_generic(data: Union[SingleVector, SingleMatrix]) -> RGBInt8ImageType:
        """
        Checks the data to determine its type, then passes it to the corresponding function.
        Slightly less inefficient. Use only if you have to.
        :param data: Data to synthesize.
        :return: Raw image from network.
        """
        if is_vector(data):
            LOGGER.info(f"Generic -> Vector, shape: {data.shape}")
            return create_image_vector(data)
        else:
            LOGGER.info(f"Generic -> matrix, shape: {data.shape}")
            return create_image_matrix(data)

    return NetworkInterface(
        create_image_vector=lambda data: convert_network_output_to_image(create_image_vector(data)),
        create_image_matrix=lambda data: convert_network_output_to_image(create_image_matrix(data)),
        create_image_generic=lambda data: convert_network_output_to_image(
            create_image_generic(data)
        ),
        expected_vector_length=network.input_shape[1],
    )


def create_network_interface(
    network_path: Path, call_init_function: bool, gpu_index: Optional[int] = None
) -> NetworkInterface:
    """
    Creates the interface to be able to send vectors to and get images out of the network.
    :param network_path: Path to the networks `.pkl` file on disk.
    :param call_init_function: If True, this function will call `dnnlib.tflib.init_tf()`,
    which is required to unpickle the network. It's possible caller has already done this which is
    why it's optional.
    :param gpu_index: If given, the network will be loaded on this GPU. Allows for multiple networks
    to be loaded simultaneously across multiple GPUs.
    :return:
    """
    return wrap_loaded_network(load_network_network(network_path, call_init_function, gpu_index))


class NetworkInterfaceInProcess(NamedTuple):
    """
    This exposes the networkInterface (for creating images etc) and a stop function, which when
    called unloads the network from memory.
    """

    network_interface: NetworkInterface
    stop_function: Callable[[], None]


class _NetworkInput(NamedTuple):
    """
    Intermediate type.
    We don't want to have different queues for each possible type of vector, so we pass a flag
    indicative of type alongside the vector.
    """

    # Indicates the type of input this is. Should decide which network function is used to produce
    # an image This could easily become an enum if we have more than two input configs.
    is_vector: bool

    # The data to write to the network.
    network_input: Union[SingleVector, SingleMatrix]


def create_network_interface_process(
    network_path: Path, gpu_index: Optional[int] = None
) -> NetworkInterfaceInProcess:
    """
    Wraps the loading of/ interfacing with a styleGAN2 network with in a subprocess, so the whole
    thing can be killed avoiding well known, and unsolved problems with memory leaks in tensorflow.
    :param network_path: Path to the networks `.pkl` file on disk.
    :param gpu_index: If given, the network will be loaded on this GPU. Allows for multiple networks
    to be loaded simultaneously across multiple GPUs.
    :return: A NamedTuple that defines the interface with the network and exposes a function to
    "delete" the network from local memory and video memory.
    """

    vector_length_value = multiprocessing.Value("i", LENGTH_SENTINEL)
    input_queue: "Queue[Union[str, _NetworkInput]]" = multiprocessing.Queue()
    output_queue: "Queue[Union[str, RGBInt8ImageType]]" = multiprocessing.Queue()
    error_queue: "Queue[Union[str, Exception]]" = multiprocessing.Queue()

    started_event = multiprocessing.Event()
    vector_length_ready_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    process = multiprocessing.Process(
        target=_network_worker,
        kwargs={
            "network_path": network_path,
            "vector_length_value": vector_length_value,
            "input_queue": input_queue,
            "output_queue": output_queue,
            "error_queue": error_queue,
            "stop_event": stop_event,
            "started_event": started_event,
            "vector_length_ready_event": vector_length_ready_event,
            "gpu_index": gpu_index,
        },
    )

    process.start()

    # Wait around until the network has been loaded

    try:
        print('Press Ctrl+C to exit')
        started_event.wait()
    except KeyboardInterrupt:
        print('got Ctrl+C')

    # Only one object is going to to be put here up until this point, so the single `get()` is safe.
    startup_result: Union[str, Exception] = error_queue.get()

    if startup_result != STARTED_WITHOUT_ERROR_SENTINEL:
        logging.error(f"Couldn't load network. Exception from worker: {startup_result}")
        logging.error("Killing worker...")
        process.terminate()
        process.join()
        logging.error("Worker joined...")
        raise startup_result  # type: ignore

    # Wait for the network to be loaded
    vector_length_ready_event.wait()

    with vector_length_value.get_lock():
        vector_length = int(vector_length_value.value)
        logging.debug(f"Got lock in top, vector length {vector_length}")

    logging.debug("network Loaded")

    def image_from_vector(
        is_a_vector: bool, data: Union[SingleVector, SingleMatrix]
    ) -> RGBInt8ImageType:
        """
        Puts a vector into the queue so it can be input by the worker into the network.
        Watches the output queue until the resulting image arrives.
        :param data: The input to the network.
        :param is_a_vector: If the shallow input function should consume this vector or the
        full latents.
        :return: The resulting image.
        """
        input_queue.put(_NetworkInput(is_a_vector, data))
        return RGBInt8ImageType(output_queue.get())

    def stop_function() -> None:
        """
        Calling this function gracefully stops the worker process, which in turn unloads
        the network as described in top-level docs.
        :return: None
        """

        logging.info("Starting stop function.")

        # signals the process to stop processing input
        stop_event.set()

        # empty the queues, allows `process` to be `.join`ed reliably.
        input_queue.put(COMPLETE_SENTINEL)
        empty_queue_sentinel(output_queue)

        logging.info("GPU output queue empty.")

        try:
            process_common.cleanup_worker(process=process)
        except ValueError as e:
            LOGGER.error("Couldn't clean up network interface worker process")
            raise e

        logging.info("Stop function completed.")

    def handle_generic(data: Union[SingleVector, SingleMatrix]) -> RGBInt8ImageType:
        """
        Handle the generic case. Since we have functions already to process either type of
        data, just make the call out here.
        :param data: To input to network.
        :return: Result from network
        """
        return image_from_vector(is_vector(data), data)

    return NetworkInterfaceInProcess(
        network_interface=NetworkInterface(
            create_image_vector=partial(image_from_vector, True),
            create_image_matrix=partial(image_from_vector, False),
            create_image_generic=handle_generic,
            expected_vector_length=vector_length,
        ),
        stop_function=stop_function,
    )


def _network_worker(
    network_path: Path,
    vector_length_value: Any,
    input_queue: "Queue[Union[str, _NetworkInput]]",
    output_queue: "Queue[Union[str, RGBInt8ImageType]]",
    error_queue: "Queue[Union[str, BaseException]]",
    started_event: Any,
    vector_length_ready_event: Any,
    stop_event: Any,
    gpu_index: Optional[int],
) -> None:
    """
    Used to create images from a network inside of a child process, (an `multiprocessing.Process`)
    so the entire tensorflow/keras/cuda session can be destroyed and cleaned up by the host's OS
    rather than python. This is because the teardown process is messy in tensorflow and can often
    lead to memory leaks. The input args are all thread safe, and are used for IPC.

    This function creates a network, the loops, reading input vectors from the `input_queue`,
    and placing the resulting images in `output_queue`.

    :param network_path: The path to the network to interface with on disk.
    :param vector_length_value: Once the network is loaded, this `multiprocessing.Value` will be
    set to the expected input vector length.
    :param input_queue: Vector source. Will be emptied upon shutdown.
    :param output_queue: Image destination.
    :param error_queue: Errors that occur on network load are put into there. TODO: could
    also use this as a way to communicate errors that occur during runtime.
    :param stop_event: A `multiprocessing.Event`, it will be `.set()` to signal that this
    worker should exit.
    :param started_event: Another event, used to signal that the network has been loaded and
    is accepting input vectors.
    :param gpu_index: If given, the network will be loaded on this GPU. Allows for multiple networks
    to be loaded simultaneously across multiple GPUs.
    :return: None
    """

    logging.debug("Created new network worker")

    startup_error = False
    try:
        # Need to do the init function every time, because this will be called inside of a child
        # process only. Once these child processes are killed the tensorflow session goes away as
        # well.
        network_interface = create_network_interface(
            network_path=network_path, call_init_function=True, gpu_index=gpu_index
        )
        logging.debug("Network successfully loaded in worker process.")
    # Need to use `BaseException` here because signals from parent will be forwarded to this
    # child process. `KeyboardInterrupt` must be handled etc.
    except BaseException as e:  # pylint: disable=broad-except
        # Catch everything, this error will be consumed in the parent.
        startup_error = True
        error_queue.put(e)

    if not startup_error:
        error_queue.put(STARTED_WITHOUT_ERROR_SENTINEL)

    # Parent will now know to look in the error queue to see if network loaded correctly.
    started_event.set()

    if startup_error:
        logging.error(f"Could not load network at path {network_path}. Exiting worker process.")
        return None

    with vector_length_value.get_lock():
        logging.debug("Got lock for vector length.")
        vector_length_value.value = network_interface.expected_vector_length

    logging.debug("Set vector length.")
    logging.debug("Ready to start processing vectors.")

    vector_length_ready_event.set()

    queue_needs_cleaning = True

    while not stop_event.is_set():
        try:
            network_input_or_sentinel: Union[_NetworkInput, str] = input_queue.get_nowait()
            if isinstance(network_input_or_sentinel, _NetworkInput):
                try:
                    image = (
                        network_interface.create_image_vector(
                            network_input_or_sentinel.network_input
                        )
                        if network_input_or_sentinel.is_vector
                        else network_interface.create_image_matrix(
                            network_input_or_sentinel.network_input
                        )
                    )
                    output_queue.put(image)
                except Exception:  # pylint: disable=broad-except
                    logging.error(
                        f"Couldn't make image. network: {network_path.name}, "
                        f"vector: {network_input_or_sentinel}"
                    )
                    # TODO - we need to actually bring down the worker at this point.
            elif network_input_or_sentinel == COMPLETE_SENTINEL:
                logging.info("Got stop sentinel in GPU worker.")
                queue_needs_cleaning = False
            else:
                raise ValueError(f"Got bad object out of input queue: {network_input_or_sentinel}")
        except Empty:
            pass
        # Need to use `BaseException` here because signals from parent will be forwarded to
        # this child process. `KeyboardInterrupt` must be handled etc.
        except BaseException:  # pylint: disable=broad-except
            logging.info("Found unexpected exception during run")
            break

    LOGGER.info("Shutting down GPU worker.")

    if queue_needs_cleaning:
        empty_queue_sentinel(input_queue)

    output_queue.put(COMPLETE_SENTINEL)

    LOGGER.info("GPU worker exited.")


@typing.no_type_check
def _raise_exception_if_unloaded(function):
    """
    Designed to be used as a decorator within the Multinetwork class.
    This is a pretty steep price to pay for the cleanliness of the property.
    :raises ValueError: If the `self.load()` function has not been called before trying to
    use the inner network.
    :param function: Function being decorated.
    :return: None
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        """
        A function that can wrap a function and maintain access to the `self.` of the calling
        object.
        :param args: Positional args passed to the wrapped function.
        :param kwargs: Keyword args to be passed to wrapped function.
        :return: the result of the wrapped function
        """
        self = args[0]
        if any(
            member is None
            for member in [
                self._currently_loaded_network,  # pylint: disable=protected-access
                self._expected_vector_length,  # pylint: disable=protected-access
            ]
        ):
            raise ValueError("Multinetwork is not initialized! Call load or use context manager.")
        return function(*args, **kwargs)

    return wrapper


class MultiNetwork:
    """
    Allows user to switch between networks during a run.
    networks are loaded/unloaded into memory/GPU on demand.
    """

    def __init__(self: "MultiNetwork", network_paths: List[Path], load: bool = False) -> None:
        """
        :param network_paths: The list of candidate networks.
        :param load: If True, the default model is loaded on creation of this object.
        """

        self._network_paths: List[Path] = network_paths
        self._currently_loaded_network_index: int = 0

        # These will get set when the context manager is used or the `self.load()` function is
        # called.
        self._currently_loaded_network: Optional[NetworkInterfaceInProcess] = None
        self._expected_vector_length: Optional[int] = None

        if load:
            self.load()

    @property  # type: ignore
    @_raise_exception_if_unloaded
    def expected_vector_length(self: "MultiNetwork") -> int:
        """
        The length of expected input vectors, as reported by the network.
        :return: value as an int.
        """
        return self._expected_vector_length

    def __enter__(self: "MultiNetwork") -> Optional["MultiNetwork"]:
        """
        For using this object as a context manager. Calls the load method and returns self.
        If the network cannot be loaded into the GPU, the resulting object will be `None`.
        Consumers should know about this.
        :return: The now loaded self
        """
        try:
            self.load()
        except RuntimeError:
            logging.warning("Couldn't load network into GPU, proceeding without network.")
            return None

        return self

    @typing.no_type_check
    def __exit__(self, exec_type, exec_value, exec_traceback) -> None:
        """
        For using this object as a context manager. Unloads currently loaded network.
        :param exec_type: The type of exception raised during use if any.
        :param exec_value: The value of the exception raised during use if any.
        :param exec_traceback: The traceback of the exception raised during use if any.
        :return: None
        """
        if self._currently_loaded_network is not None:
            self.unload()

    def _load_network_at_index(self: "MultiNetwork", index: int) -> None:
        """
        Load the network at the given index. If the network is already loaded, do nothing.
        :param index: network index to load.
        :return: None
        """

        # Check to actually make sure the new network is going to be a different network.
        # The same path can get passed in via the network list more than once.
        if (
            index != self._currently_loaded_network_index
            and self._network_paths[self._currently_loaded_network_index]
            != self._network_paths[index]
        ):
            logging.info(
                f"Unloading {self._network_paths[self._currently_loaded_network_index].name}, "
                f"Loading {self._network_paths[index].name}"
            )
            self._currently_loaded_network.stop_function()
            self._currently_loaded_network_index = index
            self.load()

    @_raise_exception_if_unloaded
    def indexed_create_image_vector(
        self: "MultiNetwork", index: int, data: SingleVector
    ) -> RGBInt8ImageType:
        """
        Returns a frame for the network at the given index given the input vector.
        :param index: The network's index in the list of input networks.
        :param data: The latents to get the output for.
        :return: The image from the network.
        """
        self._load_network_at_index(index)
        return self._currently_loaded_network.network_interface.create_image_vector(data)

    @_raise_exception_if_unloaded
    def indexed_create_image_matrix(
        self: "MultiNetwork", index: int, data: SingleMatrix
    ) -> RGBInt8ImageType:
        """
        Returns a frame for the network at the given index given the input matrix.
        :param index: The network's index in the list of input networks.
        :param data: The latents to get the output for.
        :return: The image from the network.
        """
        self._load_network_at_index(index)
        return self._currently_loaded_network.network_interface.create_image_matrix(data)

    @_raise_exception_if_unloaded
    def indexed_create_image_generic(
        self: "MultiNetwork", index: int, data: Union[SingleVector, SingleMatrix]
    ) -> RGBInt8ImageType:
        """
        Returns a frame for the network at the given index given the input matrix.
        :param index: The network's index in the list of input networks.
        :param data: The latents to get the output for.
        :return: The image from the network.
        """
        self._load_network_at_index(index)
        return self._currently_loaded_network.network_interface.create_image_generic(data)

    def load(self: "MultiNetwork") -> None:
        """
        Loads the network at the set index into memory for use.
        :return: None
        """
        network_path = self._network_paths[self._currently_loaded_network_index]
        LOGGER.info(f"Loading network: {network_path}")
        self._currently_loaded_network = create_network_interface_process(network_path=network_path)
        self._expected_vector_length = (
            self._currently_loaded_network.network_interface.expected_vector_length
        )

    @_raise_exception_if_unloaded
    def unload(self: "MultiNetwork") -> None:
        """
        Kills the child process, and frees the corresponding resources associated with the
        currently loaded network.
        :return: None
        """
        self._currently_loaded_network.stop_function()

    @property
    def network_indices(self: "MultiNetwork") -> List[int]:
        """
        The number of different networks this object can switch between, candidates
        for the `index` parameter of `create_image`.
        :return: Indices
        """
        return [index for index, _ in enumerate(self._network_paths)]

    @property
    def network_paths(self: "MultiNetwork") -> List[Path]:
        """
        Exposes the network paths for reading.
        :return: Paths to the network files
        """
        return self._network_paths


def parse_network_paths(
    networks_directory: Optional[str], networks: Optional[List[str]], networks_json: Optional[str]
) -> List[Path]:
    """
    Given the user's input from the CLI, get a list of the networks to be used in the run.
    :param networks_directory: A string representing a path to a directory that contains network
    files. Optionally given.
    :param networks: Paths (as strings) leading directly to networks. Optionally given.
    :param networks_json: Path to a json file with a list of network paths.
    :return: Path objects leading to the networks. Sorted by filename.
    :raises ValueError: If something goes wrong with a parse.
    """

    all_networks = []

    if networks_directory is not None:
        networks_directory_path = Path(networks_directory)
        all_networks += sorted_networks_in_directory(networks_directory=networks_directory_path)

    if networks is not None:
        all_networks += list(map(Path, networks))

    if networks_json is not None:
        LOGGER.info(f"Loading network JSON: {networks_json}")
        try:
            with open(networks_json) as f:
                all_networks += list(map(Path, NetworksFile(**json.load(f)).networks))
        except pydantic.error_wrappers.ValidationError as e:
            raise ValueError("Ran into formatting problem with networks JSON.") from e
        except Exception as e:
            raise ValueError("Couldn't open networks JSON.") from e

    if not all_networks:
        raise ValueError("No networks given, cannot continue.")

    LOGGER.info("Discovered networks: ")
    for path in all_networks:
        LOGGER.info(f"\t{path}")

    return all_networks


class NetworksFile(BaseModel):
    """
    Describes a `NetworksFile`, a .json file full paths to pickled StyleGAN networks.
    """

    networks: List[FilePath]
