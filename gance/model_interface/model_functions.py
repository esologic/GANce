"""
Functions to be able to load models from disk and interact with them.
Working with these models can be messy and this module tries to only expose the important things.
"""

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

MODEL_SUFFIX = ".pkl"


def sorted_models_in_directory(models_directory: Path) -> List[Path]:
    """
    Given a directory with model `.pkl` files in it, return a list of the paths in this directory
    sorted alphabetically.
    :param models_directory: Path to the directory with models in it.
    :return: List of Paths to models.
    """
    return list(sorted(models_directory.glob(f"*{MODEL_SUFFIX}")))


class ImageFunction(Protocol):  # pylint: disable=too-few-public-methods
    """
    Defines the function that writes vectors to a model producing images.
    """

    def __call__(
        self: "ImageFunction", data: Union[SingleVector, SingleMatrix]
    ) -> RGBInt8ImageType:
        """
        Coverts some data, a vector or matrix to an image.
        :param data: The vector to be input to the model.
        :return: The image.
        """


class ModelInterface(NamedTuple):
    """
    A common interface to only expose the necessary parts of a model to avoid confusion.
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


def load_model_network(model_path: Path, call_init_function: bool) -> Network:
    """
    Load the model from a file.
    :param model_path: Path to the model.
    :param call_init_function: If true, will init the tensorflow session.
    :return: The network.
    """

    if call_init_function:
        fix_cuda_path()
        dnnlib.tflib.init_tf()

    # Once you load this thing it does all kinds of bullshit under the hood, and it's almost
    # impossible to unload it. If you need to unload/load a new model in an application
    # look at `create_model_interface_process`.
    model: Network = pickle.load(
        open(str(model_path), "rb")  # pylint: disable=consider-using-with
    )[2]
    return model


def wrap_loaded_model(model: Network) -> ModelInterface:
    """
    Given the model's network object produce a wrapper NT.
    :param model: The network to wrap.
    :return: The wrapper NT.
    """

    network_kwargs = dnnlib.EasyDict()
    network_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    network_kwargs.input_transform = None
    network_kwargs.truncation_psi = 1.2
    network_kwargs.randomize_noise = False

    def reshape_input_for_model(data: Union[SingleVector, SingleMatrix]) -> RGBInt8ImageType:
        """
        Model expects the data reshaped to be (1, original, shape).
        :param data: Data to reshape.
        :return: Reshaped data
        """
        new_shape = (1, *data.shape)
        return RGBInt8ImageType(np.reshape(data, new_shape))

    def convert_model_output_to_image(model_output: np.ndarray) -> RGBInt8ImageType:
        """
        Takes the raw output from an inference and converts it to a more usable image type.
        :param model_output: Raw output from model.
        :return: Image
        """
        return RGBInt8ImageType(np.array(PIL.Image.fromarray(model_output[0])))

    def create_image_vector(data: SingleVector) -> RGBInt8ImageType:
        """
        Canonical way to go from vector -> image.
        :param data: Vector to synthesize.
        :return: Raw image from model.
        """
        # Verified by hand that this cast is valid.
        return RGBInt8ImageType(
            model.run(
                reshape_input_for_model(data),
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
        :return: Raw image from model.
        """
        return RGBInt8ImageType(
            model.components.synthesis.run(reshape_input_for_model(data), **network_kwargs)
        )

    def create_image_generic(data: Union[SingleVector, SingleMatrix]) -> RGBInt8ImageType:
        """
        Checks the data to determine its type, then passes it to the corresponding function.
        Slightly less inefficient. Use only if you have to.
        :param data: Data to synthesize.
        :return: Raw image from model.
        """
        if is_vector(data):
            LOGGER.info(f"Generic -> Vector, shape: {data.shape}")
            return create_image_vector(data)
        else:
            LOGGER.info(f"Generic -> matrix, shape: {data.shape}")
            return create_image_matrix(data)

    return ModelInterface(
        create_image_vector=lambda data: convert_model_output_to_image(create_image_vector(data)),
        create_image_matrix=lambda data: convert_model_output_to_image(create_image_matrix(data)),
        create_image_generic=lambda data: convert_model_output_to_image(create_image_generic(data)),
        expected_vector_length=model.input_shape[1],
    )


def create_model_interface(model_path: Path, call_init_function: bool) -> ModelInterface:
    """
    Creates the interface to be able to send vectors to and get images out of the model.
    :param model_path: Path to the models `.pkl` file on disk.
    :param call_init_function: If True, this function will call `dnnlib.tflib.init_tf()`,
    which is required to unpickle the model. It's possible caller has already done this which is
    why it's optional.
    :return:
    """
    return wrap_loaded_model(load_model_network(model_path, call_init_function))


class ModelInterfaceInProcess(NamedTuple):
    """
    This exposes the ModelInterface (for creating images etc) and a stop function, which when
    called unloads the model from memory.
    """

    model_interface: ModelInterface
    stop_function: Callable[[], None]


class _ModelInput(NamedTuple):
    """
    Intermediate type.
    We don't want to have different queues for each possible type of vector, so we pass a flag
    indicative of type alongside the vector.
    """

    # Indicates the type of input this is. Should decide which model function is used to produce
    # an image This could easily become an enum if we have more than two input configs.
    is_vector: bool

    # The data to write to the model.
    model_input: Union[SingleVector, SingleMatrix]


def create_model_interface_process(model_path: Path) -> ModelInterfaceInProcess:
    """
    Wraps the loading of/ interfacing with a styleGAN2 model with in a subprocess, so the whole
    thing can be killed avoiding well known, and unsolved problems with memory leaks in tensorflow.
    :param model_path: Path to the models `.pkl` file on disk.
    :return: A NamedTuple that defines the interface with the model and exposes a function to
    "delete" the model from local memory and video memory.
    """

    vector_length_value = multiprocessing.Value("i", LENGTH_SENTINEL)
    input_queue: "Queue[Union[str, _ModelInput]]" = multiprocessing.Queue()
    output_queue: "Queue[Union[str, RGBInt8ImageType]]" = multiprocessing.Queue()
    error_queue: "Queue[Union[str, Exception]]" = multiprocessing.Queue()

    started_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    process = multiprocessing.Process(
        target=_model_worker,
        kwargs={
            "model_path": model_path,
            "vector_length_value": vector_length_value,
            "input_queue": input_queue,
            "output_queue": output_queue,
            "error_queue": error_queue,
            "stop_event": stop_event,
            "started_event": started_event,
        },
    )

    process.start()

    while not started_event.is_set():
        # Wait around until the model has been loaded
        pass

    # Only one object is going to to be put here up until this point, so the single `get()` is safe.
    startup_result: Union[str, Exception] = error_queue.get()

    if startup_result != STARTED_WITHOUT_ERROR_SENTINEL:
        logging.error(f"Couldn't load model. Exception from worker: {startup_result}")
        logging.error("Killing worker...")
        process.terminate()
        process.join()
        logging.error("Worker joined...")
        raise startup_result  # type: ignore

    # Wait for the model to be loaded
    while True:
        with vector_length_value.get_lock():
            vector_length = int(vector_length_value.value)
            logging.debug(f"Got lock in top, vector length {vector_length}")
            if vector_length != LENGTH_SENTINEL:
                break

    logging.debug("Model Loaded")

    def image_from_vector(
        is_a_vector: bool, data: Union[SingleVector, SingleMatrix]
    ) -> RGBInt8ImageType:
        """
        Puts a vector into the queue so it can be input by the worker into the model.
        Watches the output queue until the resulting image arrives.
        :param data: The input to the model.
        :param is_a_vector: If the shallow input function should consume this vector or the
        full latents.
        :return: The resulting image.
        """
        input_queue.put(_ModelInput(is_a_vector, data))

        while True:
            try:
                return RGBInt8ImageType(output_queue.get_nowait())
            except Empty:
                pass

    def stop_function() -> None:
        """
        Calling this function gracefully stops the worker process, which in turn unloads
        the model as described in top-level docs.
        :return: None
        """

        # signals the process to stop processing input
        stop_event.set()

        # empty the queues, allows `process` to be `.join`ed reliably.
        input_queue.put(COMPLETE_SENTINEL)
        empty_queue_sentinel(output_queue)

        try:
            process_common.cleanup_worker(process=process)
        except ValueError as e:
            LOGGER.error("Couldn't clean up model interface worker process")
            raise e

    def handle_generic(data: Union[SingleVector, SingleMatrix]) -> RGBInt8ImageType:
        """
        Handle the generic case. Since we have functions already to process either type of
        data, just make the call out here.
        :param data: To input to model.
        :return: Result from model
        """
        return image_from_vector(is_vector(data), data)

    return ModelInterfaceInProcess(
        model_interface=ModelInterface(
            create_image_vector=partial(image_from_vector, True),
            create_image_matrix=partial(image_from_vector, False),
            create_image_generic=handle_generic,
            expected_vector_length=vector_length,
        ),
        stop_function=stop_function,
    )


def _model_worker(
    model_path: Path,
    vector_length_value: Any,
    input_queue: "Queue[Union[str, _ModelInput]]",
    output_queue: "Queue[Union[str, RGBInt8ImageType]]",
    error_queue: "Queue[Union[str, Exception]]",
    started_event: Any,
    stop_event: Any,
) -> None:
    """
    Used to create images from a model inside of a child process, (an `multiprocessing.Process`) so
    the entire tensorflow/keras/cuda session can be destroyed and cleaned up by the host's OS
    rather than python. This is because the teardown process is messy in tensorflow and can often
    lead to memory leaks. The input args are all thread safe, and are used for IPC.

    This function creates a model, the loops, reading input vectors from the `input_queue`,
    and placing the resulting images in `output_queue`.

    :param model_path: The path to the model to interface with on disk.
    :param vector_length_value: Once the model is loaded, this `multiprocessing.Value` will be
    set to the expected input vector length.
    :param input_queue: Vector source. Will be emptied upon shutdown.
    :param output_queue: Image destination.
    :param error_queue: Errors that occur on model load are put into there. TODO: could
    also use this as a way to communicate errors that occur during runtime.
    :param stop_event: A `multiprocessing.Event`, it will be `.set()` to signal that this
    worker should exit.
    :param started_event: Another event, used to signal that the model has been loaded and
    is accepting input vectors.
    :return: None
    """

    logging.debug("Created new model worker")

    startup_error = False
    try:
        # Need to do the init function every time, because this will be called inside of a child
        # process only. Once these child processes are killed the tensorflow session goes away as
        # well.
        model_interface = create_model_interface(model_path=model_path, call_init_function=True)
        logging.debug("Model successfully loaded in worker process.")
    except Exception as e:  # pylint: disable=broad-except
        # Catch everything, this error will be consumed in the parent.
        startup_error = True
        error_queue.put(e)

    if not startup_error:
        error_queue.put(STARTED_WITHOUT_ERROR_SENTINEL)

    # Parent will now know to look in the error queue to see if model loaded correctly.
    started_event.set()

    if startup_error:
        logging.error(f"Could not load model at path {model_path}. Exiting worker process.")
        return None

    with vector_length_value.get_lock():
        logging.debug("Got lock for vector length")
        vector_length_value.value = model_interface.expected_vector_length

    logging.debug("Set vector length.")
    logging.debug("Ready to start processing vectors.")

    queue_needs_cleaning = True

    while not stop_event.is_set():
        try:
            model_input_or_sentinel: Union[_ModelInput, str] = input_queue.get_nowait()
            if isinstance(model_input_or_sentinel, _ModelInput):
                try:
                    image = (
                        model_interface.create_image_vector(model_input_or_sentinel.model_input)
                        if model_input_or_sentinel.is_vector
                        else model_interface.create_image_matrix(
                            model_input_or_sentinel.model_input
                        )
                    )
                    output_queue.put(image)
                except Exception:  # pylint: disable=broad-except
                    logging.error(
                        f"Couldn't make image. Model: {model_path.name}, "
                        f"vector: {model_input_or_sentinel}"
                    )
                    # TODO - we need to actually bring down the worker at this point.
            elif model_input_or_sentinel == COMPLETE_SENTINEL:
                queue_needs_cleaning = False
            else:
                raise ValueError(f"Got bad object out of input queue: {model_input_or_sentinel}")
        except Empty:
            pass

    if queue_needs_cleaning:
        empty_queue_sentinel(input_queue)

    output_queue.put(COMPLETE_SENTINEL)


@typing.no_type_check
def _raise_exception_if_unloaded(function):
    """
    Designed to be used as a decorator within the MultiModel class.
    This is a pretty steep price to pay for the cleanliness of the property.
    :raises ValueError: If the `self.load()` function has not been called before trying to
    use the inner model.
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
                self._currently_loaded_model,  # pylint: disable=protected-access
                self._expected_vector_length,  # pylint: disable=protected-access
            ]
        ):
            raise ValueError("MultiModel is not initialized! Call load or use context manager.")
        return function(*args, **kwargs)

    return wrapper


class MultiModel:
    """
    Allows user to switch between models during a run.
    Models are loaded/unloaded into memory/GPU on demand.
    """

    def __init__(self: "MultiModel", model_paths: List[Path]) -> None:
        """
        :param model_paths: The list of candidate models.
        """

        self._model_paths: List[Path] = model_paths
        self._currently_loaded_model_index: int = 0

        # These will get set when the context manager is used or the `self.load()` function is
        # called.
        self._currently_loaded_model: Optional[ModelInterfaceInProcess] = None
        self._expected_vector_length: Optional[int] = None

    @property  # type: ignore
    @_raise_exception_if_unloaded
    def expected_vector_length(self: "MultiModel") -> int:
        """
        The length of expected input vectors, as reported by the model.
        :return: value as an int.
        """
        return self._expected_vector_length

    def __enter__(self: "MultiModel") -> Optional["MultiModel"]:
        """
        For using this object as a context manager. Calls the load method and returns self.
        If the model cannot be loaded into the GPU, the resulting object will be `None`.
        Consumers should know about this.
        :return: The now loaded self
        """
        try:
            self.load()
        except RuntimeError:
            logging.warning("Couldn't load model into GPU, proceeding without model.")
            return None

        return self

    @typing.no_type_check
    def __exit__(self, exec_type, exec_value, exec_traceback) -> None:
        """
        For using this object as a context manager. Unloads currently loaded model.
        :param exec_type: The type of exception raised during use if any.
        :param exec_value: The value of the exception raised during use if any.
        :param exec_traceback: The traceback of the exception raised during use if any.
        :return: None
        """
        if self._currently_loaded_model is not None:
            self.unload()

    def _load_model_at_index(self: "MultiModel", index: int) -> None:
        """
        Load the model at the given index. If the model is already loaded, do nothing.
        :param index: Model index to load.
        :return: None
        """

        # Check to actually make sure the new model is going to be a different model.
        # The same path can get passed in via the model list more than once.
        if (
            index != self._currently_loaded_model_index
            and self._model_paths[self._currently_loaded_model_index] != self._model_paths[index]
        ):
            logging.info(
                f"Unloading {self._model_paths[self._currently_loaded_model_index].name}, "
                f"Loading {self._model_paths[index].name}"
            )
            self._currently_loaded_model.stop_function()
            self._currently_loaded_model_index = index
            self.load()

    @_raise_exception_if_unloaded
    def indexed_create_image_vector(
        self: "MultiModel", index: int, data: SingleVector
    ) -> RGBInt8ImageType:
        """
        Returns a frame for the model at the given index given the input vector.
        :param index: The model's index in the list of input models.
        :param data: The latents to get the output for.
        :return: The image from the model.
        """
        self._load_model_at_index(index)
        return self._currently_loaded_model.model_interface.create_image_vector(data)

    @_raise_exception_if_unloaded
    def indexed_create_image_matrix(
        self: "MultiModel", index: int, data: SingleMatrix
    ) -> RGBInt8ImageType:
        """
        Returns a frame for the model at the given index given the input matrix.
        :param index: The model's index in the list of input models.
        :param data: The latents to get the output for.
        :return: The image from the model.
        """
        self._load_model_at_index(index)
        return self._currently_loaded_model.model_interface.create_image_matrix(data)

    @_raise_exception_if_unloaded
    def indexed_create_image_generic(
        self: "MultiModel", index: int, data: Union[SingleVector, SingleMatrix]
    ) -> RGBInt8ImageType:
        """
        Returns a frame for the model at the given index given the input matrix.
        :param index: The model's index in the list of input models.
        :param data: The latents to get the output for.
        :return: The image from the model.
        """
        self._load_model_at_index(index)
        return self._currently_loaded_model.model_interface.create_image_generic(data)

    def load(self: "MultiModel") -> None:
        """
        Loads the model at the set index into memory for use.
        :return: None
        """
        model_path = self._model_paths[self._currently_loaded_model_index]
        LOGGER.info(f"Loading Model: {model_path}")
        self._currently_loaded_model = create_model_interface_process(model_path=model_path)
        self._expected_vector_length = (
            self._currently_loaded_model.model_interface.expected_vector_length
        )

    @_raise_exception_if_unloaded
    def unload(self: "MultiModel") -> None:
        """
        Kills the child process, and frees the corresponding resources associated with the
        currently loaded model.
        :return: None
        """
        self._currently_loaded_model.stop_function()

    @property
    def model_indices(self: "MultiModel") -> List[int]:
        """
        The number of different models this object can switch between, candidates
        for the `index` parameter of `create_image`.
        :return: Indices
        """
        return [index for index, _ in enumerate(self._model_paths)]
