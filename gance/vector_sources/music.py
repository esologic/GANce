"""
Functions/types to be able to read in music via wav files and then present that data as an input
vector to models.
"""
import pickle
from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Optional, Union, overload

import numpy as np
import resampy
from scipy.io import wavfile

from gance.logger_common import LOGGER
from gance.vector_sources.vector_sources_common import pad_array, remap_values_into_range

WavDataType = Union["np.ndarray[np.int16]", "np.ndarray[np.float32]", "np.ndarray[np.int16]"]


class WavFileProperties(NamedTuple):
    """
    Intermediate type to link different parts of wav file data
    """

    # The sample rate of the wav data, used for writing to file
    sample_rate: int

    # The actual amplitude data for the wav file
    # These are the data types I've seen in working pn this project, I'm sure there are more.
    wav_data: WavDataType

    # The name of the file, used for output/logging
    name: str


def read_wavs_scale_for_video(
    wavs: List[Path],
    vector_length: int,
    frames_per_second: Optional[float] = None,
    target_num_vectors: Optional[int] = None,
) -> WavDataType:
    """
    Reads multiple `.wav` files into memory, converts them to mono, and then re-samples the signals.
    For each of the files, the output is stretched/shrank in the time domain such that enough data
    is available to create one vector of `vector_length` samples for a video the length of the
    `.wav` file in time at the input `frames_per_second`. Additionally, If the scaled wav file is
    not evenly divisible in length by `vector_length`, zeros are added to the end.
    :param wavs: A list of paths to wav files.
    :param vector_length: The side length of the network, ex: 1024.
    :param frames_per_second: The FPS of the resulting video.
    :return: A single numpy array of the wav data of each of the individual files, scaled and then
    concatenated together.
    """

    wav_data = np.concatenate(
        [
            read_wav_scale_for_video(
                wav=path,
                vector_length=vector_length,
                frames_per_second=frames_per_second,
                target_num_vectors=target_num_vectors,
                pad_to_length=False,
            ).wav_data
            for path in wavs
        ]
    )

    return pad_array(
        wav_data,
        int(np.ceil(wav_data.shape[0] / vector_length) * vector_length),
    )


@overload
def read_wav_scale_for_video(
    wav: WavFileProperties,
    vector_length: int,
    frames_per_second: Optional[float] = None,
    target_num_vectors: Optional[int] = None,
    cache_path: Optional[Path] = None,
    pad_to_length: bool = True,
) -> WavFileProperties:
    ...


@overload
def read_wav_scale_for_video(
    wav: Path,
    vector_length: int,
    frames_per_second: Optional[float] = None,
    target_num_vectors: Optional[int] = None,
    cache_path: Optional[Path] = None,
    pad_to_length: bool = True,
) -> WavFileProperties:
    ...


def read_wav_scale_for_video(
    wav: Union[Path, WavFileProperties],
    vector_length: int,
    frames_per_second: Optional[float] = None,
    target_num_vectors: Optional[int] = None,
    cache_path: Optional[Path] = None,
    pad_to_length: bool = True,
) -> WavFileProperties:
    """
    Reads a `.wav` file into memory, converting it to mono, and then re-sampling the signal.
    The output is stretched/shrank in the time domain such that enough data is available to create
    one vector of `vector_length` samples for a video the length of the `.wav` file in time at the
    input `frames_per_second`.
    :param wav: Path to the `.wav` file.
    :param vector_length: The side length of the network, ex: 1024.
    :param frames_per_second: The FPS of the resulting video.
    :param cache_path: This is an expensive operation, write the result to this path.
    If this path exists, return it rather than computing the file.
    :param pad_to_length: If given, and If the scaled wav file is not evenly divisible in length
    by `vector_length`, zeros are added to the end.
    :return: The data, the new sample rate, and a label as a NT.
    """

    if frames_per_second is not None and target_num_vectors is not None:
        raise ValueError("Can't use both FPS mode and target vector count mode.")

    if frames_per_second is None and target_num_vectors is None:
        raise ValueError("Need to use FPS mode or target vector count mode.")

    if cache_path is not None:
        LOGGER.debug("Looking for cached audio.")
        if cache_path.exists():
            with open(str(cache_path), "rb") as read_file:
                LOGGER.info("Cached audio found. Loading.")
                output: WavFileProperties = pickle.load(read_file)
                LOGGER.info("Audio loaded.")
                return output
        else:
            LOGGER.warning("No audio cache found, will be created.")

    input_wav = read_wav_file(wav) if isinstance(wav, Path) else wav

    if len(input_wav.wav_data.shape) > 1:
        input_wav = WavFileProperties(
            wav_data=input_wav.wav_data.mean(axis=1),
            sample_rate=input_wav.sample_rate,
            name=f"{input_wav.name}_mono",
        )

    num_wav_samples = input_wav.wav_data.shape[0]

    if frames_per_second is not None:
        scaled_sample_rate = int(
            input_wav.sample_rate
            * (vector_length * (frames_per_second * (num_wav_samples / input_wav.sample_rate)))
            / num_wav_samples
        )
    elif target_num_vectors is not None:
        original_num_vectors = num_wav_samples / vector_length
        ratio = target_num_vectors / original_num_vectors
        scaled_sample_rate = float(input_wav.sample_rate) * ratio

    else:
        raise ValueError("Need to use FPS mode or target vector count mode.")

    scaled_wav = _scale_wav_to_sample_rate(input_wav, scaled_sample_rate)

    output = WavFileProperties(
        wav_data=pad_array(
            scaled_wav.wav_data,
            int(np.ceil(scaled_wav.wav_data.shape[0] / vector_length) * vector_length),
        )
        if pad_to_length
        else scaled_wav.wav_data,
        sample_rate=input_wav.sample_rate,
        name=f"{scaled_wav.name}_padded",
    )

    if cache_path is not None:
        with open(str(cache_path), "wb") as write_file:
            pickle.dump(output, write_file)

    return output


def read_wav_file(wav_path: Path, convert_to_32bit_float: bool = True) -> WavFileProperties:
    """
    Read in a wav file from disk return an NT representing the important parts of it.

    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8

    :param wav_path: The path to the wav file on disk.
    :param convert_to_32bit_float: If True, and if the input wav file is of a compatible type,
    the audio file will be scaled to a 32-bit float value between -1 and 1.
    :return: The NT.
    """
    sample_rate, wav_data = wavfile.read(str(wav_path))

    if convert_to_32bit_float and wav_data.dtype != np.float32:
        remapped_wav_data = partial(
            lambda input_range: remap_values_into_range(
                data=wav_data, input_range=input_range, output_range=(-1, 1)
            )
        )
        if wav_data.dtype == np.int32:
            wav_data = remapped_wav_data((-2147483648, 2147483647))
        elif wav_data.dtype == np.int16:
            wav_data = remapped_wav_data((-32768, 32767))
        elif wav_data.dtype == np.int8:
            wav_data = remapped_wav_data((0, 255))
        else:
            raise ValueError(
                "Cannot safely convert wav data to np.float32, unknown input format: "
                f"{wav_data.dtype}"
            )

        wav_data = np.array(list(wav_data)).astype(np.float32)

    return WavFileProperties(
        sample_rate=sample_rate, wav_data=wav_data, name=wav_path.with_suffix("").name
    )


def _scale_wav_to_sample_rate(
    wav_file: WavFileProperties, new_sample_rate: int
) -> WavFileProperties:
    """
    Scale a wav file to a new sample rate to speed it up or slow it down. The output
    NT will have the input sample rate.
    :param wav_file: The wav file to scale.
    :param new_sample_rate: The new sample rate.
    :return: A new WFP with the scaled data.
    """
    return WavFileProperties(
        wav_data=resampy.resample(
            wav_file.wav_data,
            sr_orig=wav_file.sample_rate,
            sr_new=new_sample_rate,
        ),
        sample_rate=wav_file.sample_rate,
        name=f"{wav_file.name}_scaled",
    )
