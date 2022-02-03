"""
Read a projection file back into memory
"""

import itertools
from pathlib import Path
from types import TracebackType
from typing import Iterator, Optional, Type, Union, cast

import h5py
import numpy as np
from h5py._hl.dataset import Dataset  # pylint: disable=protected-access
from h5py._hl.group import Group  # pylint: disable=protected-access

from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.logger_common import LOGGER
from gance.model_interface.model_functions import ModelInterface
from gance.projection.projection_types import (
    CompleteLatentsType,
    FlattenedNoisesType,
    complete_latents_to_matrix,
)
from gance.projection.projector_file_writer import (
    FINAL_IMAGE_GROUP_NAME,
    FINAL_LATENTS_GROUP_NAME,
    IMAGES_HISTORIES_GROUP_NAME,
    LATENTS_HISTORIES_GROUP_NAME,
    NOISES_HISTORIES_GROUP_NAME,
    TARGET_IMAGES_GROUP_NAME,
    ProjectionAttributes,
)
from gance.vector_sources.vector_types import ConcatenatedMatrices, MatricesLabel, SingleMatrix


def _double_iter(
    group: Group, inner_matrix: bool
) -> Iterator[Iterator[Union[np.ndarray, np.ndarray]]]:
    """
    Helper function.
    :param group: Group to search
    :return: Yield iterators that produce numpy arrays.
    """
    yield from (
        _datasets_in_group(group=group, inner_matrix=inner_matrix)
        for group in _groups_in_group(group)
    )


def _types_in_group(
    group: Group, h5_type: Union[Group, Dataset]
) -> Iterator[Union[Group, Dataset]]:
    """
    Get all of the given type in the group.
    :param group: The group to search.
    :param h5_type: The type of item to return
    :return: Yield the items of the given type in the group.
    """

    for name, item in sorted(
        filter(lambda name_item: isinstance(name_item[1], h5_type), group.items()),
        key=lambda name_set: int(name_set[0].split("_")[-1]),
    ):
        LOGGER.debug(f"Yielding {h5_type.__name__}: {name}")
        yield item


def _groups_in_group(group: Group) -> Iterator[Group]:
    """
    Get the subgroups in the given group. Does not do recursion.
    :param group: Group to search.
    :return: Yield top level subgroups.
    """
    yield from _types_in_group(group, Group)


def _datasets_in_group(
    group: Group, inner_matrix: bool
) -> Iterator[Union[RGBInt8ImageType, SingleMatrix]]:
    """
    For the datasets in the group, yield them as numpy arrays.
    :param group: Group to search.
    :return: Numpy arrays in group
    """

    def dataset_for_output(
        dataset: Dataset,
    ) -> Union[RGBInt8ImageType, SingleMatrix]:
        """
        Covert the dataset to a numpy array, and if it's a latent, pull out the matrix.
        :param dataset: To process.
        :return: As a numpy array.
        """
        as_numpy_array: Union[RGBInt8ImageType, CompleteLatentsType] = np.array(dataset)
        if inner_matrix:
            return complete_latents_to_matrix(as_numpy_array)
        return RGBInt8ImageType(as_numpy_array)

    yield from (dataset_for_output(dataset) for dataset in _types_in_group(group, Dataset))


class ProjectionFileReader:  # pylint: disable=too-many-instance-attributes
    """
    Points to everything available for a given projection.
    """

    def __init__(self, projection_file_path: Path) -> None:
        """
        :param projection_file_path: Path to the projection file on disk.
        """

        self._file = h5py.File(name=str(projection_file_path), mode="r")

        a = ProjectionAttributes.from_dict(self._file.attrs)  # type: ignore # pylint: disable=no-member
        self._projection_attributes: ProjectionAttributes = a

        self._target_images: ImageSourceType = cast(
            ImageSourceType,
            _datasets_in_group(self._file[TARGET_IMAGES_GROUP_NAME], inner_matrix=False),
        )

        self._final_latents: Iterator[SingleMatrix] = cast(
            Iterator[SingleMatrix],
            _datasets_in_group(self._file[FINAL_LATENTS_GROUP_NAME], inner_matrix=True),
        )

        self._final_images: ImageSourceType = cast(
            ImageSourceType,
            _datasets_in_group(self._file[FINAL_IMAGE_GROUP_NAME], inner_matrix=False),
        )

        self._latents_histories: Iterator[Iterator[SingleMatrix]] = cast(
            Iterator[Iterator[SingleMatrix]],
            _double_iter(self._file[LATENTS_HISTORIES_GROUP_NAME], inner_matrix=True),
        )

        self._noises_histories: Iterator[Iterator[FlattenedNoisesType]] = cast(
            Iterator[Iterator[FlattenedNoisesType]],
            _double_iter(self._file[NOISES_HISTORIES_GROUP_NAME], inner_matrix=False),
        )

        self._images_histories: Iterator[ImageSourceType] = cast(
            Iterator[ImageSourceType],
            _double_iter(self._file[IMAGES_HISTORIES_GROUP_NAME], inner_matrix=False),
        )

    @property
    def projection_attributes(self: "ProjectionFileReader") -> ProjectionAttributes:
        """
        Contains metadata about the projections, and the source material that was fed in.
        :return:
        """
        return self._projection_attributes

    @property
    def target_images(self: "ProjectionFileReader") -> ImageSourceType:
        """
        The image that this projection was targeting.
        Should be able to see some visual resemblance.
        :return:
        """
        return self._target_images

    @property
    def final_latents(self: "ProjectionFileReader") -> Iterator[SingleMatrix]:
        """
        The final entry in `latents_history`. This is the most interesting matrix and is exposed
        so you don't have to process all of `latents_history` to get to this.
        :return:
        """
        return self._final_latents

    @property
    def final_images(self: "ProjectionFileReader") -> ImageSourceType:
        """
        These are the final resulting images from each projection, retrained here for convenience.
        :return:
        """
        return self._final_images

    @property
    def latents_histories(self: "ProjectionFileReader") -> Iterator[Iterator[SingleMatrix]]:
        """
        The latents over time. Early in list = early in projection.
        :return:
        """
        return self._latents_histories

    @property
    def noises_histories(self: "ProjectionFileReader") -> Iterator[Iterator[FlattenedNoisesType]]:
        """
        The noises over time. Early in list = early in projection.
        :return:
        """
        return self._noises_histories

    @property
    def images_histories(self: "ProjectionFileReader") -> Iterator[ImageSourceType]:
        """
        The images produced by projection over time. Early in list = early in projection.
        :return:
        """
        return self._images_histories

    def close(self: "ProjectionFileReader") -> None:
        """

        :return:
        """
        self._file.close()

    def __enter__(self: "ProjectionFileReader") -> "ProjectionFileReader":
        return self

    def __exit__(
        self: "ProjectionFileReader",
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> bool:
        self.close()

        if excinst is not None:
            raise excinst

        return True


def verify_projection_file_assumptions(projection_file_path: Path) -> None:
    """
    Checks to make sure that the given projection file doesn't violate any of the assumptions
    this project is built on.
    :param projection_file_path:
    :raises: An Assertion error of any of the assumptions have been violated.
    :return: None
    """

    def verify_all_latents_are_the_same(latents: Iterator[SingleMatrix]) -> None:
        """
        Helper function.
        :param latents: Iterator of latents.
        :return: None
        """
        for matrices in latents:
            first_matrix = matrices[0]
            for matrix in matrices:
                assert np.array_equal(first_matrix, matrix)

    with load_projection_file(projection_file_path) as reader:
        verify_all_latents_are_the_same(reader.final_latents)
        if reader.projection_attributes.latents_histories_enabled:
            for history_of_latents in reader.latents_histories:
                verify_all_latents_are_the_same(history_of_latents)


def _iterator_of_single_matrix_to_matrices_label(
    iterator: Iterator[SingleMatrix], label: str
) -> MatricesLabel:
    """
    Read all of the latents in an iterator. Take the matrices out of each of these latents.
    Concatenate the result and return it as a `MatricesLabel`
    Each latent in `iterator` is concatenated to form the `.data` member of the resulting NT.
    :param iterator: Iterator, probably from a projection file.
    :param label: Label field of the resulting NT.
    :return: The NT.
    """

    try:
        first_matrix = next(iterator)
    except StopIteration as e:
        raise StopIteration(f"Iterator labeled: {label} was empty!") from e

    return MatricesLabel(
        data=ConcatenatedMatrices(np.concatenate([first_matrix] + list(iterator), axis=-1)),
        vector_length=first_matrix.shape[-1],
        label=label,
    )


def final_latents_matrices_label(reader: ProjectionFileReader) -> MatricesLabel:
    """
    Consume the final latents of a Projection reader creating a standard for plotting etc.
    :param reader: Where the final latents are stored.
    :return: The New NT
    """

    return _iterator_of_single_matrix_to_matrices_label(
        reader.final_latents,
        label=(
            f"{Path(reader.projection_attributes.original_target_path).name} "
            f"proj by {Path(reader.projection_attributes.original_model_path).name}"
        ),
    )


def _yield_position_from_sub_iterables(
    iterator_of_iterators: Iterator[Iterator[SingleMatrix]], position: int
) -> Iterator[SingleMatrix]:
    """
    Parse an iterator of iterators, yield the value at the given position as a new iterator.
    Stops producing values once a sub iterator does not have a value at the given position.
    :param iterator_of_iterators: Like a latent history or an image history or a noises history.
    An iterator that yields another iterator of numpy arrays.
    :param position: The position of the sub array to emit.
    :return: The iterator of the value of the given position in the sub array.
    """

    for iterator in iterator_of_iterators:
        try:
            yield next(itertools.islice(iterator, position, None))
        except StopIteration:
            return


def projection_history_step_matrices_label(
    reader: ProjectionFileReader, projection_step: int
) -> MatricesLabel:
    """
    Retrieve the given step from every latent history in the projection file, return the
    concatenated result as a standard NT.
    :param reader: Where the latent histories are stored.
    :param projection_step: The step to take from the projection history.
    :return: The New NT
    """

    complete_latents_at_position = _yield_position_from_sub_iterables(
        iterator_of_iterators=reader.latents_histories, position=projection_step
    )

    return _iterator_of_single_matrix_to_matrices_label(
        iterator=complete_latents_at_position,
        label=(
            f"{Path(reader.projection_attributes.original_target_path).name} "
            f"proj by {Path(reader.projection_attributes.original_model_path).name} "
            f"step {projection_step}"
        ),
    )


def model_outputs_at_projection_step(
    projection_file_path: Path,
    model_interface: ModelInterface,
    projection_step_to_take: int,
) -> Iterator[RGBInt8ImageType]:
    """
    At the given step in the projection history for each frame in the projection file,
    feed the matrices into the model and return the image.
    Note: This is a one-shot helper function. If you need to do more operations on the file,
    it's faster to use the `ProjectionFileReader` interface.
    :param projection_file_path: Path to file to read.
    :param model_interface: Latents from the projection history will be fed into this model.
    :param projection_step_to_take: This step of projection will be retrieved and fed to the model.
    :return: Resulting images for each frame in the file.
    """

    with load_projection_file(projection_file_path) as reader:

        yield from [
            model_interface.create_image_matrix(latents)
            for latents in _yield_position_from_sub_iterables(
                iterator_of_iterators=reader.latents_histories, position=projection_step_to_take
            )
        ]


def load_final_latents_matrices_label(projection_file_path: Path) -> MatricesLabel:
    """
    Simple wrapper, loads the final latents for processing.
    Note: This is a one-shot helper function. If you need to do more operations on the file,
    it's faster to use the `ProjectionFileReader` interface.
    :param projection_file_path: See call docs.
    :return: See call docs.
    """

    with load_projection_file(projection_file_path) as reader:
        return final_latents_matrices_label(reader)


def final_latents_at_frame(projection_file_path: Path, frame_number: int) -> SingleMatrix:
    """
    Simple wrapper, loads the final latents for processing.
    Note: This is a one-shot helper function. If you need to do more operations on the file,
    it's faster to use the `ProjectionFileReader` interface.
    :param projection_file_path: See call docs.
    :return: See call docs.
    """

    with load_projection_file(projection_file_path) as reader:
        return next(itertools.islice(reader.final_latents, frame_number, frame_number + 1))


def model_outputs_at_final_latents(
    projection_file_path: Path, model_interface: ModelInterface
) -> Iterator[RGBInt8ImageType]:
    """
    For the final latents for each frame in the projection file,
    feed the matrices into the model and return the image.
    Note: This is a one-shot helper function. If you need to do more operations on the file,
    it's faster to use the `ProjectionFileReader` interface.
    :param projection_file_path: Path to file to read.
    :param model_interface: Latents from the projection history will be fed into this model.
    :return: The results from the model.
    """

    with load_projection_file(projection_file_path) as reader:
        yield from [
            model_interface.create_image_matrix(final_latents)
            for final_latents in reader.final_latents
        ]


def final_images(projection_file_path: Path) -> Iterator[RGBInt8ImageType]:
    """
    Retrieves the final images for each frame in the projection file.
    Note: This is a one-shot helper function. If you need to do more operations on the file,
    it's faster to use the `ProjectionFileReader` interface.
    :param projection_file_path: Path to file to read.
    :return: The images per frame.
    """

    with load_projection_file(projection_file_path) as reader:
        yield from reader.final_images


def target_images(projection_file_path: Path) -> Iterator[RGBInt8ImageType]:
    """
    Retrieves the target images for each frame in the projection file.
    Note: This is a one-shot helper function. If you need to do more operations on the file,
    it's faster to use the `ProjectionFileReader` interface.
    :param projection_file_path: Path to file to read.
    :return: The images per frame.
    """

    with load_projection_file(projection_file_path) as reader:
        yield from reader.target_images


def projection_attributes(projection_file_path: Path) -> ProjectionAttributes:
    """
    Read the projection attributes of a given projection file.
    :param projection_file_path: Path to file to read.
    :return: The dataclass.
    """

    with load_projection_file(projection_file_path) as reader:
        return reader.projection_attributes


def load_projection_file(projection_file_path: Path) -> ProjectionFileReader:
    """
    Creates a context manager that can then be used to read data from a given projection file.
    :param projection_file_path: Path to projection file.
    :return: An NT used to read the data out of the file. Read those docs for more info.
    """

    return ProjectionFileReader(projection_file_path=projection_file_path)
