"""
Common types used in projection
"""

from typing import NewType, Tuple

from gance.vector_sources.vector_types import SingleMatrix

# Shape (1, Any, Any)
CompleteLatentsType = NewType("CompleteLatentsType", "np.ndarray[np.float32]")  # type: ignore

# TODO: Need to verify this type, placeholder for now
TFRecordsType = NewType("TFRecordsType", "np.ndarray[float32]")  # type: ignore

# TODO: double check if this type is correct.
NoisesType = NewType("NoisesType", "np.ndarray[np.float32]")  # type: ignore

FlattenedNoisesType = NewType("FlattenedNoisesType", "np.ndarray[np.float32]")  # type: ignore
NoisesShapesType = NewType("NoisesShapesType", Tuple[Tuple[int, int, int, int], ...])


def complete_latents_to_matrix(complete_latents: CompleteLatentsType) -> SingleMatrix:
    """
    Canonical way to make this conversion.
    :param complete_latents: Contains the matrix.
    :return: The matrix.
    """
    return SingleMatrix(complete_latents[0])
