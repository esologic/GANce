"""
Unit tests around working with models.
"""

from test.assets import SAMPLE_BATCH_1_MODEL_PATH
from typing import Union

import numpy as np
import pytest
from pytest_mock import MockFixture

from gance.gance_types import RGBInt8ImageType
from gance.model_interface.model_functions import (
    ModelInterface,
    ModelInterfaceInProcess,
    MultiModel,
)
from gance.vector_sources.vector_types import SingleMatrix, SingleVector


def fake_stop_function() -> None:
    """
    No-Op, target for mocking.
    :return: None
    """


@pytest.mark.parametrize("load", [True, False])
def test_multi_model_unloaded_leads_to_errors(load: bool, mocker: MockFixture) -> None:
    """
    Test to make sure that `MultiModel` will raise ValueErrors if you try to access the model
    without first calling the `load()` method directly or with the context manager.
    Also makes sure that the context manager correctly loads and unloads the underlying model.
    :param load: if the model should actually be loaded or not.
    :param mocker: mocking fixture.
    :return: None
    """

    expected_vector_length = 10
    expected_image = RGBInt8ImageType(np.zeros((10, 10, 3)))

    def fake_vector_function(
        data: Union[SingleVector, SingleMatrix]  # pylint: disable=unused-argument
    ) -> RGBInt8ImageType:
        """
        Mock vector function that will always create the same, fake "image".
        :param data: Input vector, ignored here.
        :return: Image, the same, test one every time.
        """
        return expected_image

    patched_stop = mocker.patch("test.test_model_functions.fake_stop_function")

    patched_load = mocker.patch(
        "gance.model_interface.model_functions.create_model_interface_process",
        return_value=ModelInterfaceInProcess(
            ModelInterface(
                create_image_vector=fake_vector_function,
                create_image_matrix=fake_vector_function,
                create_image_generic=fake_vector_function,
                expected_vector_length=expected_vector_length,
            ),
            stop_function=fake_stop_function,
        ),
    )

    def make_model() -> MultiModel:
        """
        Helper function to create the model for the test.
        :return: The multi model for testing.
        """
        return MultiModel(model_paths=[SAMPLE_BATCH_1_MODEL_PATH])

    assert not patched_stop.called

    if load:
        with make_model() as multi_model:
            assert multi_model.expected_vector_length == expected_vector_length
            assert (
                multi_model.indexed_create_image_vector(index=0, data=SingleVector(np.zeros((10,))))
                == expected_image
            ).all()
        # Verifies that the context manager is calling the load/stop functions.
        assert patched_load.called
        assert patched_stop.called
    else:
        multi_model = make_model()
        with pytest.raises(ValueError):
            # This won't execute, just need something here that accesses the value.
            print(multi_model.expected_vector_length)
        with pytest.raises(ValueError):
            multi_model.indexed_create_image_vector(index=0, data=SingleVector(np.zeros((10,))))
        assert not patched_load.called
        assert not patched_stop.called
