"""
Unit tests around working with networks.
"""

from test.assets import SAMPLE_BATCH_1_NETWORK_PATH
from typing import Union

import numpy as np
import pytest
from pytest_mock import MockFixture

from gance.gance_types import RGBInt8ImageType
from gance.network_interface.network_functions import (
    MultiNetwork,
    NetworkInterface,
    NetworkInterfaceInProcess,
    create_network_interface_process,
)
from gance.vector_sources.vector_types import SingleMatrix, SingleVector


def fake_stop_function() -> None:
    """
    No-Op, target for mocking.
    :return: None
    """


@pytest.mark.parametrize("load", [True, False])
def test_multi_network_unloaded_leads_to_errors(load: bool, mocker: MockFixture) -> None:
    """
    Test to make sure that `Multinetwork` will raise ValueErrors if you try to access the network
    without first calling the `load()` method directly or with the context manager.
    Also makes sure that the context manager correctly loads and unloads the underlying network.
    :param load: if the network should actually be loaded or not.
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

    patched_stop = mocker.patch("test.test_network_functions.fake_stop_function")

    patched_load = mocker.patch(
        "gance.network_interface.network_functions.create_network_interface_process",
        return_value=NetworkInterfaceInProcess(
            NetworkInterface(
                create_image_vector=fake_vector_function,
                create_image_matrix=fake_vector_function,
                create_image_generic=fake_vector_function,
                expected_vector_length=expected_vector_length,
            ),
            stop_function=fake_stop_function,
        ),
    )

    def make_network() -> MultiNetwork:
        """
        Helper function to create the network for the test.
        :return: The multi network for testing.
        """
        return MultiNetwork(network_paths=[SAMPLE_BATCH_1_NETWORK_PATH])

    assert not patched_stop.called

    if load:
        with make_network() as multi_network:
            assert multi_network.expected_vector_length == expected_vector_length
            assert (
                multi_network.indexed_create_image_vector(
                    index=0, data=SingleVector(np.zeros((10,)))
                )
                == expected_image
            ).all()
        # Verifies that the context manager is calling the load/stop functions.
        assert patched_load.called
        assert patched_stop.called
    else:
        multi_network = make_network()
        with pytest.raises(ValueError):
            # This won't execute, just need something here that accesses the value.
            print(multi_network.expected_vector_length)
        with pytest.raises(ValueError):
            multi_network.indexed_create_image_vector(index=0, data=SingleVector(np.zeros((10,))))
        assert not patched_load.called
        assert not patched_stop.called


@pytest.mark.gpu
@pytest.mark.timeout(60)
def test_network_interface_process_stop() -> None:
    """
    Check to make sure an image can be created and then shut down.
    :return: None
    """

    network_interface_process = create_network_interface_process(
        network_path=SAMPLE_BATCH_1_NETWORK_PATH
    )
    image = network_interface_process.network_interface.create_image_vector(
        data=SingleVector(
            np.zeros((network_interface_process.network_interface.expected_vector_length,))
        )
    )
    assert image.shape == (1024, 1024, 3)
    assert np.sum(image) > 0
    network_interface_process.stop_function()
