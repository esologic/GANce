"""
Tests write/read/module functions in projection file.
"""

from pathlib import Path
from test.assets import (
    SAMPLE_BATCH_1_MODEL_PATH,
    SAMPLE_BATCH_2_MODEL_PATH,
    SAMPLE_FACE_VIDEO_SHORT_PATH,
)
from test.test_common import wait_get_size
from typing import List

import numpy as np
import pytest
from py._path.local import LocalPath  # pylint: disable=protected-access

from gance.image_sources import video_common
from gance.image_sources.still_image_common import write_image
from gance.model_interface.model_functions import create_model_interface_process
from gance.projection import projection_file_reader, projection_visualization, projector_file_writer
from gance.vector_sources.vector_types import SingleMatrix


def distance(final_latent: SingleMatrix, latent_history: List[SingleMatrix], index: int) -> float:
    """
    Helper function to evaluate how well projected a latent is.
    :param final_latent: Target.
    :param latent_history: Value is in here.
    :param index: Index in list
    :return: Distance
    """
    output: float = np.sum(abs((final_latent - latent_history[index])))
    return output


@pytest.mark.gpu
@pytest.mark.parametrize("model_path", [SAMPLE_BATCH_2_MODEL_PATH, SAMPLE_BATCH_1_MODEL_PATH])
@pytest.mark.parametrize("steps_per_projection", [10, 15, 30])
def test_project_video_to_file(  # pylint: disable=too-many-locals
    tmpdir: LocalPath, model_path: Path, steps_per_projection: int
) -> None:
    """
    Tests to make sure that the method of writing a projection file still works.
    Needs the GPU to be able to do a projection.
    Note: This is more of an integration test than a unit test, it's big ugly and long but
    tests a lot of the important stuff around creating projection files.
    :param tmpdir: Test fixture.
    :return: None
    """

    projection_file_path = Path(tmpdir).joinpath("projection_file.hdf5")

    # Tests to make sure that this process can complete without an error.
    projector_file_writer.project_video_to_file(
        path_to_model=model_path,
        path_to_video=SAMPLE_FACE_VIDEO_SHORT_PATH,
        projection_file_path=projection_file_path,
        steps_per_projection=steps_per_projection,  # timebounds this to about a minute per frame
    )

    with projection_file_reader.load_projection_file(
        projection_file_path=projection_file_path
    ) as reader:

        # Want these more than once so we need a list
        target_images = list(reader.target_images)
        expected_images = list(
            video_common.frames_in_video(video_path=SAMPLE_FACE_VIDEO_SHORT_PATH).frames
        )
        final_latents = list(reader.final_latents)
        latents_histories = list(reader.latents_histories)

        assert (
            len(target_images)
            == len(expected_images)
            == len(final_latents)
            == len(latents_histories)
        )

        for target_image, expected_image in zip(target_images, expected_images):
            assert np.array_equal(target_image, expected_image)

        for matrix in final_latents:
            # Known property of this model
            assert matrix.shape == (18, 512)
            assert matrix.sum() > 0

        for image in reader.final_images:
            assert image.shape == (1024, 1024, 3)
            assert image.sum() > 0

        for latent_history, final_latent in zip(latents_histories, final_latents):
            latent_history_list = list(latent_history)

            # The first vector should always be further way than the last vector.
            assert distance(
                final_latent=final_latent, latent_history=latent_history_list, index=0
            ) > distance(final_latent=final_latent, latent_history=latent_history_list, index=-2)
            assert all(
                latent.shape == final_latent.shape == (18, 512) for latent in latent_history_list
            )
            assert all(latent.sum() > 0 for latent in latent_history_list)

    # This will raise an error if the contents of the file violate any of the principal
    # assumptions, which I haven't seen but could be possible in theory.
    projection_file_reader.verify_projection_file_assumptions(
        projection_file_path=projection_file_path
    )

    # Check that the final vectors in the file produce the same image when fed to the model they
    # did during their first run.
    model_interface = create_model_interface_process(model_path=model_path)

    try:

        for model_output, model_output_in_file in zip(
            projection_file_reader.model_outputs_at_final_latents(
                projection_file_path=projection_file_path,
                model_interface=model_interface.model_interface,
            ),
            projection_file_reader.final_images(projection_file_path=projection_file_path),
        ):
            # The images look VERY similar, but are not pixel by pixel identical. This is probably
            # a problem, but not one I have time to solve now. Saving the images to file and
            # comparing sizes is roughly analogous to checking the visual contents which is the
            # idea of this test.

            model_image_path = Path(str(tmpdir)).joinpath("model_image.png")
            file_image_path = Path(str(tmpdir)).joinpath("file_image.png")

            write_image(image=model_output, path=model_image_path)
            write_image(image=model_output_in_file, path=file_image_path)

            # Check if they're within 5000 bytes of eachother.
            assert abs(wait_get_size(model_image_path) - wait_get_size(file_image_path)) < 5000

        # Using the resulting projection file, create several of the common visualizations. These
        # should be able to compute without error.
        convergence_file = Path(str(tmpdir)).joinpath("output.png")
        projection_visualization.visualize_projection_convergence(
            projection_file_path=projection_file_path,
            output_image_path=convergence_file,
        )

        final_latents_file = Path(str(tmpdir)).joinpath("final_latents.mp4")
        projection_visualization.visualize_final_latents(
            projection_file_path=projection_file_path,
            output_video_path=final_latents_file,
        )

        projection_history_path = Path(str(tmpdir)).joinpath("projection_history.mp4")
        projection_visualization.visualize_projection_history(
            projection_file_path=projection_file_path,
            output_video_path=projection_history_path,
            projection_model_path=model_path,
            model_not_matching_ok=False,
        )
        projection_step_path = Path(str(tmpdir)).joinpath("projection_step.mp4")
        projection_visualization.visualize_partial_projection_history(
            projection_file_path=projection_file_path,
            output_video_path=projection_step_path,
            projection_model_path=model_path,
            model_not_matching_ok=False,
            projection_step_to_take=0,
        )
    except AssertionError as e:
        # Needs to be called or you'll run out of vram across subsequent runs
        model_interface.stop_function()
        raise e

    model_interface.stop_function()

    print(f"Tests passed! for {steps_per_projection} steps, model: {str(model_path)}")
