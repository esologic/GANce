"""
Fucker
"""
import itertools
from time import time

from gance import assets
from music_into_models import projection_file

if __name__ == "__main__":

    top_output_dir = assets.OUTPUT_DIRECTORY.joinpath(f"prod_parameter_checks_{int(time())}")
    top_output_dir.mkdir(exist_ok=True)

    alphas = [0.1, 0.25, 0.5, 0.75]

    models_dirs = ["./example_models", "./good_only", "./single_model"]

    fft_roll_enableds = [True, False]

    fft_amplitude_ranges = [
        (-1, 1),
        (-4, 4),
        (-10, 10),
    ]

    fft_depths = [1, 5, 10, 16]

    for alpha, model_dir, fft_roll_enabled, fft_amplitude_range, fft_depth in itertools.product(
        *[alphas, models_dirs, fft_roll_enableds, fft_amplitude_ranges, fft_depths]
    ):

        name = (
            "_".join(
                [
                    str(item)
                    for item in [alpha, model_dir, fft_roll_enabled, fft_amplitude_range, fft_depth]
                ]
            )
            .replace(".", "")
            .replace("/", "")
            .replace("-", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "")
            .replace(",", "")
        )

        print("Computing", name)

        output = top_output_dir.joinpath(name)
        output.mkdir(exist_ok=True)

        projection_file(
            wav=str(assets.NOVA_PATH),
            output_directory=str(output),
            models_directory=model_dir,
            vector_length=None,
            index=None,
            frames_to_visualize=None,
            alpha=alpha,
            projection_file_path=assets.NOVA_PROJECTION_FILE,
            output_fps=60.0,
            debug_2d=False,
            note="messing with spectro, increasing gzip smoothing window",
            fft_roll_enabled=fft_roll_enabled,
            fft_amplitude_range=fft_amplitude_range,
            fft_depth=fft_depth,
        )
