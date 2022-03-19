from pathlib import Path
from typing import List, NamedTuple

from music_into_networks import _projection_file_blend_api
from gance.network_interface.network_functions import MultiNetwork, parse_network_paths


class BatchVariation(NamedTuple):
    wavs: List[str]
    projection_file_path: str
    name: str


if __name__ == "__main__":

    inputs = [
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/01 - yellow intro (pass 2).wav",
                "./gance/assets/audio/masters/02 - sub domain grass 4th (pass 2).wav",
                "./gance/assets/audio/masters/03 - NOD3 .work (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/prod_intro_sub_domain_node_work-2.hdf5",
            name="intro_subdomain_nodework",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/06 - leaveit - pass 02 (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/resumed_prod_leave_it-1.hdf5",
            name="leave_it",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/12 - buzzz (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/resumed_buzzzz_1.hdf5",
            name="buzzz",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/11 - town (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/2_resumed_prod_town_1.hdf5",
            name="town",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/13 - change - pass 02 (pass 2).wav",
                "./gance/assets/audio/masters/14 - usb hard disk - pass 02 (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/prod_change_usb_hard_disk-1.hdf5",
            name="change_usb_hard_disk",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/10 - balloons (pop) (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/prod_ballons-1.hdf5",
            name="ballons",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/16 - birds of solaris (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/prod_birds_of_solaris-1.hdf5",
            name="birds_of_solaris",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/04 - downinthe_(56r FLK of) (pass 2).wav",
                "./gance/assets/audio/masters/05 - familiar___+ (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/prod_down_in_the_familiar_cropped-1.hdf5",
            name="down_in_the_familiar",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/08 - foam pt. 2 (pass 2).wav",
                "./gance/assets/audio/masters/05 - familiar___+ (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/prod_foam_part_2_forthought_titan-1.hdf5",
            name="foam_forethought_titan",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/15 - NOVA (end 2) (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/resumed_prod_nova_3-1.hdf5",
            name="nova",
        ),
        BatchVariation(
            wavs=["./gance/assets/audio/masters/07 - fourth dance theme (pass 2).wav"],
            projection_file_path="./gance/assets/projection_files/third_fourth.hdf5",
            name="fourth_dance_theme",
        ),
    ]

    output_dir = Path("./gance/assets/output/final_runs/4/")
    output_dir.mkdir(exist_ok=True)

    network_paths = parse_network_paths(
        networks_directory=None, networks=None, networks_json="./prod_networks.json"
    )

    with MultiNetwork(network_paths=network_paths) as multi_networks:

        for batch in inputs:

            _projection_file_blend_api(
                wav=batch.wavs,
                output_path=str(output_dir.joinpath(f"{batch.name}.mp4")),
                multi_networks=multi_networks,
                frames_to_visualize=None,
                output_fps=60,
                output_side_length=1024,
                debug_path=str(output_dir.joinpath(f"{batch.name}_debug.mp4")),
                debug_window=300,
                alpha=0.5,
                fft_roll_enabled=True,
                fft_amplitude_range=(-5, 5),
                run_config=str(output_dir.joinpath(f"{batch.name}.json")),
                projection_file_path=batch.projection_file_path,
                blend_depth=10,
                complexity_change_rolling_sum_window=30,
                complexity_change_threshold=5000,
                phash_distance=25,
                bbox_distance=50,
                track_length=5,
            )
