import os
from pathlib import Path
from typing import List, NamedTuple

from jinja2 import Template


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
            wavs=["./gance/assets/audio/masters/07 - fourth dance theme (pass 2).wav"],
            projection_file_path="./gance/assets/projection_files/third_fourth.hdf5",
            name="fourth_dance_theme",
        ),
        BatchVariation(
            wavs=[
                "./gance/assets/audio/masters/15 - NOVA (end 2) (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/resumed_prod_nova_3-1.hdf5",
            name="nova",
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
                "./gance/assets/audio/masters/06 - leaveit - pass 02 (pass 2).wav",
            ],
            projection_file_path="./gance/assets/projection_files/resumed_prod_leave_it-1.hdf5",
            name="leave_it",
        ),
    ]

    output_dir = Path("./gance/assets/output/final_runs/7/")
    output_dir.mkdir(exist_ok=True)

    commands = [
        " ".join(
            [
                "./venv/bin/python /home/gpu/gance/music_into_networks.py projection-file-blend",
                *[f'--wav "{wav}"' for wav in command_parameters.wavs],
                f"--output-path {str(output_dir.joinpath(f'{command_parameters.name}.mp4'))}",
                f"--run-config {str(output_dir.joinpath(f'{command_parameters.name}.json'))}",
                f"--log log.txt",
                f"--projection-file-path {command_parameters.projection_file_path}",
                f"--networks-json ./prod_networks.json",
                f"--output-fps 60",
                f"--output-side-length 1080",
                f"--alpha 0.25",
                f"--fft-roll-enabled",
                f"--fft-amplitude-range -5 5",
                f"--blend-depth 12",
                f"--phash-distance 25",
                f"--bbox-distance 50",
                f"--track-length 5",
            ]
        )
        + os.linesep
        for command_parameters in inputs
    ]

    with open("./batch.sh.jinja2") as f:
        template = Template(f.read())

        with open("./batch.sh", "w") as f:
            f.write(template.render(commands=" ".join(commands)))
