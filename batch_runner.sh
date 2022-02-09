#!/usr/bin/env bash

declare -a PathArray=("./models.json" "./models.json")

for path in ${PathArray[*]}; do
  python music_into_models.py projection-file-blend \
  --wav ./gance/assets/audio/nova_prod.wav \
  --output_path ./gance/assets/output/model_index_runs/"$(date +%s)"_test.mp4 \
  --output_fps 30 \
  --output_side_length 1024 \
  --debug_path ./gance/assets/output/model_index_runs/"$(date +%s)"_test_debug.mp4 \
  --debug_window 300 \
  --alpha 0.35 \
  --fft_roll_enabled \
  --fft_amplitude_range -5 5 \
  --projection_file_path ./gance/assets/projection_files/resumed_prod_nova_3-1.hdf5 \
  --blend_depth 10 --complexity_change_rolling_sum_window 30 \
  --complexity_change_threshold 100 \
  --phash_distance 55 \
  --bbox_distance 55 \
  --track_length 10 \
  --blend_depth 10 \
  --frames_to_visualize 1 \
  --run_config ./gance/assets/output/model_index_runs/"$(date +%s)"_config.json \
  --models_json $path
done