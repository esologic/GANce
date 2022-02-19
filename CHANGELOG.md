# Changelog

0.19.0 - (2022-02-XX)
------------------

* Added a few more arguments to `music_into_models.py` to be able to make selecting the production
list of models easier. Now input args can be saved to file, and lists of models can be passed in
as json.


0.18.0 - (2022-02-08)
------------------

* For music/video synthesis, introduced concept of the overlay, which is taking regions of the
target images from a projection file, and displaying them over the synthesized output from the
network. This gives the viewer a clue as to how the video was generated, and creates another
interesting dimension of the resulting videos.
* Refactored synthesis pipeline invoked in `music_into_models.py` as a series of iterators. This
enabled introducing the overlay concept while still being able to efficiently use memory. This
stream processing approach required a few modifications to existing concepts to be able to handle
streams rather than complete lists etc.
* Introduced video writers that can write video/audio to a file, and then forward the resulting 
stream of frames to be re-used. 
* Broke out some more standard functionality for dealing with images and videos. Introduced
a set of types for single frames, and streams of frames (videos) as well as resolutions etc.
* The `visualize_final_latents` command in `read_projection_file.py` can now accept multiple input 
audio files that will be appended to each other and then added to the resulting video.
* Finally, resolved the dlib import problem by creating `faces.py`, which enables the import of the
offending library right before use, rather than on loading.


0.17.0 - (2021-11-22)
------------------

* Breaking up project and renaming it `GANce` for publication on GitHub.
* Added a number of examples for a blog post explaining this 
project [here](https://www.esologic.com/gance/).


0.16.1 - (2021-10-28)
------------------

* Fixed a bug in synthesis file reader/writer. Both now use the standard vector types. 


0.16.0 - (2021-10-23)
------------------

* This version will create the final projection files for the yellow album cover project.
* In `project_video_to_file.py`, added ability to process multiple input videos at once. Either
a directory of videos or multiple videos passed via CLI. Added unit testing for this as well.
* Created a few projection specific docker containers in `docker-compose.yml` based on the `develop`
model.
* Added a very small cli, `read_projection_file.py` to read a projection file and turn the final
images vs. target images into a video.


0.15.0 - (2021-10-17)
------------------

* In `project_video_to_file.py`, added ability to override the fps of the input video. Use with
caution.
* Added type system for key data types. Images, Vectors/Matrices, Latents etc. See comment in
readme but these are mostly to aid in documentation.
* Cleaned up repo structure a bit during refactor to add types, but still some things that need to
be deleted.
* Added integration style testing of projection file reader and writer that require GPU.

0.14.0 - (2021-10-12)
------------------

* Added visualization/numeric analysis functionality to decide the projection steps and projection
FPS that will be used in the production projections.
* Added a number of helper functions to make working with projection files more expedient.
* Fixed bug in `synthesis_file_into_models.py`


0.13.0 - (2021-09-20)
------------------

* Added a CLI to, given a directory of models, create a number of random output images, some with
and some without faces. This tool, `images_from_models.py` will be used to create the prod album
cover.
* Introduced notion of a synthesis file, which is a json file produced alongside the images 
previously mentioned which records the associated vector and model used to create the image.
* Added a CLI `synthesis_file_into_models.py` to read one of these synthesis file jsons and input
the vector into a list of models to see the "growth" of a given output image over time. 
* Added a CLI to go from a list of directories of models to a single directory of renamed and
validated models. `check_move_models.py` will be used on training VM to copy the batch 2 models.
* Created tooling to visualize projection files, including a deterministic way to see when 
projection stops improving to inform the big prod run. No CLI for any of this yet.
* Added CLI to input a vector file or vector files into a model or models and save the resulting 
images.
* Fixed bug where the iterators in `read_projection_file` were not in the correct order, needed
to parse dataset name and then sort by the key int in the filename.
* More usability features for working with projection files.
* Added more functions in `vector_sources` to service researching how to best reduce number of
frames/projection steps needed in the final runs.
* Started to add testing that is GPU non-optional.

0.12.0 - (2021-09-12)
------------------

* Adds lots of new functionality around creating projections (going from an arbitrary image to
 vectors that could be input to the model to re-create the arbitrary image) programmatically.
* Introduces concept of a "projection file", a HDF5 file that contains the massive amount of 
data related to a projection.
* Adds video -> projection file. `stylegan2` natively supports projecting single images but with
this addition we can project videos.
* There are a number of crufty assets/visualization functions hanging around. These'll get 
removed before publication but were helpful in understanding this new technology.


0.11.0 - (2021-08-22)
------------------

* Small release to get new functionality out before working on new big feature.
* Adds a few more things to the vector primitives file to better explore the latent space of models.
* Improves visualization functions, adding ability to quickly input a vector array into a model etc.


0.10.0 - (2021-08-03)
------------------

* Massive refactoring of the matplotlib data visualization pipeline in order to better label the
data in each of the visualizations, and to add visualizations for the dynamic model index.
* Added audio compression that uses gzip as the basis for reduction.
* Created a CLI, `music_into_models.py` to select `.wav` files and directories of models for 
visualization.
* Added standalone visualization for audio reducers to aid in development.
* Switched local version of `stylegan2` to a fork where resuming from a crashed job is possible.


0.9.0 - (2021-07-16)
------------------

* Added functionality to switch between models during a run, allowing a mapping between audio
complexity and the resulting "training resolution" of the output image.
* Added ability to compute the rolling RMS power of an audio file, this is fed into the function
that switches models during a run.


0.8.0 - (2021-07-07)
------------------

* There was a massive bug in `smooth_across_vectors`, and that function didn't do at all what I 
thought it did.
* Added flags to disable 2d/3d visualizations alongside model outputs in `vectors_into_model.py`.
* Re-worked Spectrogram code to remove the need for the scaling step.
* To the main visualization function, `viz_model_ins_outs`, added the ability to, in 2d
see both sides of the combination (music, noise, combined), had to modify types here which will
be extended later.


0.7.0 - (2021-06-24)
------------------

* Updated `docker-compose.yml` and the various `*.Dockerfile`s to create the production dataset of
images for the yellow album cover. There isn't a lot of abstraction on those config files, and it's
pretty specific to this project but everything is there.


0.6.0 - (2021-06-22)
------------------

* Created `select_images_for_training.py` to use face recognition to select a sets of images for
training.

0.5.0 - (2021-06-17)
------------------

* Improved `begin_dataset_upload.py`, added ability to detect if a given dataset was "partially
uploaded" meaning that the copy operation from the dataset directory to the ownCloud directory
wasn't able to finish.


0.4.0 - (2021-05-25)
------------------

* Modified `read_wav_scale_for_video` to work on an entire `np.ndarray` of vectors rather
than the `Sampler`-style re-sample from before.
* Added this function to `_visualize_audio_file_and_model_output` so it can stretch wav files to
make sure there are enough vectors to have high frame rate video.
* Generally encapsulated this workflow so it's more discrete and easier to apply over multiple
audio files at once.
* Added functionality to add audio files to output videos directly. 
* Updated linters to pick up the loose `.py` files in this repo's top directory.


0.3.0 - (2021-05-20)
------------------

* Created dockerfiles, `docker-compose` configs to run
dataset creation, and model training as `docker-compose run` commands. 


0.2.0 - (2021-05-15)
------------------

* Adds functionality to go from vector array -> spectrogram as an input to the model.


0.1.0 - (2021-05-13)
------------------

* Massive merge, I've been lazy about splitting up features.
* Work on using docker to develop model visualizations.
* Added basic music visualizations.


0.0.1 - (2020-10-21)
------------------

* Project begins
