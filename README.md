# GANce - gance 

![Sample network output](./gance_sample.png)

Maps music and video into the latent space of StyleGAN (mostly focused on StyleGAN2) networks.

See [esologic.com/gance](https://www.esologic.com/gance) for more information.

## Usage

The functionality available in the libraries are exposed through a series of command line 
interfaces. The following is a listing of the commands that are stable and mostly usable at this
point.

| **Script**                 | **Description**                                                                                                                                                                                                                                          |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `check_move_networks.py`   | Scan directories of networks for pickled networks, for each of these networks, load them, and feed in a vector to make sure they are still functional (no bit rot).                                                                                      |
| `music_into_networks.py`   | Feed inputs (music, videos) into a network and record the output. Also tools to visualize these vectors against the network outputs.                                                                                                                     |
| `process_images.py`        | Tools to prepare images for styleGAN training.                                                                                                                                                                                                           |
| `project_video_to_file.py` | Tools to [project](https://github.com/NVlabs/stylegan2#projecting-images-to-latent-space) video into the latent space of a given network, and also visualize the results. Resulting projection files are consumed primarily in `music_into_networks.py`. |
| `snythesize_images.py`     | Given some styleGAN networks, load each network and synthesize a number of images with them. Interesting vectors can be reused with other networks.                                                                                                      |

## Getting Started

### Creating Development/Usage Environment

The specific hardware environment used to train networks and create videos has changed a few times
during the project. There's a detailed description of one of these instances in [this post](https://www.esologic.com/introducing-gance/),
but the important bit is that the hardware is centered around the [NVIDIA Tesla K80](https://www.nvidia.com/en-gb/data-center/tesla-k80/) GPU.
The [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) runtime is used to expose these cards to the python code.

In this repo, there's an ansible playbook that should be able to automatically configure your linux
environment with the correct drivers and software to run the NVIDIA docker runtime.

This playbook can be found in `provision_hardware`. You'll need to modify the `hosts.yml` file
with the ip of your development environment. With this completed, and with the projects virtual 
environment activated (meaning ansible is installed) run:

```bash
ansible-playbook install_nvidia_docker.yml -i hosts.yml
```

Now, you should be able to create docker containers that can use the GPU. To standup a development
container, run:

```bash
sudo docker-compose up -d --build
```

The bind paths referencing `/home/devon/development_share/*` obviously aren't going exist on your
system, and can be removed.

### Python Dependencies

See the `requirements` directory for required Python modules for building, testing, developing etc.
They can all be installed in a [virtual environment](https://docs.python.org/3/library/venv.html) 
using the follow commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r ./requirements/dev.txt -r ./requirements/prod.txt -r ./requirements/test.txt
```

There's also a bin script to do this:

```
./tools/create_venv.sh
```


## Developer Guide

The following is documentation for developers that would like to contribute
to GANce.

### Type System Limitations

A common idiom found in this application is the following:

```python
# Shape (1, Any, Any)
CompleteLatentsType = NewType("CompleteLatentsType", "np.ndarray[np.float32]")  # type: ignore
```

Where a `NewType` is created based on an `np.ndarray`. This doesn't really do anything outside
of making the programs a bit more simple to understand to the reader. The following would pass
`mypy`:

```python
CompleteLatentsType("oh no!")
```

This is because:
* We're stuck on `numpy==1.16.4` because of the `tensorflow` dependency of `stylegan2`.
* `np.ndarray` types are `Any` as far as `mypy` is concerned in this version.
* When you `NewType("MyType", Any)`, it stops `mypy` from being able to identify type mismatches.

Read more here: https://github.com/python/mypy/issues/6701

We could resolve this if we take the time to verify that it's okay to upgrade the `numpy` version.

### Pycharm Note

Make sure you mark `gance` and `./test` as source roots!

### Testing

This project uses pytest to manage and run unit tests. Unit tests located in the `test` directory 
are automatically run during the CI build. You can run them manually with:

```
./tools/run_tests.sh
```

### Local Linting

There are a few linters/code checks included with this project to speed up the development process:

* Black - An automatic code formatter, never think about python style again.
* Isort - Automatically organizes imports in your modules.
* Pylint - Check your code against many of the python style guide rules.
* Mypy - Check your code to make sure it is properly typed.

You can run these tools automatically in check mode, meaning you will get an error if any of them
would not pass with:

```
./tools/run_checks.sh
```

Or actually automatically apply the fixes with:

```
./tools/apply_linters.sh
```

There are also scripts in `./tools/` that include run/check for each individual tool.


### Using pre-commit

First you need to init the repo as a git repo with:

```
git init
```

Then you can set up the git hook scripts with:

```
pre-commit install
```

By default:

* black
* pylint
* isort
* mypy

Are all run in apply-mode and must pass in order to actually make the commit.

Also by default, pytest needs to pass before you can push.

If you'd like skip these checks you can commit with:

```
git commit --no-verify
```

If you'd like to quickly run these pre-commit checks on all files (not just the staged ones) you
can run:

```
pre-commit run --all-files
```

