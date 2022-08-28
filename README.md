# WaveY-Net

For detailed information about WaveY-Net, please see the ACS Photonics article and the Supporting Information:
https://pubs.acs.org/doi/abs/10.1021/acsphotonics.2c00876

## How to Run

**To reliably run the WaveY-Net training script, we first build a Docker container using the official WaveY-Net Docker image uploaded to [Docker Hub](https://hub.docker.com/r/rclupoiu/surrogate).**

To download the Docker image locally, enter the following command in your terminal after installing the [Docker Engine](https://docs.docker.com/engine/install/).
```
docker pull rclupoiu/surrogate:latest_torch
```

*What is a Docker image?* It is a file that contains instructions to locally build a Docker container.

*What is a Docker container?* According to https://www.docker.com/resources/what-container/:
"A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings."

*In short, the Docker container locally builds an environment to allow you to reliably run the WaveY-Net code.*

To build the Docker container, link it to the directory where you pulled this GitHub repo, enable GPU acceleration in your Docker container, and open an interactive bash shell:
```
docker run -v /your/local/path/to/waveynet:/workspace --gpus all -ti --name waveynet_test_container --rm rclupoiu/surrogate:latest_torch bash
```

### Description of WaveY-Net Python Scripts

In the workspace directory of the Docker container interactive bash shell, you will find the three WaveY-Net Python scripts:

**waveynet_trainer.py**: This "master script" calls *simulation_dataset.py* and *learner.py*, initializes the data and the PyTorch training process, executes the training process, executes evaluation and testing, and outputs the trained neural network weights along with the training history statistics.

**simulation_dataset.py**: This script loads the numpy dataset into memory and creates the custom PyTorch DataLoader class.

**learner.py**: This script defines the neural network components used in the WaveY-Net's UNet architecture, and subsequently specifies how data passes through the model.

### Description of the Included Dataset

Please see the official [WaveY-Net webpage on Metanet](http://metanet.stanford.edu/search/waveynet-study/) for a thorough description of the included dataset, as well as snippets of code showing how to import and use the data.

By default, *simulation_dataset.py* downloads the training and testing data from Metanet each time the script is run. If you would like a local copy of the data to bypass the downloading step and speed up experimentation, set the `--local_data` script argument to `True` after [manually downloading the data from Metanet](http://metanet.stanford.edu/search/waveynet-study/) and placing the data files in the same directory as the *simulation_dataset.py* script.

### Training WaveY-Net

From the interactive Docker container bash shell, use the following command to run the training script. Pass any arguments needed to customize the script for your needs.
```
python3 Unet_train.py --model_name waveynet_test
```

To run the training script directly inside a Docker container, without first opening an interactive bash shell, use the following command.
```
docker run -v /your/local/path/to/waveynet:/workspace --gpus all -ti --name waveynet_test_container --rm rclupoiu/surrogate:latest_torch python3 Unet_train.py --model_name waveynet_test
```

### Training Results

The training script outputs the trained weights of the neural network, along with a CSV file containing the training statistics. These are outputted in a directory of the same name as the `--model_name` script argument, which is located in the directory specified by the `--model_save_path` argument.

## Citation
If you use this code or data for your research, please cite:
[High Speed Simulation and Freeform Optimization of Nanophotonic Devices with Physics-Augmented Deep Learning<br>](https://pubs.acs.org/doi/abs/10.1021/acsphotonics.2c00876)
Mingkun Chen, Robert Lupoiu, Chenkai Mao, Der-Han Huang, Jiaqi Jiang, Philippe Lalanne, and Jonathan A. Fan*

*jonfan@stanford.edu
