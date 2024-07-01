# Open X-Embodiment

![](./imgs/teaser.png)

Open X-Embodiment aims to provide all open-sourced robotic data in the same unified format, for easy downstream consumption.

The first publication using the Open X-Embodiment dataset is [`Open X-Embodiment: Robotic Learning Datasets and RT-X Models`](https://robotics-transformer-x.github.io/)

## Dataset Access

### Dataset structure

Each data set is represented as a sequence of episodes, where each episode is represented using the [RLDS episode format](https://github.com/google-research/rlds#dataset-format).

### Dataset colab

We provide a [self-contained colab](https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb) that demonstrates how to visualize a few episodes from each data set, and how to create batches of data ready for training and inference.

### List of datasets

We provide the list of dataset that is included as part of the open-sourcing effort and their metadata in [the dataset spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0).

## RT-1-X Model checkpoint

### Explanation of observation space

The model takes as input a RGB image from the robot workspace camera and a task string describing the task that the robot is supposed to perform.

What task the model should perform is communicated to the model purely through the task string. The image communicates to the model the current state of the world, i.e. assuming the model runs at three hertz, every 333 milliseconds, we feed the latest RGB image from a robot workspace camera into the model to obtain the next action to take.

Please note that the model currently does not take in additional camera images such as wrist camera images, in hand camera images, or depth.

### Explanation of action space

The action dimensions we consider include seven variables for the gripper movement (x, y, z, roll, pitch, yaw, opening of the gripper). Each variable represents the absolute value, the delta change to the dimension value or the velocity of the dimension.

[The inference colab](https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb) of trained RT-1-X Tensorflow checkpoint demonstrates how to load the model checkpoint, run inference on offline episodes and overlay the predicted and ground truth action.

### RT-1-X jax checkpoint

A jax checkpoint that can be used by the flax checkpoint loader in the [rt1_inference_example.py](https://github.com/google-deepmind/open_x_embodiment/blob/main/models/rt1_inference_example.py) can be downloaded by

```gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_jax .```

## FAQ and Common Issues

### Dataset not found

If you run into this issue when trying to run `tfds.load({dataset_name})`

```tensorflow_datasets.core.registered.DatasetNotFoundError: Dataset {dataset_name} not found.```

Try downloading the dataset manually by running

```gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/{dataset_name} ~/tensorflow_datasets/```

Once you download the dataset like this, you can use the dataset with the regular `tfds.load({dataset_name})` command!

## Citation

If you're using the Open X-Embodiment dataset and RT-X in your research, [please cite](https://robotics-transformer-x.github.io/citation.txt). If you're specifically using datasets that have been contributed to the joint effort, please cite those as well. The [dataset spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0) contains the citation for each dataset for your convenience.

## License and Disclaimer

This is not an official Google product.

Copyright 2023 DeepMind Technologies Limited.

- All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

- All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

- Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

- testing to see if i could submit a code change
