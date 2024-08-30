# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

<p align="center">
    <img src="./docs/tease_more.jpg" alt="Image" width="100%"/>
</p>

This is a repository for the visualization of GraspXL Dataset. The repository is based on [arctic-digit](https://github.com/zc-alexfan/arctic-digit), which is used for the visualization of [ARCTIC](https://arctic.is.tue.mpg.de/) dataset.

Our dataset contains diverse grasping motions of 500k+ objects from [Objaverse](https://objaverse.allenai.org/) with different dexterous hands:

<p align="center">
    <img src="./docs/large.gif" alt="Image" width="80%"/>
</p>



<p align="center">
    <img src="./docs/robot_hand.gif" alt="Image" width="80%"/>
</p>


### Why use GraspXL?

Summary on the dataset:
- It contains 10M+ diverse grasping motions for 500k+ objects of different dexterous hands.
- All the grasping motions are generated with a physics simulation, which makes sure the physical plausibility of the generated motions.
- Each motion contains accurate object and hand poses of each frame.

Potential tasks with GraspXL:
- Generating [full-body motion](https://goal.is.tue.mpg.de/) for general objects
- Achieving zero-shot text-to-motion generation with off-the-shelf [text-to-mesh](https://dreamfusion3d.github.io/) generation methods
- Generating large scale of pseudo-real 3D RGBD grasping motions with [texture generation](https://mq-zhang1.github.io/HOIDiffusion/) methods
- [Simulating general human hand motions](https://eth-ait.github.io/synthetic-handovers/) for human-robot interaction
- Serving as expert demonstrations for robot imitation learning


Check out our [project page](https://eth-ait.github.io/graspxl/) for more details.



### Getting started

Clone the GraspXL_visualization repository:

```
$ git clone https://github.com/zdchan/GraspXL_visualization.git
$ cd GraspXL_visualization
```

Install the dependencies listed in [environment.yaml](./environment.yaml)

```
$ conda env create -f environment.yaml
```

Download MANO pickle data-structures

- Visit [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format `mano_v*_*.zip`). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the contents to the `data/body_models/mano` folder
- Your `data/body_models/mano` folder structure should look like this:

```
data/body_models/mano
   ├── info.txt
   ├── LICENSE.txt
   ├── MANO_LEFT.pkl
   ├── MANO_RIGHT.pkl
   ├── SMPLH_female.pkl
   └── SMPLH_male.pkl
```





### Citation

```
@inProceedings{zhang2024graspxl,
  title={{GraspXL}: Generating Grasping Motions for Diverse Objects at Scale},
  author={Zhang, Hui and Christen, Sammy and Fan, Zicong and Hilliges, Otmar and Song, Jie},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

Our paper benefits a lot from [aitviewer](https://github.com/eth-ait/aitviewer). If you find our viewer useful, to appreciate their hard work, consider citing:

```bibtex
@software{kaufmann_vechev_aitviewer_2022,
  author = {Kaufmann, Manuel and Vechev, Velko and Mylonopoulos, Dario},
  doi = {10.5281/zenodo.1234},
  month = {7},
  title = {{aitviewer}},
  url = {https://github.com/eth-ait/aitviewer},
  year = {2022}
}
```
