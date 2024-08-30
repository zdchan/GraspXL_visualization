# GraspXL: Generating Grasping Motions for Diverse Objects at Scale

<p align="center">
    <img src="./docs/tease_more.jpg" alt="Image" width="100%"/>
</p>

This is a repository for the visualization of GraspXL Dataset. The repository is based on [arctic-digit](https://github.com/zc-alexfan/arctic-digit), which is used for the visualization of [ARCTIC](https://arctic.is.tue.mpg.de/) dataset.

Our dataset contains diverse grasping motions of 500k+ objects with different dexterous hands:

<p align="center">
    <img src="./docs/large.gif" alt="Image" width="80%"/>
</p>



<p align="center">
    <img src="./docs/robot_hand.gif" alt="Image" width="80%"/>
</p>

### Getting started

Clone the GraspXL_visualization repository:

```
$ git clone https://github.com/zdchan/GraspXL_visualization.git
$ cd GraspXL_visualization
```

Install the dependencies listed in [environment.yaml](./environment.yaml)

```
$ conda env create -f environment.yaml
$ conda activate graspxl_viewer
```

Download MANO pickle data-structures

- Visit [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format `mano_v*_*.zip`). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the contents in `mano_v*_*/models/` to the `data/body_models/mano` folder
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
You can now run the grasping visualization scripts for MANO, Allegro, or Shadow Hand in the [./scripts](./scripts) folder. For example, if you want to visualize a MANO grasping sequence, run
```
$ python ./scripts/visualizer_mano.py
```
We use a wine glass as an example. If you want to visualize another object or another sequence, put the object mesh (.obj file) in [./data/GraspXL/object_mesh/](./data/GraspXL/object_mesh/) and the sequence in [./data/GraspXL/recorded/](./data/GraspXL/recorded/), and run
```
$ python ./scripts/visualizer_mano.py --seq_name <sequence name> --obj_name <object name>
```


### Citation

```bibtex
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
