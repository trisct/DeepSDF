# DeepSDF

[TOC]

## Installation of Preprocessing Tool

Install Pangolin, nanoflann, eigen and CLI11. Do
```
mkdir build
cd build
cmake -DCMAKE_CXX_STANDARD=17 ..
make -j
```

#### Known Issues

##### 1. `mpark` not found

Full error message is

```
In file included from /usr/local/include/pangolin/geometry/geometry.h:35,
                 from /home/trisst/3dlab/DeepSDF/src/SampleVisibleSurfaceNormals.cpp:11:
/usr/local/include/pangolin/compat/variant.h:10:13: fatal error: mpark/variant.hpp: No such file or directory
   10 | #   include <mpark/variant.hpp>
      |             ^~~~~~~~~~~~~~~~~~~
compilation terminated.
make[2]: *** [CMakeFiles/SampleVisibleSurfaceNormals.dir/build.make:63: CMakeFiles/SampleVisibleSurfaceNormals.dir/src/SampleVisibleSurfaceNormals.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:104: CMakeFiles/SampleVisibleSurfaceNormals.dir/all] Error 2
make: *** [Makefile:130: all] Error 2
```

This is because you have not passed `-DCMAKE_CXX_STANDARD=17` to cmake. You can do this as above or specify in `CmakeLists.txt`.



## Prepare Data (Original)

You need the following files and folders to prepare the training data and start training.

### Raw dataset (unprocessed) folder

The raw data folder should be organized like this.

```
datasets_raw/
	dataset_A/
		class_A_1/
			instance_A_1_1/
				instance_A_1_1.obj
			instance_A_1_2/
				instance_A_1_2.obj
			...
		class_A_2/
			instance_A_2_1/
				instance_A_2_1.obj
			...
		...
	dataset_B/
		class_B_1/
			instance_B_1/
				instance_B_1.obj
		class_B_2/
			instance_B_2/
				instance_B_2.obj
		...
```

### Dataset specific `json` file

For example, for `dataset_A` above you would have a `json` file called `dataset_A.json` like this.

```json
{
	"dataset_A": {
		"class_A_1": [
			"instance_A_1_1",
			"instance_A_1_2",
			...
		],
		"class_A_2": [
			"instance_A_2_1",
			"instance_A_2_2",
			...
		],
        ...
	}
}
```

If you don't have a `json` file already, you can generate one with

```
python generate_json_for_deepsdf.py -d dataset_A
```

The dataset source folder is defaulted to `datasets_raw`, but you can specify another with `-s/--split`.

### Processed dataset folder

All processed datasets are put here. It should be an empty folder at first. Let's say it is called `datasets_processed`. After preprocessing (below), it will look like

```
datasets_processed/
	SdfSamples/
		dataset_A/
			class_A_1/
				instance_A_1_1.npz
				instance_A_1_2.npz
				...
			class_A_2/
				...
			...
		...
```

### Preprocessing of data

If you have prepared the above, do
<span style="color:red"> PLEASE convert to `ply` format for preprocessing!</span>
```
python preprocess_data_concurrent.py -d datasets_processed -s datasets_raw --split dataset_A.json
```

#### Known Issues

##### 1. `what(): Not implemented.`

Full error message is

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Not implemented.
```

This is because the mesh file is of unsupported format for `pangolin:LoadGeometry`. Currently, ascii format ply files are unsupported but binary ply files are supported.

### Dataset Inspection

## Prepare Data (with Normal Samples)

Currently you can run an example with

```
/home/trisst/3dlab/DeepSDF/bin/SampleSurfaceAndNormals -m datasets_raw/simple_shapes/sphere/sphere_uv/sphere_uv_aalnormed.obj -o what --fn_points datasets_processed/SurfaceNormalSamples/sphere_points.obj --fn_normals datasets_processed/SurfaceNormalSamples/sphere_normals.obj
```



## Training

For training you need to set up an experiment folder.

### Experiment folder

The experiment folder only needs a `specs.json` for the training to start. See any `spces.json` in `examples` to see what are needed. The only keys that are related to datasets are:

- `DataSource`: The folder of all processed datasets. In the example above, it is `datasets_processed`
- `TrainSplit` and `TestSplit`: The `json` files containing the names of the training and testing instances. 

Training only requires setting the experiment folder. Suppose the folder is called `deepsdf_exp`. Do

```shell
python train_deep_sdf.py -e deepsdf_exp
# alternatively, you can do
# python train_grad_superv.py -e deefpsdf_exp
# to train with gradiet supervision
```

The training process is recorded in the experiment folders. During training or after doing reconstructions (see below) several folders will be created under the experiment folder.

#### LatentCodes

This contains the latent codes of the training shapes at each checkpoint in `pth` format.

#### ModelParameters, OptimizerParameters

This contains the model parameters and optimizer parameters at each checkpoint in `pth` format.

#### Reconstructions

This folder is created if `reconstruct.py` is used (see <span style="color:red">where ?</span>). It is structured as follows.

```
Reconstructions/
	10000/ # step number of the checkpoint
		Codes/
			dataset_A/
				class_A_1/
					instance_A_1_1.pth # Latent code of that instance
					...
				...
			...
		Meshes/
			dataset_A/
				class_A_1/
					instance_A_1_1.ply # Reconstructed mesh of that instance
					...
				...
			...
```

## Reconstructing Meshes

To use a trained model to reconstruct explicit mesh representations of shapes from the test set, run:

```
python reconstruct.py -e <experiment_directory>
```

This will use the latest model parameters to reconstruct all the meshes in the split. To specify a particular checkpoint to use for reconstruction, use the ```--checkpoint``` flag followed by the epoch number. Generally, test SDF sampling strategy and regularization could affect the quality of the test reconstructions. For example, sampling aggressively near the surface could provide accurate surface details but might leave under-sampled space unconstrained, and using high L2 regularization coefficient could result in perceptually better but quantitatively worse test reconstructions.

## Evaluating Reconstructions

Before evaluating a DeepSDF model, a second mesh preprocessing step is required to produce a set of points sampled from the surface of the test meshes. This can be done as with the sdf samples, but passing the `--surface` flag to the pre-processing script. Once this is done, evaluations are done using:

```
python evaluate.py -e <experiment_directory> -d <data_directory> --split <split_filename>
```

## Other functionalities

### Visualizing Progress

All intermediate results from training are stored in the experiment directory. To visualize the progress of a model during training, run:

```
python plot_log.py -e <experiment_directory>
```

By default, this will plot the loss but other values can be shown using the `--type` flag.

### Continuing from a Saved Optimization State

If training is interrupted, pass the `--continue` flag along with a epoch index to `train_deep_sdf.py` to continue from the saved state at that epoch. Note that the saved state needs to be present --- to check which checkpoints are available for a given experiment, check the `ModelParameters`, `OptimizerParameters`, and `LatentCodes` directories (all three are needed).


### Shape Completion

The current release does not include code for shape completion. Please check back later!
