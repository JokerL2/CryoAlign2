---
CryoAlign2: efficient global and local Cryo-EM map retrieval based on parallel-accelerated local spatial structural features
---

# Introduction

We developed an alignment-based retrieval tool to perform both global and local retrieval. Our approach adopts parallel-accelerated CryoAlign for high-precision 3D alignment and transforms density maps into point clouds for efficient retrieval and storage. Additionally, a multi-dimension scoring function is introduced to accurately assess structural similarities between superimposed density maps. 

# Installation

## Prerequisites

Note that CryoAlign2 depends on and uses several external programs and
libraries.If you have a Docker environment with GPU, we strongly
recommend you to generate image using the dockerfile.

-   Ubuntu 18.04 or later

<!-- -->

-   CMake 3.20+

<!-- -->

-   Open3D 0.18.0

<!-- -->

-   Libtorch 2.2.0 CUDA(Version above 12.2)

<!-- -->

-   FLANN

<!-- -->

-   PCL(point cloud libraries)

<!-- -->

-   EIGEN3

<!-- -->

-   MLPACK 3.2.2

<!-- -->

-   CNPY

<!-- -->

-   TEASER++

<!-- -->

-   FFTW

## Installtion of CryoAlign2

We store the public release versions of CryoAlign2 on GitHub, a site
that provides code-development with version control and issue tracking
through the use of git. To clone the repository, run the following
command

        git clone https://github.com/JokerL2/CryoAlign2.git

For the convenience of environment configuration, we have provided
Docker images that include the necessary environments and external
libraries. Users must ensure that the external libraries have been
properly installed on their servers.

-   Using Docker

``` {.numberLines numbers="left" xleftmargin="2em"}
docker build -f dockerfile -t [image name] .
docker run -it --name [container name] --gpus all [image name]
```

-   Some external libraries and tools

``` {.numberLines numbers="left" xleftmargin="2em"}
Download libtorch
cd ~/
wget https://download.pytorch.org/libtorch/nightly/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.0.dev20231213%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.2.0.dev20231213+cu121.zip
```

-   Install CryoAlign2

``` {.numberLines numbers="left" xleftmargin="2em"}
cd /CryoAlign2 && mkdir build && cd build
cmake ..
make
```

# Executable file description

After installation, three executable files will be generated.

``` {.numberLines numbers="left" xleftmargin="2em"}
CryoAlign: Full-process density map similarity comparison, including density map sampling, keypoint extraction, and alignment scoring.

CryoAlign_extract_keypoints: Performs density map sampling and keypoint extraction only.

CryoAlign_alignment: Direct alignment scoring using sampled points and keypoints from density maps.

```
## CryoAlign
Usage:

    CryoAlign [data dir] [source.map] [source contour level] [target.map]
    [target contour level] [source.pdb] [source sup.pdb] [voxel_size] [feature_radius] [alg_type]

Options:

-   `–data_dir`: Map file path.

-   `–source_map`: Source emdb num.

-   `–source_contour_level`: Author recommend contour level.

-   `–target_map`: Target emdb num.

-   `–target_contour_level`: Author recommend contour level.

-   `–source_pdb` (optional): Source pdb name.

-   `–source_sup_pdb` (optional): Transformed source pdb name (ground
    truth).

-   `–voxel_size`: Sampling interval (defaults 5.0).

-   `–feature_radius`: Radius for feature construction (defaults 7.0).

-   `–alg_type`: `Global_alignment` or `Mask_alignment`.

Example:

For `Global_alignment`:

    CryoAlign --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_map EMD-3695.map --source_contour_level 0.008 --target_map EMD-3696.map --target_contour_level 0.002 --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type global

For `Mask_alignment`:

    CryoAlign --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_map EMD-3695.map --source_contour_level 0.008 --target_map EMD-3696.map --target_contour_level 0.002 --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type mask

## CryoAlign_extract_keypoints

Usage:

    CryoAlign_extract_keypoints [data dir] [source.map] [source contour level] [target.map] [target contour level] [voxel_size]

Options:

-   `–data_dir`: Map file path.

-   `–map_name`: Source emdb num.

-   `–contour_level`: Author recommend contour level.

-   `–voxel_size`: Sampling interval. (defaults 5.0)

Example:

    CryoAlign_extract_keypoints --data_dir ../../example_dataset/emd_3695_emd_3696/ --map_name EMD-3695.map --contour_level 0.008 --voxel_size 5.0

## CryoAlign_alignment

    CryoAlign_alignment [data dir] [source_xyz] [target_xyz] [source_sample]
    [target_sample] [source.pdb] [source sup.pdb] [voxel_size] [feature_radius]
    [alg_type]

    Options:
      --data_dir: Map file path.
      --source_xyz: Source map keypoints file.
      --target_xyz: Target map keypoints file
      --source_sample: Source map sample file.
      --target_sample: Target map sample file.
      --source_pdb(optional): Source pdb name.
      --source_sup_pdb(optional): Transformed source pdb name (ground truth).
      --voxel_size: Sampling interval. (defaults 5.0)
      --feature_radius: Radius for feature construction. (defaults 7.0)
      --alg_type: Global_alignment or Mask_alignment.

Examples

For Global Alignment

    CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ \
    --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz \
    --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt \
    --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 \
    --feature_radius 7.0 --alg_type global

For Mask Alignment

    CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ \
    --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz \
    --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt \
    --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 \
    --feature_radius 7.0 --alg_type mask
# Retrieval steps
## Step1.Build a retrieval library
``` {.numberLines numbers="left" xleftmargin="2em"}
Use CryoAlign_extract_keypoints to build a retrieval database. For the script, see script/CreateDB.py.
python script/CreateDB.py
You will get a key point cloud and sample point cloud library, similar to the database example
```
## Step2.Perform a retrieval
``` {.numberLines numbers="left" xleftmargin="2em"}
Use CryoAlign_alignment to perform the search. See script/CryoSearch.py ​​for the script.
python script/CryoSearch.py
You will get a density map similarity score file res.txt
8553-8551	0.810762
8587-8551	0.7663
8889-8587	0.456922
8889-8551	0.689744
...
```
## Step3.Density map overlay
``` {.numberLines numbers="left" xleftmargin="2em"}
Use script/Transform_map.py to overlay the density map, and save the RT (rotation translation) matrix for each alignment.
python script/Transform_map.py
```
