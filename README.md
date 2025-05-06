---
title: "CryoAlign: Accurate global and local 3D alignment of cryo-EM
  density maps using local spatial structural features"
---

# Introduction

Advances in cryo-electron imaging technologies have led to a rapidly
increasing number of density maps. Alignment and comparison of density
maps play a crucial role in interpreting structural information, such as
conformational heterogeneity analysis using global alignment and atomic
model assembly through local alignment. Here, we propose a fast and
accurate global and local cryo-electron microscopy density map alignment
method CryoAlign, which leverages local density feature descriptors to
capture spatial structure similarities. CryoAlign is the first
feature-based EM map alignment tool, in which the employment of
feature-based architecture enables the rapid establishment of point pair
correspondences and robust estimation of alignment parameters. Extensive
experimental evaluations demonstrate the superiority of CryoAlign over
the existing methods in both alignment accuracy and speed.

Here, we offer both global and local registration methods for CryoAlign.

# Installation

The sections below explain how to download and install CryoAlign on your
computer.

## Prerequisites

Note that CryoAlign depends on and uses several external programs and
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

## Installtion of CryoAlign

We store the public release versions of CryoAlign on GitHub, a site that
provides code-development with version control and issue tracking
through the use of git. To clone the repository, run the following
command

        git clone https://github.com/JokerL2/CryoAlign_cpp.git

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
Download MMalign
cd /CryoAlign_cpp/bin
wget https://zhanggroup.org/MM-align/bin/module/MMalign.zip
unzip MMalign.zip
```

-   Install CryoSearch

``` {.numberLines numbers="left" xleftmargin="2em"}
cd /CryoSearch && mkdir build && cd build
cmake ..
make
```

## Executable file description

After installation, two executable files will be generated,
corresponding to the two execution steps of CryoAlign.

-   MMalign

::: adjustwidth
2em An algorithm for structurally aligning a pair of multiple-chain
protein complexes. a quick algorithm for aligning multiple-chain protein
complex structures using iterative dynamic programming.MMalign is used
to generate the superposition state PDB file, which serves as the ground
truth for CryoAlign, thereby validating the effectiveness of CryoAlign.
:::

-   CryoAlign

::: adjustwidth
2em Accurate global and local 3D alignment of cryo-EM density maps using
local spatial structural features.CryoAlign is a feature-based cryo-EM
map alignment tool, in which the employment of feature-based
architecture enables the rapid establishment of point pair
correspondences and robust estimation of alignment parameters.CryoAlign
offers two alignment methods: global alignment and local alignment.
:::

# Explanation of parameters and examples

## Parameter explanation of MMalign

Usage: MMalign complex1.pdb complex2.pdb \[Options\]

Example usages: MMalign complex1.pdb complex2.pdb -a T -o complex1.sup

==== Required options ====

-   -a

TM-score normalized by the average length of two structures T or F,
(default F)

-   -o

Output the superposition of complex1.pdb to sup.pdb

## Parameter explanation of Cryoalign

Selects the alignment method:

-   global alignment: direct

    -   for global alignment:

    -   Usage:
        `CryoAlign [data_dir] [source.map] [source_contour_level] [target.map] [target_contour_level] [source.pdb] [source_sup.pdb] [direct]`

        -   options:

            -   `data_dir`: Map file path.

            -   `source.map`: Source emdb num.

            -   `source_contour_level`: Author recommend contour level.

            -   `target.map`: Target emdb num.

            -   `target_contour_level`: Author recommend contour level.

            -   `source.pdb`: Source pdb name.

            -   `source_sup.pdb`: Transformed source pdb name (ground
                truth).

            -   `direct`: Global alignment indicator.

-   local alignment: mask

    -   for local alignment:

    -   Usage:
        `CryoAlign [data_dir] [source.map] [source_contour_level] [target.map] [target_contour_level] [source.pdb] [source_sup.pdb] [mask]`

        -   options:

            -   `data_dir`: Map file path.

            -   `source.map`: Source emdb num.

            -   `source_contour_level`: Author recommend contour level.

            -   `target.map`: Target emdb num.

            -   `target_contour_level`: Author recommend contour level.

            -   `source.pdb`: Source pdb name.

            -   `source_sup.pdb`: Transformed source pdb name (ground
                truth).

            -   `direct`: Local alignment indicator.

## Explanation of output file

-   for global alignment: If the align method you are using is global
    align, the results will be directly displayed in the terminal,
    including the RMSD value and the RT transformation matrix.

<!-- -->

-   for local alignment: If the align method you are using is local
    align, the results will be saved in the directory specified as
    \[data_dir\], in a file named extract_top_10.txt. This file will
    contain the scores of the top 10 RT transformation matrices, as well
    as the RMSD values.

## Examples

When using CryoAlign for cryo-electron microscopy density map alignment,
the necessary parameters provided by the user include the source MRC
file, the target MRC file, the contour_level of source MRC file and
target MRC file, the source pdb file,the source_target_sup pdb file and
the alignment method. (see subsection 3.1). The following are two
examples, corresponding to the global alignment and the local alignment.

-   for global alignment:

``` {.numberLines numbers="left" xleftmargin="2em" breaklines="true"}
Step1
./MMalign ../example_dataset/emd_2677_emd_3240/4upc.cif ../example_dataset/emd_2677_emd_3240/5fn5.pdb -a T -o  ../example_dataset/emd_2677_emd_3240/4upc_5fn5_sup.cif
Step2
./CryoAlign ../example_dataset/emd_2677_emd_3240/ EMD-2677.map 0.12 EMD-3240.map 0.04 4upc.cif 4upc_5fn5_sup.cif direct
```

-   for local alignment:

``` {.numberLines numbers="left" xleftmargin="2em" breaklines="true"}
Step1
./MMalign ../example_dataset/emd_2677_emd_3240/4upc.cif ../example_dataset/emd_2677_emd_3240/5fn5.pdb -a T -o  ../example_dataset/emd_2677_emd_3240/4upc_5fn5_sup.cif
Step2
./CryoAlign ../example_dataset/emd_2677_emd_3240/ EMD-2677.map 0.12 EMD-3240.map 0.04 4upc.cif 4upc_5fn5_sup.cif mask
```
