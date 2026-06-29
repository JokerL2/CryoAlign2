# CryoAlign2

CryoAlign2 is an alignment-based tool for global and local Cryo-EM density-map retrieval. It converts density maps into sampled point clouds, extracts structural keypoints, performs 3D alignment, and evaluates aligned maps using configurable similarity scores.

Sampling is built into CryoAlign2. CPU convolution and local alignment can be accelerated with MPI, while CUDA sampling is available as an optional runtime mode.

## Installation

### Prerequisites

- Ubuntu 20.04 or later
- CMake 3.20 or later
- A C++17 compiler
- OpenMPI
- OpenMP
- CUDA 12.2
- LibTorch 2.2.x with CUDA support
- Open3D 0.18.0
- PCL
- FLANN
- Eigen3
- Boost
- Armadillo
- mlpack 3.2.2
- CNPY
- TEASER++

### Clone the repository

```bash
git clone https://github.com/JokerL2/CryoAlign2.git
cd CryoAlign2
```

### Build from source

```bash
cmake -S . -B build \
  -DLIBTORCH_PATH=/path/to/libtorch

cmake --build build -j2
```

`LIBTORCH_PATH` can also be supplied as an environment variable:

```bash
export LIBTORCH_PATH=/path/to/libtorch
cmake -S . -B build
cmake --build build -j2
```

The executables are generated in `bin/`:

```text
bin/CryoAlign
bin/CryoAlign_extract_keypoints
bin/CryoAlign_alignment
```

### Build with Docker

```bash
docker build -f docker/dockerfile -t cryoalign2 .
docker run -it --name cryoalign2 --gpus all cryoalign2
```

## Runtime Modes

### CPU

CPU is the default execution mode:

```bash
./bin/CryoAlign_extract_keypoints \
  --data_dir /path/to/dataset \
  --map_name EMD-3695.map \
  --contour_level 0.008 \
  --voxel_size 5.0
```

Use `--cpu` or `--no_gpu` to explicitly force CPU execution.

### CPU with MPI

MPI accelerates CPU convolution during sampling and distributes local alignment tasks. Limit each MPI rank to one BLAS/OpenMP thread to avoid CPU oversubscription:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign_extract_keypoints \
  --data_dir ../../example_dataset/emd_3661_emd_6647 \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0 \
  --cpu
```

### GPU

GPU sampling is enabled with `--use_gpu` or `--gpu`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./bin/CryoAlign_extract_keypoints \
  --data_dir ../../example_dataset/emd_3661_emd_6647 \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0 \
  --use_gpu
```

Use one process per GPU. Do not launch multiple MPI ranks on the same GPU. If GPU mode is requested with multiple MPI ranks, CryoAlign2 falls back to the CPU/MPI path.

The CPU/MPI path is recommended when exact reproducibility is required. GPU floating-point calculations may produce small mean-shift coordinate differences, which can affect DBSCAN boundary points and the selected alignment.

## Scoring Modes

Both `global` and `mask` alignment support two scoring modes.

### Single score

`single` is the default mode and returns the normal-consistency score:

```text
score = number of correspondences with normal cosine similarity >= 0.6
        / total number of correspondences
```

### Multidimensional score

`multi` combines four metrics:

- Normal consistency
- Point-distance similarity
- Local geometric-density similarity
- SHOT feature similarity

The combined score is:

```text
score =
    normal_weight   * normal_score   +
    distance_weight * distance_score +
    density_weight  * density_score  +
    shot_weight     * shot_score
```

Every weight must be within `[0, 1]`, and all four weights must sum to `1.0`. The default weight is `0.25` for each metric.

## Executables

### CryoAlign

`CryoAlign` runs the complete workflow:

1. Density-map sampling
2. Keypoint extraction
3. Global or local alignment
4. Similarity scoring

Usage:

```text
CryoAlign
  --data_dir DIR
  --source_map MAP
  --source_contour_level LEVEL
  --target_map MAP
  --target_contour_level LEVEL
  [--source_pdb PDB]
  [--source_sup_pdb PDB]
  [--voxel_size 5.0]
  [--feature_radius 7.0]
  [--use_gpu|--cpu]
  --alg_type global|mask
  [--score_mode single|multi]
  [--normal_weight 0.25]
  [--distance_weight 0.25]
  [--density_weight 0.25]
  [--shot_weight 0.25]
```

Important options:

- `--data_dir`: Directory containing the input maps and optional structure files.
- `--source_map`: Source density-map filename.
- `--source_contour_level`: Contour level for the source map.
- `--target_map`: Target density-map filename.
- `--target_contour_level`: Contour level for the target map.
- `--source_pdb`: Optional source structure.
- `--source_sup_pdb`: Optional transformed source structure used as ground truth.
- `--voxel_size`: Sampling interval in angstroms. Default: `5.0`.
- `--feature_radius`: Radius used to construct local features. Default: `7.0`.
- `--alg_type`: `global` for global alignment or `mask` for local alignment.
- `--use_gpu`: Enable GPU sampling.
- `--cpu`: Force CPU sampling.
- `--score_mode`: `single` or `multi`. Default: `single`.

CPU and MPI complete-workflow example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign \
  --data_dir ../../example_dataset/emd_3661_emd_6647 \
  --source_map emd_3661.map \
  --source_contour_level 0.07 \
  --target_map emd_6647.map \
  --target_contour_level 0.017 \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global \
  --score_mode single \
  --cpu
```

GPU complete-workflow example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./bin/CryoAlign \
  --data_dir ../../example_dataset/emd_3661_emd_6647 \
  --source_map emd_3661.map \
  --source_contour_level 0.07 \
  --target_map emd_6647.map \
  --target_contour_level 0.017 \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global \
  --score_mode single \
  --use_gpu
```

### CryoAlign_extract_keypoints

`CryoAlign_extract_keypoints` performs sampling and keypoint extraction only.

Usage:

```text
CryoAlign_extract_keypoints
  --data_dir DIR
  --map_name MAP
  --contour_level LEVEL
  [--voxel_size 5.0]
  [--use_gpu|--cpu]
```

CPU and MPI example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign_extract_keypoints \
  --data_dir ../../example_dataset/emd_3661_emd_6647 \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0 \
  --cpu
```

GPU example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./bin/CryoAlign_extract_keypoints \
  --data_dir ../../example_dataset/emd_3661_emd_6647 \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0 \
  --use_gpu
```

The command writes sampled points and keypoints under `--data_dir`:

```text
<map_name>_<voxel_size>.txt
Points_<map_id>_<voxel_size>_Key.xyz
```

### CryoAlign_alignment

`CryoAlign_alignment` aligns previously generated sample and keypoint files. GPU options are not used by this executable.

Usage:

```text
CryoAlign_alignment
  --data_dir DIR
  --source_xyz XYZ
  --target_xyz XYZ
  --source_sample TXT
  --target_sample TXT
  [--source_pdb PDB]
  [--source_sup_pdb PDB]
  [--voxel_size 5.0]
  [--feature_radius 7.0]
  --alg_type global|mask
  [--score_mode single|multi]
  [--normal_weight 0.25]
  [--distance_weight 0.25]
  [--density_weight 0.25]
  [--shot_weight 0.25]
```

File arguments are resolved relative to `--data_dir`. Use filenames rather than absolute paths.

Global alignment with the single score:

```bash
./bin/CryoAlign_alignment \
  --data_dir ../../example_dataset/emd_3695_emd_3696 \
  --source_xyz Points_3695_5.00_Key.xyz \
  --target_xyz Points_3696_5.00_Key.xyz \
  --source_sample EMD-3695_5.00.txt \
  --target_sample EMD-3696_5.00.txt \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global \
  --score_mode single
```

Global alignment with a weighted multidimensional score:

```bash
./bin/CryoAlign_alignment \
  --data_dir ../../example_dataset/emd_3695_emd_3696 \
  --source_xyz Points_3695_5.00_Key.xyz \
  --target_xyz Points_3696_5.00_Key.xyz \
  --source_sample EMD-3695_5.00.txt \
  --target_sample EMD-3696_5.00.txt \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global \
  --score_mode multi \
  --normal_weight 0.4 \
  --distance_weight 0.2 \
  --density_weight 0.2 \
  --shot_weight 0.2
```

Mask alignment with MPI:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign_alignment \
  --data_dir ../../example_dataset/emd_3661_emd_6647 \
  --source_xyz Points_3661_Key.xyz \
  --target_xyz Points_6647_5.00_Key.xyz \
  --source_sample emd_3661_5.00.txt \
  --target_sample emd_6647_5.00.txt \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type mask \
  --score_mode single
```

The same scoring options apply to both `--alg_type global` and `--alg_type mask`.

## Retrieval Workflow

### Build a retrieval database

Use `CryoAlign_extract_keypoints` to generate sampled point clouds and keypoint files:

```bash
python script/CreateDB.py
```

Generated files can be organized like the contents of `database example/`.

### Perform retrieval

```bash
python script/CryoSearch.py
```

The retrieval script writes density-map similarity scores to `res.txt`.

### Overlay density maps

```bash
python script/Transform_map.py
```

This script applies the saved rotation and translation matrix to a density map.

## Help

```bash
./bin/CryoAlign --help
./bin/CryoAlign_extract_keypoints --help
./bin/CryoAlign_alignment --help
```
