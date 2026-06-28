# CryoAlign2

CryoAlign2 is an alignment-based tool for global and local Cryo-EM density-map
retrieval. It converts density maps into sampled point clouds, extracts
structural keypoints, performs high-precision 3D alignment, and evaluates
superimposed maps with a multidimensional similarity score.

The sampling step is built into CryoAlign2. CPU sampling and local alignment can
be accelerated with MPI, while CUDA sampling is available as an optional runtime
mode.

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

Set `LIBTORCH_PATH` to the extracted LibTorch directory:

```bash
cmake -S . -B build \
  -DLIBTORCH_PATH=/path/to/libtorch
cmake --build build -j2
```

The executables are generated in `bin/`:

```text
bin/CryoAlign
bin/CryoAlign_extract_keypoints
bin/CryoAlign_alignment
```

`LIBTORCH_PATH` can also be supplied as an environment variable:

```bash
export LIBTORCH_PATH=/path/to/libtorch
cmake -S . -B build
cmake --build build -j2
```

### Build with Docker

The repository includes a Dockerfile for environments with NVIDIA GPU support:

```bash
docker build -f docker/dockerfile -t cryoalign2 .
docker run -it --name cryoalign2 --gpus all cryoalign2
```

## Runtime Modes

### CPU

CPU is the default mode. A single-process command uses the stable serial CPU
path:

```bash
./bin/CryoAlign_extract_keypoints \
  --data_dir /path/to/dataset \
  --map_name EMD-3695.map \
  --contour_level 0.008 \
  --voxel_size 5.0
```

Use `--cpu` or `--no_gpu` to explicitly force CPU execution.

### CPU with MPI

MPI accelerates CPU convolution during sampling and distributes local alignment
work. Limiting each rank to one BLAS/OpenMP thread avoids CPU oversubscription:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign_extract_keypoints \
  --data_dir /path/to/dataset \
  --map_name EMD-3695.map \
  --contour_level 0.008 \
  --voxel_size 5.0
```

### GPU

GPU sampling is enabled explicitly with `--use_gpu` or `--gpu`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./bin/CryoAlign_extract_keypoints \
  --data_dir /path/to/dataset \
  --map_name EMD-3695.map \
  --contour_level 0.008 \
  --voxel_size 5.0 \
  --use_gpu
```

Use one process per GPU. If GPU mode is requested with multiple MPI ranks,
CryoAlign2 falls back to the CPU/MPI path.

The CPU/MPI path is recommended when exact reproducibility is required. GPU
floating-point calculations can introduce small differences in mean-shift
coordinates, which may affect DBSCAN boundary points and the selected alignment.

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
```

Options:

- `--data_dir`: Directory containing the input maps and optional structure files.
- `--source_map`: Source density-map filename.
- `--source_contour_level`: Recommended contour level for the source map.
- `--target_map`: Target density-map filename.
- `--target_contour_level`: Recommended contour level for the target map.
- `--source_pdb`: Optional source structure.
- `--source_sup_pdb`: Optional transformed source structure used as ground truth.
- `--voxel_size`: Sampling interval in angstroms. Default: `5.0`.
- `--feature_radius`: Radius used to construct local features. Default: `7.0`.
- `--alg_type`: `global` for global alignment or `mask` for local alignment.
- `--use_gpu`: Enable optional GPU sampling.
- `--cpu`: Force CPU sampling.

Global alignment example:

```bash
./bin/CryoAlign \
  --data_dir ../../example_dataset/emd_3695_emd_3696 \
  --source_map EMD-3695.map \
  --source_contour_level 0.008 \
  --target_map EMD-3696.map \
  --target_contour_level 0.002 \
  --source_pdb 5nsr.pdb \
  --source_sup_pdb 5nsr_sup.pdb \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global
```

MPI global alignment example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign \
  --data_dir ../../example_dataset/emd_3695_emd_3696 \
  --source_map EMD-3695.map \
  --source_contour_level 0.008 \
  --target_map EMD-3696.map \
  --target_contour_level 0.002 \
  --source_pdb 5nsr.pdb \
  --source_sup_pdb 5nsr_sup.pdb \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global
```

For local alignment, use `--alg_type mask`.

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

Example:

```bash
./bin/CryoAlign_extract_keypoints \
  --data_dir ../../example_dataset/emd_3695_emd_3696 \
  --map_name EMD-3695.map \
  --contour_level 0.008 \
  --voxel_size 5.0
```

The command writes the sampled point cloud and extracted keypoints under
`--data_dir`:

```text
<map_name>_<voxel_size>.txt
Points_<map_id>_<voxel_size>_Key.xyz
```

### CryoAlign_alignment

`CryoAlign_alignment` aligns previously generated sample and keypoint files.

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
```

The file arguments are resolved relative to `--data_dir`; use filenames rather
than absolute paths.

Global alignment example:

```bash
./bin/CryoAlign_alignment \
  --data_dir ../../example_dataset/emd_3695_emd_3696 \
  --source_xyz Points_3695_5.00_Key.xyz \
  --target_xyz Points_3696_5.00_Key.xyz \
  --source_sample EMD-3695_5.00.txt \
  --target_sample EMD-3696_5.00.txt \
  --source_pdb 5nsr.pdb \
  --source_sup_pdb 5nsr_sup.pdb \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global
```

For local alignment, use `--alg_type mask`. With MPI, local alignment tasks are
distributed across ranks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign_alignment \
  --data_dir ../../example_dataset/emd_3695_emd_3696 \
  --source_xyz Points_3695_5.00_Key.xyz \
  --target_xyz Points_3696_5.00_Key.xyz \
  --source_sample EMD-3695_5.00.txt \
  --target_sample EMD-3696_5.00.txt \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type mask
```

## Retrieval Workflow

### 1. Build a retrieval database

Use `CryoAlign_extract_keypoints` to generate sampled point clouds and keypoint
files. The repository includes a database-building script:

```bash
python script/CreateDB.py
```

The generated files can be organized like the contents of `database example/`.

### 2. Perform retrieval

Use the alignment executable through the retrieval script:

```bash
python script/CryoSearch.py
```

The script writes density-map similarity scores to `res.txt`.

### 3. Overlay density maps

Use the transformation script to apply the saved rotation and translation
matrix:

```bash
python script/Transform_map.py
```

## Help

```bash
./bin/CryoAlign --help
./bin/CryoAlign_extract_keypoints --help
./bin/CryoAlign_alignment --help
```

The alignment commands support two scoring modes:

```text
--score_mode single|multi
--normal_weight VALUE
--distance_weight VALUE
--density_weight VALUE
--shot_weight VALUE
```

`single` is the default and returns the normal-consistency score. `multi`
combines normal consistency, point-distance similarity, local geometric-density
similarity, and SHOT feature similarity. Multi-mode weights must be within
`[0, 1]` and sum to `1.0`; their default values are `0.25` each.

```bash
# Single normal-consistency score
./bin/CryoAlign_alignment ... \
  --alg_type global \
  --score_mode single

# Weighted multidimensional score
./bin/CryoAlign_alignment ... \
  --alg_type global \
  --score_mode multi \
  --normal_weight 0.4 \
  --distance_weight 0.2 \
  --density_weight 0.2 \
  --shot_weight 0.2
```

The same scoring options apply to both `--alg_type global` and
`--alg_type mask`.
