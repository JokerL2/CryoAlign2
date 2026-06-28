# CryoAlign2 V6

CryoAlign2 用于 Cryo-EM 密度图的关键点提取、全局/局部配准和相似度评分。V6
内置了 Sample，使用 MPI 加速 CPU 抽点卷积和局部 alignment，并提供可选 GPU
抽点路径。

V6 是一个默认走 CPU 稳定路径、GPU 可选启用的版本。

- 默认不加参数时：使用 CPU 路径；单进程时等价于稳定 CPU 版，多进程时可用 MPI 加速抽点卷积。
- 加 `--use_gpu` 时：使用 GPU 版 `conv3d + meanshift`，速度更快，但 keypoints 可能和 CPU 版有很小数值差异。
- 加 `--cpu` 或 `--no_gpu` 时：强制 CPU 路径。

当前建议：

- 最终可复现实验：用 CPU/MPI 路径。
- 快速预筛选或探索：用 `--use_gpu`。

## 获取代码

```bash
git clone https://github.com/JokerL2/CryoAlign2.git
cd CryoAlign2
```

主要可执行文件：

```text
bin/CryoAlign                 # 完整流程：抽点 + alignment
bin/CryoAlign_extract_keypoints # 只抽 keypoints
bin/CryoAlign_alignment         # 已有 keypoints/sample 时只做 alignment
```

## 依赖

- Ubuntu 20.04 或更高版本
- CMake 3.20+
- 支持 C++17 的 GCC/Clang
- OpenMPI
- CUDA 12.2 和 LibTorch 2.2.1 CUDA 版
- OpenMP、Boost、Eigen3、Armadillo、mlpack
- Open3D 0.18、PCL、FLANN、CNPY、TEASER++

## 编译

```bash
cd CryoAlign2
cmake -S . -B build \
  -DLIBTORCH_PATH=/path/to/libtorch
cmake --build build -j2
```

如果想完全重新编译：

```bash
cd CryoAlign2
rm -rf build
cmake -S . -B build \
  -DLIBTORCH_PATH=/path/to/libtorch
cmake --build build -j2
```

也可以设置 `LIBTORCH_PATH` 环境变量，或通过 `CMAKE_PREFIX_PATH` 提供 Torch 的
CMake 包路径。

## 运行模式

### 1. CPU 单进程

默认就是 CPU，不需要额外参数：

```bash
./bin/CryoAlign_extract_keypoints \
  --data_dir /tmp/cryoalign_v6_test \
  --map_name emd_3661.map \
  --contour_level 0.07 \
  --voxel_size 5.0
```

也可以显式指定：

```bash
./bin/CryoAlign_extract_keypoints ... --cpu
```

### 2. CPU + MPI

MPI 主要用于 CPU 路径里的抽点卷积加速。建议限制每个 rank 的 BLAS/OpenMP 线程，避免互相抢核：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign_extract_keypoints \
  --data_dir /tmp/cryoalign_v6_test \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0
```

日志里出现下面内容，说明走了 MPI CPU 卷积：

```text
MPI CPU conv3d enabled with 4 ranks
MPI CPU conv3d finished
```

### 3. GPU 可选模式

GPU 只建议单进程运行，不建议 `mpirun -np 4` 抢同一张 GPU：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./bin/CryoAlign_extract_keypoints \
  --data_dir /tmp/cryoalign_v6_test \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0 \
  --use_gpu
```

日志里出现下面内容，说明走了 GPU 路径：

```text
CUDA conv3d + CUDA meanshift enabled
CUDA conv3d finished
```

也可以用环境变量启用 GPU：

```bash
CRYOALIGN_USE_GPU=1 ./bin/CryoAlign_extract_keypoints ...
```

强制禁用 GPU：

```bash
CRYOALIGN_DISABLE_GPU=1 ./bin/CryoAlign_extract_keypoints ...
```

## 输出文件

抽点会在 `--data_dir` 下生成：

```text
<map_without_ext>_<voxel_size>.txt
Points_<map_id>_<voxel_size>_Key.xyz
```

例如：

```text
emd_3661_5.00.txt
Points_3661_5.00_Key.xyz
emd_6647_5.00.txt
Points_6647_5.00_Key.xyz
```

`*.txt` 是 sample 点文件，`Points_*_Key.xyz` 是 DBSCAN 后的 keypoints。

## 示例数据

测试数据：

```text
/path/to/example_dataset/emd_3661_emd_6647
```

本例使用的 contour level：

```text
emd_3661: 0.07
emd_6647: 0.017
```

为了不污染原始数据目录，建议用 `/tmp` 目录和软链接运行：

```bash
rm -rf /tmp/cryoalign_v6_3661_6647
mkdir -p /tmp/cryoalign_v6_3661_6647
ln -sf /path/to/example_dataset/emd_3661_emd_6647/emd_3661.map /tmp/cryoalign_v6_3661_6647/emd_3661.map
ln -sf /path/to/example_dataset/emd_3661_emd_6647/emd_6647.map /tmp/cryoalign_v6_3661_6647/emd_6647.map
```

## 只抽 keypoints

### CPU 默认抽 3661

```bash
cd CryoAlign2

./bin/CryoAlign_extract_keypoints \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --map_name emd_3661.map \
  --contour_level 0.07 \
  --voxel_size 5.0
```

### CPU + MPI 抽 6647

```bash
cd CryoAlign2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign_extract_keypoints \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0
```

### GPU 抽 6647

```bash
cd CryoAlign2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./bin/CryoAlign_extract_keypoints \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --map_name emd_6647.map \
  --contour_level 0.017 \
  --voxel_size 5.0 \
  --use_gpu
```

## 完整流程：抽点 + global alignment

CPU 默认完整流程：

```bash
cd CryoAlign2

./bin/CryoAlign \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --source_map emd_3661.map \
  --source_contour_level 0.07 \
  --target_map emd_6647.map \
  --target_contour_level 0.017 \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global
```

CPU + MPI 完整流程：

```bash
cd CryoAlign2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/CryoAlign \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --source_map emd_3661.map \
  --source_contour_level 0.07 \
  --target_map emd_6647.map \
  --target_contour_level 0.017 \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global
```

GPU 完整流程：

```bash
cd CryoAlign2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./bin/CryoAlign \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --source_map emd_3661.map \
  --source_contour_level 0.07 \
  --target_map emd_6647.map \
  --target_contour_level 0.017 \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global \
  --use_gpu
```

注意：GPU 完整流程速度快，但 keypoints 可能和 CPU 稳定版有小数值差异，最终 alignment 也可能选到不同解。最终可复现实验建议使用 CPU/MPI。

## 只做 alignment

如果已经有 keypoint 文件和 sample 文件，可以只跑 alignment：

```bash
cd CryoAlign2

./bin/CryoAlign_alignment \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --source_xyz Points_3661_5.00_Key.xyz \
  --target_xyz Points_6647_5.00_Key.xyz \
  --source_sample emd_3661_5.00.txt \
  --target_sample emd_6647_5.00.txt \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type global
```

`CryoAlign_alignment` 的文件参数是相对 `--data_dir` 的文件名，不建议传绝对路径。

## mask alignment

将 `--alg_type global` 改成 `--alg_type mask` 即可：

```bash
./bin/CryoAlign \
  --data_dir /tmp/cryoalign_v6_3661_6647 \
  --source_map emd_3661.map \
  --source_contour_level 0.07 \
  --target_map emd_6647.map \
  --target_contour_level 0.017 \
  --voxel_size 5.0 \
  --feature_radius 7.0 \
  --alg_type mask
```

如果使用 MPI，mask alignment 的局部任务也会被分配到多个 rank。

## 已测基准

测试环境：WSL，RTX 3080，`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

### CPU/MPI 稳定路径

重启 WSL 后重新测试的抽点时间：

| map | v3/CPU 单进程 | v4/V6 CPU MPI `np=4` | 加速比 | CPU keypoints |
|---|---:|---:|---:|---:|
| `emd_3661`, contour `0.07` | `115.98s` | `30.71s` | `3.78x` | `919` |
| `emd_6647`, contour `0.017` | `1111.14s` | `322.78s` | `3.44x` | `2485` |

两个 map 合计：`1227.12s -> 353.49s`，约 `3.47x`。

### GPU 可选路径

V5/V6 GPU 路径实测：

| map | GPU 时间 | GPU keypoints | 备注 |
|---|---:|---:|---|
| `emd_3661`, contour `0.07` | `13.85s` | `918` | 可以完成 alignment，但和 CPU keypoints 数量差 1 |
| `emd_6647`, contour `0.017` | `131.97s` | `2485` | 数量一致，坐标有小差异 |

GPU 路径对 `emd_6647` 的抽点速度：

- 相对 v3 CPU 单进程：约 `8.42x`
- 相对 CPU MPI `np=4`：约 `2.45x`

## 正确性说明

CPU 默认路径是稳定路径，用于和 V3/V4 对齐：

- `emd_3661`: `919` keypoints
- `emd_6647`: `2485` keypoints

GPU 路径是加速实验路径：

- 卷积和 meanshift 在 GPU 上执行。
- DBSCAN 仍然在 CPU 上执行。
- GPU 浮点路径会让 meanshift 后的点产生小差异，DBSCAN 对边界点敏感，因此 keypoints 或 alignment 可能和 CPU 稳定路径不完全一致。

## 诊断模式

V6 保留一个诊断环境变量，用来判断差异来自 GPU conv 还是 GPU meanshift：

```bash
CRYOALIGN_GPU_CONV_CPU_MEANSHIFT=1 ./bin/CryoAlign_extract_keypoints ... --use_gpu
```

该模式使用 GPU 计算 `C`，再把 `C` 拷回 CPU 做 meanshift。它不建议作为常规运行模式，只用于排查数值差异。

## 常见问题

### 1. GPU 模式不要用多个 MPI rank 抢同一张 GPU

推荐：

```bash
./bin/CryoAlign_extract_keypoints ... --use_gpu
```

不推荐：

```bash
mpirun -np 4 ./bin/CryoAlign_extract_keypoints ... --use_gpu
```

V6 检测到多 MPI rank 时会回到 CPU/MPI 路径。

### 2. 大图 CPU 单进程可能吃很多内存

`emd_6647` 是 `512^3`，V3/CPU 单进程旧路径峰值内存很高。如果 WSL 已运行很久，可能 OOM。可以先在 Windows PowerShell 中重启 WSL：

```powershell
wsl --shutdown
```

然后重新进入 WSL/Codex 再运行。

### 3. 查看 GPU 显存

```bash
nvidia-smi
```

### 4. 参数帮助

```bash
./bin/CryoAlign --help
./bin/CryoAlign_extract_keypoints --help
./bin/CryoAlign_alignment --help
```
