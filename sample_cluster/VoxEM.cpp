#include "VoxEM.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>  // Added for std::copy
#include <stdexcept>  // Added for exceptions
#include <iterator>
#include <sstream>
#include <limits>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <ATen/Context.h>
#include <numeric>
#include <omp.h>
#include <mpi.h>
#include <climits>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
namespace {

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return !value.empty() && value != "0" && value != "false" && value != "off" && value != "no";
}

void mpi_bcast_float_buffer(float* data, size_t count, int root, MPI_Comm comm) {
    size_t offset = 0;
    while (offset < count) {
        int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        MPI_Bcast(data + offset, chunk, MPI_FLOAT, root, comm);
        offset += static_cast<size_t>(chunk);
    }
}

void mpi_bcast_float_vector(std::vector<float>& values, int root, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    unsigned long long size = rank == root ? static_cast<unsigned long long>(values.size()) : 0ULL;
    MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
    if (rank != root) {
        values.resize(static_cast<size_t>(size));
    }
    if (size > 0) {
        mpi_bcast_float_buffer(values.data(), static_cast<size_t>(size), root, comm);
    }
}

std::pair<int64_t, int64_t> mpi_slab_range(int64_t total, int rank, int size) {
    const int64_t base = total / size;
    const int64_t remainder = total % size;
    const int64_t start = static_cast<int64_t>(rank) * base + std::min<int64_t>(rank, remainder);
    const int64_t count = base + (rank < remainder ? 1 : 0);
    return {start, count};
}

void mpi_send_float_buffer(const float* data, size_t count, int dest, int tag, MPI_Comm comm) {
    size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        MPI_Send(data + offset, chunk, MPI_FLOAT, dest, tag, comm);
        offset += static_cast<size_t>(chunk);
    }
}

void mpi_recv_float_buffer(float* data, size_t count, int source, int tag, MPI_Comm comm) {
    size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        MPI_Recv(data + offset, chunk, MPI_FLOAT, source, tag, comm, MPI_STATUS_IGNORE);
        offset += static_cast<size_t>(chunk);
    }
}

std::vector<float> compute_local_centroid_field_cpu(
    const std::vector<std::vector<std::vector<float>>>& voxel,
    int64_t owned_start,
    int64_t owned_count,
    int window,
    float bandwidth,
    const std::array<float, 3>& interval) {
    if (owned_count <= 0) {
        return {};
    }

    const int64_t D = static_cast<int64_t>(voxel.size());
    const int64_t H = D > 0 ? static_cast<int64_t>(voxel[0].size()) : 0;
    const int64_t W = H > 0 ? static_cast<int64_t>(voxel[0][0].size()) : 0;
    const int64_t pad = window / 2;
    const int64_t halo_start = std::max<int64_t>(0, owned_start - pad);
    const int64_t halo_end = std::min<int64_t>(D, owned_start + owned_count + pad);
    const int64_t local_D = halo_end - halo_start;
    const int64_t local_hw = H * W;
    const size_t local_voxel_count = static_cast<size_t>(local_D * local_hw);

    std::vector<float> vox_flat;
    vox_flat.reserve(local_voxel_count);
    for (int64_t x = halo_start; x < halo_end; ++x) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t z = 0; z < W; ++z) {
                vox_flat.push_back(voxel[static_cast<size_t>(x)][static_cast<size_t>(y)][static_cast<size_t>(z)]);
            }
        }
    }

    torch::NoGradGuard no_grad;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor Q = torch::from_blob(vox_flat.data(), {1, 1, local_D, H, W}, options).clone();

    auto mgrid = torch::meshgrid({
        torch::arange(0, window, options),
        torch::arange(0, window, options),
        torch::arange(0, window, options)
    }, "xy");

    auto w_x = mgrid[0] * interval[0];
    auto w_y = mgrid[1] * interval[1];
    auto w_z = mgrid[2] * interval[2];
    const std::vector<float> centroid = {
        window / 2.0f * interval[0],
        window / 2.0f * interval[1],
        window / 2.0f * interval[2]
    };
    w_x = (w_x - centroid[0]).pow(2);
    w_y = (w_y - centroid[1]).pow(2);
    w_z = (w_z - centroid[2]).pow(2);
    torch::Tensor W_tensor = torch::sqrt(w_x + w_y + w_z);
    W_tensor = torch::exp(-1.5 * (W_tensor / bandwidth).pow(2));
    W_tensor = W_tensor.view({1, 1, window, window, window});

    torch::Tensor denominator = torch::conv3d(Q, W_tensor, {}, 1, pad);
    denominator += 0.000001;
    denominator = 1.000 / denominator;

    const int64_t trim_start = owned_start - halo_start;
    const int64_t owned_hw = owned_count * local_hw;
    std::vector<float> local_C_vec(static_cast<size_t>(3 * owned_hw));
    for (int dim = 0; dim < 3; ++dim) {
        torch::Tensor coord;
        if (dim == 0) {
            coord = torch::arange(halo_start, halo_end, options).view({1, 1, local_D, 1, 1}) * interval[0];
        } else if (dim == 1) {
            coord = torch::arange(0, H, options).view({1, 1, 1, H, 1}) * interval[0];
        } else {
            coord = torch::arange(0, W, options).view({1, 1, 1, 1, W}) * interval[0];
        }
        torch::Tensor P = Q * coord;
        torch::Tensor nominator = torch::conv3d(P, W_tensor, {}, 1, pad);
        torch::Tensor C_dim = (nominator * denominator).contiguous();
        const float* C_data = C_dim.data_ptr<float>();
        const int64_t dst_dim_offset = dim * owned_hw;
        for (int64_t x = 0; x < owned_count; ++x) {
            const int64_t src_offset = (trim_start + x) * local_hw;
            const int64_t dst_offset = dst_dim_offset + x * local_hw;
            std::copy(C_data + src_offset,
                      C_data + src_offset + local_hw,
                      local_C_vec.begin() + dst_offset);
        }
    }

    return local_C_vec;
}

void mpi_gather_slab_field(
    std::vector<float>& C_vec,
    const std::vector<float>& local_C_vec,
    int64_t D,
    int64_t H,
    int64_t W,
    int mpi_rank,
    int mpi_size,
    MPI_Comm comm) {
    constexpr int kSlabFieldTag = 7312;
    const int64_t hw = H * W;
    if (mpi_rank == 0) {
        C_vec.assign(static_cast<size_t>(3 * D * hw), 0.0f);
        for (int r = 0; r < mpi_size; ++r) {
            const auto [slab_start, slab_count] = mpi_slab_range(D, r, mpi_size);
            const size_t slab_values = static_cast<size_t>(3 * slab_count * hw);
            std::vector<float> slab_buffer;
            const std::vector<float>* src = nullptr;
            if (r == 0) {
                src = &local_C_vec;
            } else {
                slab_buffer.resize(slab_values);
                if (slab_values > 0) {
                    mpi_recv_float_buffer(slab_buffer.data(), slab_values, r, kSlabFieldTag, comm);
                }
                src = &slab_buffer;
            }

            const int64_t src_dim_stride = slab_count * hw;
            const int64_t dst_dim_stride = D * hw;
            for (int dim = 0; dim < 3; ++dim) {
                std::copy(src->begin() + static_cast<size_t>(dim * src_dim_stride),
                          src->begin() + static_cast<size_t>((dim + 1) * src_dim_stride),
                          C_vec.begin() + static_cast<size_t>(dim * dst_dim_stride + slab_start * hw));
            }
        }
    } else if (!local_C_vec.empty()) {
        mpi_send_float_buffer(local_C_vec.data(), local_C_vec.size(), 0, kSlabFieldTag, comm);
    }
}

void meanshift_cpu_from_centroid_field(
    const std::vector<float>& C_vec,
    std::vector<float>& Y,
    int64_t D,
    int64_t H,
    int64_t W,
    const std::array<float, 3>& interval,
    float lower_bound,
    int iteration,
    float step_size,
    float convergence) {
    const size_t num_cols = static_cast<size_t>(D);
    const size_t num_rows = static_cast<size_t>(H);
    const size_t num_slices = static_cast<size_t>(W);
    const size_t num_points = Y.size() / 3;
    float Y_diff_magnitude = std::numeric_limits<float>::max();
    int i = 0;
    std::vector<float> Y_diff(Y.size(), 0.0f);

    while (i <= iteration && Y_diff_magnitude >= convergence) {
        float sum_of_squares = 0.0f;

        #pragma omp parallel for reduction(+:sum_of_squares) schedule(dynamic)
        for (size_t j = 0; j < num_points; ++j) {
            std::array<int, 3> Y_indice;
            std::array<float, 3> Y_proposed;
            std::array<float, 3> inv_interval = {1.0f / interval[0], 1.0f / interval[1], 1.0f / interval[2]};

            for (int k = 0; k < 3; ++k) {
                const int idx = static_cast<int>(std::round((Y[j * 3 + k] - interval[k]) * inv_interval[k])) + 1;
                const int upper = static_cast<int>((k == 0) ? num_cols : (k == 1) ? num_rows : num_slices) - 1;
                Y_indice[k] = std::clamp(idx, 0, upper);
            }

            for (int dim = 0; dim < 3; ++dim) {
                size_t index = (dim * num_cols * num_rows * num_slices) + (Y_indice[0] * num_rows * num_slices) + (Y_indice[1] * num_slices) + Y_indice[2];
                if (index >= C_vec.size()) {
                    Y_proposed[dim] = 0;
                } else if (C_vec[index] < 0) {
                    Y_proposed[dim] = 0;
                } else if (C_vec[index] > 1000) {
                    Y_proposed[dim] = 0;
                } else {
                    Y_proposed[dim] = C_vec[index];
                }
            }

            float sum_squares = std::inner_product(std::begin(Y_proposed), std::end(Y_proposed), std::begin(Y_proposed), 0.0f);
            bool is_below_threshold = sum_squares < lower_bound * lower_bound;

            for (int k = 0; k < 3; ++k) {
                float diff = is_below_threshold ? 0.0f : (Y[j * 3 + k] - Y_proposed[k]);
                Y_diff[j * 3 + k] = diff;
                sum_of_squares += diff * diff;
            }
        }

        Y_diff_magnitude = sum_of_squares / (3.0f * static_cast<float>(num_points));
        for (size_t idx = 0; idx < Y.size(); ++idx) {
            Y[idx] -= Y_diff[idx] * step_size;
        }
        if (i % 50 == 0) {
            std::cout << "Meanshift: iteration " << i << " convergence " << Y_diff_magnitude << std::endl;
        }
        i++;
    }
}

}
// Constructor
VoxEM::VoxEM() {
    // Initialize Workspaces
}

void VoxEM::IO_ReadMrc(const std::string& mrc_inputname, const std::string& voxel_outputname, const std::string& grid_outputname, const std::string& description_outputname, const std::string& statistics_outputname, bool nonzerostatistics) {
	using namespace boost::interprocess;
	
    try {
		// 打开文件并创建内存映射
		file_mapping file(mrc_inputname.c_str(), read_only);
		mapped_region region(file, read_only);
		auto* MRCdata = static_cast<char*>(region.get_address());
		std::size_t size = region.get_size();
		
        // Dimension of column, row, and section in the unit cell
        int nx, ny, nz;
        std::tie(nx, ny, nz) = unpack_ints(MRCdata, 0);

        // Mode
        int mode = unpack_int(MRCdata, 12);

        // Start
        int xs, ys, zs;
        std::tie(xs, ys, zs) = unpack_ints(MRCdata, 16);

        // Sampling along axes
        int mx, my, mz;
        std::tie(mx, my, mz) = unpack_ints(MRCdata, 28);

        // Cell dimension in angstrom
        float X_angstrom, Y_angstrom, Z_angstrom;
        std::tie(X_angstrom, Y_angstrom, Z_angstrom) = unpack_floats(MRCdata, 40);
        std::array<float, 3> angstrom{X_angstrom, Y_angstrom, Z_angstrom};

        // Cell angle in degree
        float X_degree, Y_degree, Z_degree;
        std::tie(X_degree, Y_degree, Z_degree) = unpack_floats(MRCdata, 52);
        std::array<float, 3> angle{X_degree, Y_degree, Z_degree};

        // Axis
        int MAPC, MAPR, MAPS;
        std::tie(MAPC, MAPR, MAPS) = unpack_ints(MRCdata, 64);
        std::array<int, 3> axis{MAPC, MAPR, MAPS};

        // Misc
        int ISPG, NSYMBT;
        std::tie(ISPG, NSYMBT, std::ignore) = unpack_ints(MRCdata, 88);

        // Extra
        std::vector<float> EXTRA = unpack_float_array(MRCdata, 96, 12);

        // Origin
        float X_origin, Y_origin, Z_origin;
        std::tie(X_origin, Y_origin, Z_origin) = unpack_floats(MRCdata, 196);
        std::array<float, 3> origin{X_origin, Y_origin, Z_origin};

        // Character string 'MAP ' to identify file type
        std::string MAP_String(MRCdata + 208, MRCdata + 212);
        // Machine Stamp
        std::array<unsigned char, 4> MACHST;
        std::copy(MRCdata + 212, MRCdata + 216, MACHST.begin());

        // Number of labels in use
        int NLABL = unpack_int(MRCdata, 220);

        // Density
        // The original voxel from the file is stored in this.Voxel
        // 直接从映射的内存读取体素数据
        std::vector<std::vector<std::vector<float>>> Voxel(nx, std::vector<std::vector<float>>(ny, std::vector<float>(nz)));
        auto* voxelData = reinterpret_cast<float*>(MRCdata + 1024); // 假设头部是1024字节
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    Voxel[i][j][k] = voxelData[k * ny * nx + j * nx + i];
                }
            }
        }

        // Workspaces
        this->voxel_workspace[voxel_outputname] = Voxel;

        if (!grid_outputname.empty()) {
            this->Point_Create_Grid(grid_outputname);
        }

        if (!statistics_outputname.empty()) {
            this->IO_Statistics(voxel_outputname, statistics_outputname, nonzerostatistics);
        }

        if (!description_outputname.empty()) {
            this->description_workspace[description_outputname] = {
                {"Mode", mode},
                {"Start", std::array<int, 3>{xs, ys, zs}},
                {"UnitCellDim", std::array<int, 3>{mx, my, mz}},
                {"Angstrom", angstrom},
                {"Angle", angle},
                {"Axis", axis},
                {"Extra", EXTRA},
                {"Ispg", ISPG},
                {"Nsymbt", NSYMBT},
                {"MapString", MAP_String},
                {"Origin", origin},
                {"MachineString", MACHST},
                {"Label", NLABL}
            };
        }
        std::cout << "IO_ReadMrc finished" << std::endl;
		//std::cout << torch::cuda::is_available() <<std:: endl;
    	//std::cout << torch::cuda::cudnn_is_available() <<std:: endl;
    	//std::cout << torch::cuda::device_count() <<std:: endl;
    }catch (const interprocess_exception& ex) {
        std::cerr << "Failed to open the MRC file: " << ex.what() << std::endl;
    }
}
void VoxEM::Point_Create_Grid(const std::string& voxel_inputname, const std::string& point_outputname, const std::string& description_inputname) {
    // Create grid points from voxel

    // Interval between grid
    const std::array<float, 3>& angstrom = std::any_cast<const std::array<float, 3>&>(description_workspace[description_inputname]["Angstrom"]);
    const std::vector<std::vector<std::vector<float>>>& voxel = voxel_workspace[voxel_inputname];

    std::vector<float> interval(3);
    interval[0] = angstrom[0] / static_cast<float>(voxel.size());
    interval[1] = angstrom[1] / static_cast<float>(voxel[0].size());
    interval[2] = angstrom[2] / static_cast<float>(voxel[0][0].size());

    std::vector<float> point_scatter(3 * voxel.size() * voxel[0].size() * voxel[0][0].size());

    int index = 0;
    for (size_t i = 0; i < voxel.size(); ++i) {
        for (size_t j = 0; j < voxel[i].size(); ++j) {
            for (size_t k = 0; k < voxel[i][j].size(); ++k) {
                point_scatter[index] = i * interval[0];
                point_scatter[index + 1] = j * interval[0];
                point_scatter[index + 2] = k * interval[0];
                index += 3;
            }
        }
    }

    this->point_workspace[point_outputname] = point_scatter;
}

void VoxEM::Point_Create_Meanshift_sample(float lower_bound, int window, float bandwidth, int iteration, float step_size, float convergence, const std::string& voxel_inputname, const std::string& point_outputname, const std::string& point_inputname, const std::string& description_inputname) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    int mpi_rank = 0;
    int mpi_size = 1;
    if (mpi_initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }

    const bool gpu_requested = env_flag_enabled("CRYOALIGN_USE_GPU");
    const bool gpu_disabled = env_flag_enabled("CRYOALIGN_DISABLE_GPU");
    if (gpu_requested && gpu_disabled && mpi_rank == 0) {
        std::cout << "GPU requested but CRYOALIGN_DISABLE_GPU is set; using CPU path." << std::endl;
    }
    if (gpu_requested && !gpu_disabled && mpi_size > 1 && mpi_rank == 0) {
        std::cout << "GPU requested with multiple MPI ranks; using CPU/MPI path." << std::endl;
    }
    if (gpu_requested && !gpu_disabled && mpi_size == 1 && !torch::cuda::is_available() && mpi_rank == 0) {
        std::cout << "GPU requested but CUDA is unavailable; using CPU path." << std::endl;
    }
    if (gpu_requested && !gpu_disabled && mpi_size == 1 && torch::cuda::is_available()) {
        try {
            if (window % 2 == 0) {
                std::cout << "User should consider an odd window, which centers the distance window!" << std::endl;
            }

            const std::array<float, 3>& angstrom = std::any_cast<const std::array<float, 3>&>(description_workspace[description_inputname]["Angstrom"]);
            const std::vector<std::vector<std::vector<float>>>& voxel = this->voxel_workspace[voxel_inputname];
            const int64_t D = static_cast<int64_t>(voxel.size());
            const int64_t H = D > 0 ? static_cast<int64_t>(voxel[0].size()) : 0;
            const int64_t W = H > 0 ? static_cast<int64_t>(voxel[0][0].size()) : 0;
            const int64_t pad = window / 2;
            const int64_t hw = H * W;
            std::array<float, 3> interval = {
                angstrom[0] / static_cast<float>(D),
                angstrom[1] / static_cast<float>(H),
                angstrom[2] / static_cast<float>(W)
            };

            std::cout << "interval[0]: " << interval[0] << std::endl;
            std::cout << "interval[1]: " << interval[1] << std::endl;
            std::cout << "interval[2]: " << interval[2] << std::endl;
            std::cout << "CUDA conv3d + CUDA meanshift enabled" << std::endl;

            std::vector<float> vox_flat;
            vox_flat.reserve(static_cast<size_t>(D * H * W));
            for (const auto& plane : voxel) {
                for (const auto& row : plane) {
                    for (float value : row) {
                        vox_flat.push_back(value);
                    }
                }
            }

            torch::NoGradGuard no_grad;
            at::globalContext().setAllowTF32CuDNN(false);
            at::globalContext().setAllowTF32CuBLAS(false);
            const torch::Device gpu_device(torch::kCUDA, 0);
            const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            const auto gpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(gpu_device);

            torch::Tensor Q = torch::from_blob(vox_flat.data(), {1, 1, D, H, W}, cpu_options).to(gpu_options);
            std::vector<float>().swap(vox_flat);

            auto mgrid = torch::meshgrid({
                torch::arange(0, window, gpu_options),
                torch::arange(0, window, gpu_options),
                torch::arange(0, window, gpu_options)
            }, "xy");

            auto w_x = mgrid[0] * interval[0];
            auto w_y = mgrid[1] * interval[1];
            auto w_z = mgrid[2] * interval[2];
            const std::vector<float> centroid = {
                window / 2.0f * interval[0],
                window / 2.0f * interval[1],
                window / 2.0f * interval[2]
            };
            w_x = (w_x - centroid[0]).pow(2);
            w_y = (w_y - centroid[1]).pow(2);
            w_z = (w_z - centroid[2]).pow(2);
            torch::Tensor W_tensor = torch::sqrt(w_x + w_y + w_z);
            W_tensor = torch::exp(-1.5 * (W_tensor / bandwidth).pow(2));
            W_tensor = W_tensor.view({1, 1, window, window, window});

            torch::Tensor denominator = torch::conv3d(Q, W_tensor, {}, 1, pad);
            denominator += 0.000001;
            denominator = 1.000 / denominator;

            torch::Tensor C = torch::empty({3, D, H, W}, gpu_options);
            for (int dim = 0; dim < 3; ++dim) {
                torch::Tensor coord;
                if (dim == 0) {
                    coord = torch::arange(0, D, gpu_options).view({1, 1, D, 1, 1}) * interval[0];
                } else if (dim == 1) {
                    coord = torch::arange(0, H, gpu_options).view({1, 1, 1, H, 1}) * interval[0];
                } else {
                    coord = torch::arange(0, W, gpu_options).view({1, 1, 1, 1, W}) * interval[0];
                }
                torch::Tensor P = Q * coord;
                torch::Tensor nominator = torch::conv3d(P, W_tensor, {}, 1, pad);
                C.select(0, dim).copy_((nominator * denominator).view({D, H, W}));
            }
            Q.reset();
            denominator.reset();
            W_tensor.reset();
            std::cout << "CUDA conv3d finished" << std::endl;

            std::vector<float>& Y = point_workspace[point_inputname];
            const size_t num_points = Y.size() / 3;
            std::cout << "Y.size:" << Y.size() << std::endl;
            if (num_points == 0) {
                this->point_workspace[point_outputname] = Y;
                std::cout << "Meanshift Completed." << std::endl;
                return;
            }

            const char* gpu_conv_cpu_ms_env = std::getenv("CRYOALIGN_GPU_CONV_CPU_MEANSHIFT");
            const bool gpu_conv_cpu_ms = gpu_conv_cpu_ms_env != nullptr && std::string(gpu_conv_cpu_ms_env) != "0";
            if (gpu_conv_cpu_ms) {
                std::cout << "CUDA conv3d + CPU meanshift diagnostic enabled" << std::endl;
                torch::Tensor C_cpu = C.to(cpu_options).contiguous();
                std::vector<float> C_vec(C_cpu.data_ptr<float>(), C_cpu.data_ptr<float>() + C_cpu.numel());
                C.reset();
                meanshift_cpu_from_centroid_field(
                    C_vec,
                    Y,
                    D,
                    H,
                    W,
                    interval,
                    lower_bound,
                    iteration,
                    step_size,
                    convergence);
                this->point_workspace[point_outputname] = Y;
                std::cout << "Meanshift Completed." << std::endl;
                return;
            }

            torch::Tensor Y_tensor = torch::from_blob(Y.data(), {static_cast<int64_t>(num_points), 3}, cpu_options).to(gpu_options);
            torch::Tensor interval_tensor = torch::tensor({interval[0], interval[1], interval[2]}, gpu_options).view({1, 3});
            torch::Tensor inv_interval_tensor = 1.0 / interval_tensor;
            torch::Tensor C_flat = C.view({3, D * H * W});

            float Y_diff_magnitude = std::numeric_limits<float>::max();
            int i = 0;
            while (i <= iteration && Y_diff_magnitude >= convergence) {
                torch::Tensor scaled = (Y_tensor - interval_tensor) * inv_interval_tensor;
                torch::Tensor rounded = torch::where(
                    scaled >= 0,
                    torch::floor(scaled + 0.5),
                    torch::ceil(scaled - 0.5));
                torch::Tensor indices = rounded.to(torch::kLong) + 1;
                indices.select(1, 0).clamp_(0, D - 1);
                indices.select(1, 1).clamp_(0, H - 1);
                indices.select(1, 2).clamp_(0, W - 1);

                torch::Tensor linear_index =
                    indices.select(1, 0) * hw +
                    indices.select(1, 1) * W +
                    indices.select(1, 2);
                torch::Tensor Y_proposed = C_flat.index_select(1, linear_index).transpose(0, 1).contiguous();
                torch::Tensor invalid = (Y_proposed < 0) | (Y_proposed > 1000);
                Y_proposed = torch::where(invalid, torch::zeros_like(Y_proposed), Y_proposed);

                torch::Tensor proposed_norm2 = Y_proposed.pow(2).sum(1);
                torch::Tensor below_threshold = proposed_norm2 < (lower_bound * lower_bound);
                torch::Tensor diff = Y_tensor - Y_proposed;
                diff = torch::where(below_threshold.unsqueeze(1), torch::zeros_like(diff), diff);

                const float sum_of_squares = diff.pow(2).sum().item<float>();
                Y_diff_magnitude = sum_of_squares / (3.0f * static_cast<float>(num_points));
                Y_tensor = Y_tensor - diff * step_size;

                if (i % 50 == 0) {
                    std::cout << "Meanshift: iteration " << i << " convergence " << Y_diff_magnitude << std::endl;
                }
                i++;
            }

            torch::Tensor Y_cpu = Y_tensor.to(cpu_options).contiguous().view({static_cast<int64_t>(Y.size())});
            std::memcpy(Y.data(), Y_cpu.data_ptr<float>(), Y.size() * sizeof(float));
            this->point_workspace[point_outputname] = Y;
            std::cout << "Meanshift Completed." << std::endl;
            return;
        } catch (const c10::Error& e) {
            std::cerr << "CUDA path failed, falling back to CPU path: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "CUDA path failed, falling back to CPU path: " << e.what() << std::endl;
        }
    }

    if (mpi_size > 1) {
        if (window % 2 == 0 && mpi_rank == 0) {
            std::cout << "User should consider an odd window, which centers the distance window!" << std::endl;
        }

        const std::array<float, 3>& angstrom = std::any_cast<const std::array<float, 3>&>(description_workspace[description_inputname]["Angstrom"]);
        const std::vector<std::vector<std::vector<float>>>& voxel = this->voxel_workspace[voxel_inputname];
        const int64_t D = static_cast<int64_t>(voxel.size());
        const int64_t H = D > 0 ? static_cast<int64_t>(voxel[0].size()) : 0;
        const int64_t W = H > 0 ? static_cast<int64_t>(voxel[0][0].size()) : 0;
        std::array<float, 3> interval = {
            angstrom[0] / static_cast<float>(D),
            angstrom[1] / static_cast<float>(H),
            angstrom[2] / static_cast<float>(W)
        };
        if (mpi_rank == 0) {
            std::cout << "interval[0]: " << interval[0] << std::endl;
            std::cout << "interval[1]: " << interval[1] << std::endl;
            std::cout << "interval[2]: " << interval[2] << std::endl;
            std::cout << "MPI CPU conv3d enabled with " << mpi_size << " ranks" << std::endl;
        }

        const auto [slab_start, slab_count] = mpi_slab_range(D, mpi_rank, mpi_size);
        std::vector<float> local_C_vec = compute_local_centroid_field_cpu(
            voxel,
            slab_start,
            slab_count,
            window,
            bandwidth,
            interval);

        std::vector<float> C_vec;
        mpi_gather_slab_field(C_vec, local_C_vec, D, H, W, mpi_rank, mpi_size, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            std::cout << "MPI CPU conv3d finished" << std::endl;
        }
        mpi_bcast_float_vector(C_vec, 0, MPI_COMM_WORLD);

        std::vector<float> Y;
        size_t num_points = 0;
        if (mpi_rank == 0) {
            Y = point_workspace[point_inputname];
            num_points = Y.size() / 3;
            std::cout << "Y.size:" << Y.size() << std::endl;
        }

        unsigned long long y_size = mpi_rank == 0 ? static_cast<unsigned long long>(Y.size()) : 0ULL;
        MPI_Bcast(&y_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        num_points = static_cast<size_t>(y_size) / 3;
        if (num_points == 0) {
            if (mpi_rank == 0) {
                this->point_workspace[point_outputname] = Y;
            }
            return;
        }

        const size_t base_count = num_points / static_cast<size_t>(mpi_size);
        const size_t remainder = num_points % static_cast<size_t>(mpi_size);
        const size_t start = static_cast<size_t>(mpi_rank) * base_count + std::min<size_t>(static_cast<size_t>(mpi_rank), remainder);
        const size_t local_points = base_count + (static_cast<size_t>(mpi_rank) < remainder ? 1 : 0);

        std::vector<int> recv_counts(mpi_size, 0);
        std::vector<int> displs(mpi_size, 0);
        for (int r = 0; r < mpi_size; ++r) {
            size_t r_start = static_cast<size_t>(r) * base_count + std::min<size_t>(static_cast<size_t>(r), remainder);
            size_t r_points = base_count + (static_cast<size_t>(r) < remainder ? 1 : 0);
            recv_counts[r] = static_cast<int>(r_points * 3);
            displs[r] = static_cast<int>(r_start * 3);
        }
        std::vector<float> local_Y(local_points * 3);
        MPI_Scatterv(mpi_rank == 0 ? Y.data() : nullptr,
                     recv_counts.data(),
                     displs.data(),
                     MPI_FLOAT,
                     local_Y.empty() ? nullptr : local_Y.data(),
                     recv_counts[mpi_rank],
                     MPI_FLOAT,
                     0,
                     MPI_COMM_WORLD);

        const size_t num_cols = static_cast<size_t>(D);
        const size_t num_rows = static_cast<size_t>(H);
        const size_t num_slices = static_cast<size_t>(W);
        float Y_diff_magnitude = std::numeric_limits<float>::max();
        int i = 0;

        try {
            while (i <= iteration && Y_diff_magnitude >= convergence) {
                float local_sum_of_squares = 0.0f;

                for (size_t j = 0; j < local_points; ++j) {
                    std::array<int, 3> Y_indice;
                    std::array<float, 3> Y_proposed;
                    std::array<float, 3> inv_interval = {1.0f / interval[0], 1.0f / interval[1], 1.0f / interval[2]};

                    for (int k = 0; k < 3; ++k) {
                        const int idx = static_cast<int>(std::round((local_Y[j * 3 + k] - interval[k]) * inv_interval[k])) + 1;
                        const int upper = static_cast<int>((k == 0) ? num_cols : (k == 1) ? num_rows : num_slices) - 1;
                        Y_indice[k] = std::clamp(idx, 0, upper);
                    }

                    for (int dim = 0; dim < 3; ++dim) {
                        size_t index = (dim * num_cols * num_rows * num_slices) + (Y_indice[0] * num_rows * num_slices) + (Y_indice[1] * num_slices) + Y_indice[2];
                        if (index >= C_vec.size()) {
                            Y_proposed[dim] = 0;
                        } else if (C_vec[index] < 0) {
                            Y_proposed[dim] = 0;
                        } else if (C_vec[index] > 1000) {
                            Y_proposed[dim] = 0;
                        } else {
                            Y_proposed[dim] = C_vec[index];
                        }
                    }

                    float sum_squares = std::inner_product(std::begin(Y_proposed), std::end(Y_proposed), std::begin(Y_proposed), 0.0f);
                    bool is_below_threshold = sum_squares < lower_bound * lower_bound;

                    for (int k = 0; k < 3; ++k) {
                        float diff = is_below_threshold ? 0.0f : (local_Y[j * 3 + k] - Y_proposed[k]);
                        local_sum_of_squares += diff * diff;
                        local_Y[j * 3 + k] -= diff * step_size;
                    }
                }

                float global_sum_of_squares = 0.0f;
                MPI_Allreduce(&local_sum_of_squares, &global_sum_of_squares, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

                Y_diff_magnitude = global_sum_of_squares / (3.0f * static_cast<float>(num_points));
                if (mpi_rank == 0 && i % 50 == 0) {
                    std::cout << "Meanshift: iteration " << i << " convergence " << Y_diff_magnitude << std::endl;
                }
                i++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Rank " << mpi_rank << " meanshift error: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        MPI_Gatherv(local_Y.empty() ? nullptr : local_Y.data(),
                    recv_counts[mpi_rank],
                    MPI_FLOAT,
                    mpi_rank == 0 ? Y.data() : nullptr,
                    recv_counts.data(),
                    displs.data(),
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            this->point_workspace[point_outputname] = Y;
            std::cout << "Meanshift Completed." << std::endl;
        }
        return;
    }
    
    if (window % 2 == 0) {
        std::cout << "User should consider an odd window, which centers the distance window!" << std::endl;
    }
    const std::array<float, 3>& angstrom = std::any_cast<const std::array<float, 3>&>(description_workspace[description_inputname]["Angstrom"]);
    const std::vector<std::vector<std::vector<float>>>& voxel = this->voxel_workspace[voxel_inputname];

    std::vector<float> interval(3);
    interval[0] = angstrom[0] / static_cast<float>(voxel.size());
    interval[1] = angstrom[1] / static_cast<float>(voxel[0].size());
    interval[2] = angstrom[2] / static_cast<float>(voxel[0][0].size());
    std::cout << "interval[0]: " << interval[0] << std::endl;
    std::cout << "interval[1]: " << interval[1] << std::endl;
    std::cout << "interval[2]: " << interval[2] << std::endl;

    this->Point_Create_Grid();
    std::vector<float> grid = this->point_workspace["Default"];
    std::vector<std::vector<std::vector<float>>> vox = this->voxel_workspace[voxel_inputname];
    int64_t D = vox.size();
    int64_t H = D > 0 ? vox[0].size() : 0;
    int64_t W = H > 0 ? vox[0][0].size() : 0;

    // Flatten 'vox' into a 1D vector for creating a tensor
    std::vector<float> vox_flat;
    vox_flat.reserve(D * H * W);
    for (auto& plane : vox) {
        for (auto& row : plane) {
            for (float value : row) {
                vox_flat.push_back(value);
            }
        }
    }

    torch::NoGradGuard no_grad;
    const auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Create tensors from the flat vectors on CPU. Moving conv3d results from GPU is too expensive here.
    torch::Tensor Q = torch::from_blob(vox_flat.data(), {D, H, W}, tensor_options).clone();
    Q = Q.repeat({3, 1, 1, 1});
    
    // Assuming grid is a flat vector with every 3 elements representing one point (x, y, z)
    // and that the total number of points is equal to the product of D, H, and W.
    torch::Tensor X = torch::from_blob(grid.data(), {D, H, W, 3}, tensor_options).clone();

    // Transpose X to have the size [3, D, H, W]
    X = X.permute({3, 0, 1, 2});

    // Compute the background field P
    torch::Tensor P = Q * X;
    P = P.view({3, 1, D, H, W});
    Q = Q.view({3, 1, D, H, W});
    X = X.view({3, 1, D, H, W});
    std::cout << "Q size: ";
    for (auto s : Q.sizes()) std::cout << s << " ";
    std::cout << "\n";

    std::cout << "X size: ";
    for (auto s : X.sizes()) std::cout << s << " ";
    std::cout << "\n";


    // A distance window matrix W to convolve with
    auto mgrid = torch::meshgrid({torch::arange(0, window, tensor_options),
                              torch::arange(0, window, tensor_options),
                              torch::arange(0, window, tensor_options)},"xy");

    auto w_x = mgrid[0] * interval[0];
    auto w_y = mgrid[1] * interval[1];
    auto w_z = mgrid[2] * interval[2];
    // 计算窗口的质心，用于高斯核
    std::vector<float> centroid = {window / 2.0f * interval[0], window / 2.0f * interval[1], window / 2.0f * interval[2]};
    w_x = (w_x - centroid[0]).pow(2);
    w_y = (w_y - centroid[1]).pow(2);
    w_z = (w_z - centroid[2]).pow(2);
    torch::Tensor W_tensor = torch::sqrt(w_x + w_y + w_z);

    // 应用高斯核
    W_tensor = torch::exp(-1.5 * (W_tensor / bandwidth).pow(2));
    W_tensor = W_tensor.view({1, 1, window, window, window});

    // 预先计算每个网格的 Y_i+1
    torch::Tensor nominator = torch::conv3d(P, W_tensor, {}, 1, window / 2);
    torch::Tensor denominator = torch::conv3d(Q, W_tensor, {}, 1, window / 2);
    denominator += 0.000001;
    denominator = 1.000 / denominator;

    // 释放内存
    P.reset();
    Q.reset();
    W_tensor.reset();
    X.reset();
    std::cout << "conv3d finished \n";
    // 打印nominator的形状
    std::cout << "Nominator shape: " << nominator.sizes() << std::endl;

    // 打印denominator的形状
    std::cout << "Denominator shape: " << denominator.sizes() << std::endl;
    
    torch::Tensor C = nominator * denominator;
    std::cout << "Tensor C = nominator * denominator Completed." << std::endl;
    
    C = C.squeeze(1);
    std::cout << "C.squeeze Completed." << std::endl;

    // 预计算用于访问的索引
    std::vector<float> C_vec(C.data_ptr<float>(), C.data_ptr<float>() + C.numel());
    std::cout << "C_vec Completed." << std::endl;
    
    
    std::vector<float>& Y = point_workspace[point_inputname];
    const size_t num_cols = C.size(1);
    const size_t num_rows = C.size(2);
    const size_t num_slices = C.size(3);
    const size_t num_points = Y.size() / 3;
    std::cout << "Y.size:" << Y.size() <<std::endl;
    float Y_diff_magnitude = std::pow(voxel.size(), 3);
    int i = 0;
    std::vector<float> Y_diff(Y.size(), 0.0f);
    try {
        while (i <= iteration && Y_diff_magnitude >= convergence) {
            float sum_of_squares = 0.0f;

            #pragma omp parallel for reduction(+:sum_of_squares) schedule(dynamic)
            for (size_t j = 0; j < num_points; ++j) {
                std::array<int, 3> Y_indice;
                std::array<float, 3> Y_proposed;
                std::array<float, 3> inv_interval = {1.0f / interval[0], 1.0f / interval[1], 1.0f / interval[2]};

                for (int k = 0; k < 3; ++k) {
                    const int idx = static_cast<int>(std::round((Y[j * 3 + k] - interval[k]) * inv_interval[k])) + 1;
                    const int upper = static_cast<int>((k == 0) ? num_cols : (k == 1) ? num_rows : num_slices) - 1;
                    Y_indice[k] = std::clamp(idx, 0, upper);
                }
                
                for (int dim = 0; dim < 3; ++dim) {
                    size_t index = (dim * num_cols * num_rows * num_slices) + (Y_indice[0] * num_rows * num_slices) + (Y_indice[1] * num_slices) + Y_indice[2];
                    if (index >= C_vec.size()) {
                        Y_proposed[dim]=0;
                    } else if (C_vec[index]<0){
                        Y_proposed[dim]=0;
                    } else if(C_vec[index]>1000){
                        Y_proposed[dim]=0;
                    } else{
                        Y_proposed[dim] = C_vec[index];
                    }
                }
                
                float sum_squares = std::inner_product(std::begin(Y_proposed), std::end(Y_proposed), std::begin(Y_proposed), 0.0f);
                bool is_below_threshold = sum_squares < lower_bound*lower_bound;

                for (int k = 0; k < 3; ++k) {
                    float diff = is_below_threshold ? 0.0f : (Y[j * 3 + k] - Y_proposed[k]);
                    Y_diff[j * 3 + k] = diff;
                    sum_of_squares += diff * diff;
                }
                
            }

            Y_diff_magnitude = sum_of_squares / (3*num_points);
            for (size_t idx = 0; idx < Y.size(); ++idx) {
                Y[idx] -= Y_diff[idx] * step_size;
            }
            if (i % 50 == 0) {
                std::cout << "Meanshift: iteration " << i << " convergence " << Y_diff_magnitude << std::endl;
            }
            i++;
        }
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }
    
    this->point_workspace[point_outputname] = Y;
    std::cout << "Meanshift Completed." << std::endl;
}
/*

void VoxEM::Point_Create_Meanshift_sample(float lower_bound, int window, float bandwidth, 
                                        int iteration, float step_size, float convergence,
                                        const std::string& voxel_inputname, const std::string& point_outputname,
                                        const std::string& point_inputname, const std::string& description_inputname) {
    
    // Initialize MPI
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Shared variables
    std::vector<float> C_vec;
    std::vector<float> Y;
    int64_t D = 0, H = 0, W = 0;
    std::array<float, 3> interval{};
    uint64_t num_points = 0;

    // Master process prepares data
    if (rank == 0) {
        if (window % 2 == 0) {
            std::cout << "Warning: Odd window size recommended for symmetric distance calculation." << std::endl;
        }

        // Load voxel data
        const auto& angstrom = std::any_cast<const std::array<float, 3>&>(
            description_workspace[description_inputname]["Angstrom"]);
        const auto& voxel = voxel_workspace[voxel_inputname];

        // Calculate intervals
        interval = {
            angstrom[0] / std::max(voxel.size(), 1UL),
            angstrom[1] / std::max((voxel.empty() ? 1UL : voxel[0].size()), 1UL),
            angstrom[2] / std::max((voxel.empty() || voxel[0].empty()) ? 1UL : voxel[0][0].size(), 1UL)
        };

        // Generate grid and voxel data
        Point_Create_Grid();
        const auto& grid = point_workspace["Default"];
        const auto& vox = voxel_workspace[voxel_inputname];
        D = vox.size();
        H = D > 0 ? vox[0].size() : 0;
        W = H > 0 ? vox[0][0].size() : 0;

        // Tensor construction
        std::vector<float> vox_flat;
        for (const auto& plane : vox) {
            for (const auto& row : plane) {
                vox_flat.insert(vox_flat.end(), row.begin(), row.end());
            }
        }

        torch::Tensor Q = torch::from_blob(vox_flat.data(), {D, H, W}, torch::kFloat32).clone();
        Q = Q.repeat({3, 1, 1, 1});  // Shape: [3, D, H, W]

        torch::Tensor X = torch::from_blob(grid.data(), {D, H, W, 3}, torch::kFloat32)
                          .permute({3, 0, 1, 2});  // Shape: [3, D, H, W]

        // Compute background field
        torch::Tensor P = Q * X;
        P = P.view({3, 1, D, H, W});
        Q = Q.view({3, 1, D, H, W});

        // Gaussian kernel construction
        auto mgrid = torch::meshgrid({
            torch::arange(0, window, torch::kFloat32),
            torch::arange(0, window, torch::kFloat32),
            torch::arange(0, window, torch::kFloat32)
        }, "xy");
        
        auto w_x = (mgrid[0] * interval[0] - (window/2.0f)*interval[0]).pow(2);
        auto w_y = (mgrid[1] * interval[1] - (window/2.0f)*interval[1]).pow(2);
        auto w_z = (mgrid[2] * interval[2] - (window/2.0f)*interval[2]).pow(2);
        
        torch::Tensor W_tensor = torch::exp(-1.5f * torch::sqrt(w_x + w_y + w_z).pow(2) / (bandwidth*bandwidth))
                                 .view({1, 1, window, window, window});

        // 3D convolution
        torch::Tensor nominator = torch::conv3d(P, W_tensor, {}, 1, window/2);
        torch::Tensor denominator = torch::conv3d(Q, W_tensor, {}, 1, window/2) + 1e-6f;
        torch::Tensor C = (nominator / denominator).squeeze(1);

        // Get data pointers
        C_vec = std::vector<float>(C.data_ptr<float>(), C.data_ptr<float>() + C.numel());
        Y = point_workspace[point_inputname];
        num_points = Y.size() / 3;
    }

    // Broadcast dimensions (int64_t)
    int64_t dims[3] = {D, H, W};
    MPI_Bcast(dims, 3, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    D = dims[0]; H = dims[1]; W = dims[2];

    // Broadcast interval data
    MPI_Bcast(interval.data(), 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Broadcast C_vec
    uint64_t c_vec_size = C_vec.size();
    MPI_Bcast(&c_vec_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) C_vec.resize(c_vec_size);
    MPI_Bcast(C_vec.data(), c_vec_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Broadcast Y data
    uint64_t y_size = 0;
    if (rank == 0) y_size = Y.size();
    MPI_Bcast(&y_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) Y.resize(y_size);
    MPI_Bcast(Y.data(), y_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    num_points = Y.size() / 3;

    // Domain decomposition
    const uint64_t chunk_size = num_points / size;
    const uint64_t remainder = num_points % size;
    const uint64_t start = rank * chunk_size + std::min<uint64_t>(rank, remainder);
    const uint64_t end = start + chunk_size + (rank < remainder ? 1 : 0);

    // Iteration parameters
    int current_iter = 0;
    float global_diff_norm = std::numeric_limits<float>::max();
    const size_t total_dims = D * H * W;

    try {
        while (current_iter <= iteration && global_diff_norm >= convergence) {
            std::vector<float> local_diff(Y.size(), 0.0f);
            float local_sq_sum = 0.0f;

            // Process local chunk
            for (uint64_t j = start; j < end && j < num_points; ++j) {
                std::array<int, 3> indices{};
                const std::array<float, 3> inv_interval = {
                    1.0f / interval[0], 
                    1.0f / interval[1], 
                    1.0f / interval[2]
                };

                // Calculate indices with boundary checks
                for (int dim = 0; dim < 3; ++dim) {
                    const float pos = Y[j*3 + dim] - interval[dim];
                    int idx = static_cast<int>(std::round(pos * inv_interval[dim])) + 1;
                    indices[dim] = std::clamp(idx, 0, static_cast<int>((dim == 0) ? D-1 : 
                                                                      (dim == 1) ? H-1 : W-1));
                }

                // Calculate proposed position
                std::array<float, 3> proposed{};
                for (int dim = 0; dim < 3; ++dim) {
                    const size_t linear_idx = dim * total_dims + 
                                           indices[0] * H*W + 
                                           indices[1] * W + 
                                           indices[2];
                    proposed[dim] = (linear_idx < C_vec.size()) ? 
                                   std::clamp(C_vec[linear_idx], 0.0f, 1000.0f) : 0.0f;
                }

                // Determine update
                const float sq_magnitude = proposed[0]*proposed[0] + 
                                         proposed[1]*proposed[1] + 
                                         proposed[2]*proposed[2];
                const bool skip_update = sq_magnitude < lower_bound * lower_bound;

                for (int dim = 0; dim < 3; ++dim) {
                    const float diff = skip_update ? 0.0f : (Y[j*3 + dim] - proposed[dim]);
                    local_diff[j*3 + dim] = diff;
                    local_sq_sum += diff * diff;
                }
            }

            // Global reduction
            float global_sq_sum;
            MPI_Allreduce(&local_sq_sum, &global_sq_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            std::vector<float> global_diff(Y.size());
            MPI_Allreduce(local_diff.data(), global_diff.data(), Y.size(), 
                        MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            // Update Y values
            #pragma omp parallel for
            for (size_t idx = 0; idx < Y.size(); ++idx) {
                Y[idx] -= global_diff[idx] * step_size;
            }

            // Calculate convergence
            global_diff_norm = global_sq_sum / (3.0f * num_points);
            if (rank == 0 && current_iter % 50 == 0) {
                std::cout << "Iteration " << current_iter 
                        << " | Residual: " << global_diff_norm 
                        << std::endl;
            }
            ++current_iter;
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Save results
    if (rank == 0) {
        point_workspace[point_outputname] = Y;
        std::cout << "Meanshift completed after " << current_iter << " iterations." << std::endl;
    }

    MPI_Finalize();
}
*/
// 计算质心的函数
arma::colvec3 VoxEM::CalculateCentroid(const int cluster_id,
                                const arma::mat& points,
                                const arma::Row<size_t>& labels) {
    arma::colvec3 centroid = arma::zeros(3);  // 修正了这里
    size_t count = 0;
    for (size_t i = 0; i < labels.n_elem; ++i) {
        if (labels(i) == cluster_id) {
            centroid += points.col(i);
            count++;
        }
    }
    if (count > 0) {
        centroid /= count;
    }
    return centroid;
}
void VoxEM::Point_Transform_DBSCAN(
                            const double distance_tolerance,
                            const size_t clustersize_tolerance,
                            const std::string& inputname,
                            const std::string& outputname,
                            const bool centroid_only,
                            const int n_job) {
	// 获取点云数据
    std::vector<float> &Y = this->point_workspace[inputname];
	std::cout << Y.size()/3 << std::endl;
	// 将std::vector<float>转换为arma::mat，因为mlpack使用Armadillo库作为其基本数据结构
    arma::mat points(3, Y.size() / 3);
    for (size_t i = 0; i < Y.size() / 3; ++i) {
        points(0, i) = Y[i * 3];
        points(1, i) = Y[i * 3 + 1];
        points(2, i) = Y[i * 3 + 2];
    }
    // 确保点云数据是3倍数长度
    if (Y.size() % 3 != 0) {
        std::cerr << "Input data size is not a multiple of 3!" << std::endl;
        return;
    }
    // DBSCAN算法
    mlpack::dbscan::DBSCAN<> dbscan(distance_tolerance, clustersize_tolerance);
    
    // 拟合模型
    arma::Row<size_t> assignments; // 所有点的簇标签
    dbscan.Cluster(points, assignments);
	
	// 移除噪声点标签后的簇标签集合
	std::set<size_t> labels_without_noise;

	for (size_t label : assignments) {
		if (label != std::numeric_limits<size_t>::max()) {
			labels_without_noise.insert(label);
		}
	}

    // 计算找到的簇数量
    size_t num_clusters = labels_without_noise.size();
	/*
	// 打印 unique_labels 中的所有元素
	for (size_t label : unique_labels) {
		std::cout << label << " ";
	}
	std::cout << std::endl;
	size_t noiseCount = 0;
	for (size_t i = 0; i < assignments.n_elem; ++i) {
		if (assignments[i] == std::numeric_limits<size_t>::max()) {
			noiseCount++;
		}
	}
	std::cout << "Number of noise points: " << noiseCount << std::endl;
	*/
    

    // 存储簇数量（包括噪声点作为一个簇）
	this->point_workspace["clusters"] = std::vector<float>{static_cast<float>(num_clusters)};
    
    std::cout << "DBSCAN found " << num_clusters << " clusters" << std::endl;
	// 如果只需要质心
    if (centroid_only) {
        std::vector<arma::colvec3> cluster_centroids(num_clusters);
        
        // 使用OpenMP并行计算质心
        #pragma omp parallel for num_threads(n_job) schedule(dynamic)
        for (int i = 0; i < static_cast<int>(num_clusters); ++i) {
            cluster_centroids[i] = CalculateCentroid(i, points, assignments);
        }

        // 转换回std::vector<float>以存储在point_workspace中
        std::vector<float> centroids_flat;
        for (const auto& centroid : cluster_centroids) {
            centroids_flat.push_back(centroid(0));
            centroids_flat.push_back(centroid(1));
            centroids_flat.push_back(centroid(2));
        }
        this->point_workspace[outputname] = centroids_flat;
    } else {
        // 如果需要所有点
        this->point_workspace[outputname] = Y; // Y是原始的点云数据
    }    

    // 输出转换后的点云大小信息
    std::cout << Y.size() / 3 << " grids were reduced to " << this->point_workspace[outputname].size() / 3 << " grid after this step" << std::endl;
    
    
}
// 定义函数IO_WriteXYZ，用于写XYZ文件
void VoxEM::IO_WriteXYZ(const std::string& point_inputname,
                 const std::string& file_outputname,
                 const std::string& atom_name,
                 const std::array<float, 3>& shifting) {
    
    std::vector<float>& Y = point_workspace[point_inputname];
	// 打印shifting数组
	/*
    std::cout << "Shifting: ["
              << shifting[0] << ", "
              << shifting[1] << ", "
              << shifting[2] << "]" << std::endl;
    for (size_t i = 0; i < Y.size(); i += 3) {
            std::cout << "Point " << (i / 3) << ": "
                      << Y[i] << ", " << Y[i + 1] << ", " << Y[i + 2] << std::endl;
    }
	*/
    for (size_t i = 0; i < Y.size(); i += 3) {
        Y[i] = Y[i] + shifting[0];
        Y[i + 1] = Y[i + 1] + shifting[1];
        Y[i + 2] = Y[i + 2] + shifting[2];
    }
	/*
    for (size_t i = 0; i < Y.size(); i += 3) {
            std::cout << "Point " << (i / 3) << ": "
                      << Y[i] << ", " << Y[i + 1] << ", " << Y[i + 2] << std::endl;
    }
	*/
    // 调用XYZ函数
    XYZ(Y, atom_name, file_outputname);
}
// 定义函数XYZ，用于将坐标点写入文件
void VoxEM::XYZ(const std::vector<float>& points, const std::string& atom_name, const std::string& filename) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "无法打开文件：" << filename << std::endl;
		return;
	}

	// 设置浮点数精度
	file << std::fixed << std::setprecision(5);
	// 确保 points 的大小是3的倍数
    if (points.size() % 3 != 0) {
        std::cerr << "点集的大小不是3的倍数，可能会丢失数据或写入无效数据。" << std::endl;
    }
	for (size_t i = 0; i < points.size(); i += 3) {
		file << points[i] << "        " << points[i + 1] << "        " << points[i + 2] << std::endl;
	}
}
void VoxEM::IO_Statistics(const std::string& voxel_inputname, const std::string& statistics_outputname, bool nonzerostatistics) {
    const std::vector<std::vector<std::vector<float>>>& vox = this->voxel_workspace[voxel_inputname];

    float RMS, dmin, dmax, dmean;

    if (nonzerostatistics) {
        std::vector<float> positive_values;
        for (const auto& plane : vox) {
            for (const auto& row : plane) {
                for (float value : row) {
                    if (value > 0.0) {
                        positive_values.push_back(value);
                    }
                }
            }
        }
        RMS = calculateRMS(positive_values);
        dmin = calculateMin(positive_values);
        dmax = calculateMax(positive_values);
        dmean = calculateMean(positive_values);
    } else {
        std::vector<float> flattened_vox;
        for (const auto& plane : vox) {
            for (const auto& row : plane) {
                flattened_vox.insert(flattened_vox.end(), row.begin(), row.end());
            }
        }
        RMS = calculateRMS(flattened_vox);
        dmin = calculateMin(flattened_vox);
        dmax = calculateMax(flattened_vox);
        dmean = calculateMean(flattened_vox);
    }

    // Update Dimension
    std::vector<int> dimension = {static_cast<int>(vox.size()), static_cast<int>(vox[0].size()), static_cast<int>(vox[0][0].size())};

    // 使用 std::unordered_map 保存多个键值对
    this->statistics_workspace[statistics_outputname] = {
        {"Rms", RMS},
        {"Min", dmin},
        {"Max", dmax},
        {"Mean", dmean},
        {"Dim", dimension}
    };
	float contour_level = dmean + 3*RMS;
	std::cout << "Rms: " << RMS << std::endl;
	std::cout << "Min: " << dmin << std::endl;
    std::cout << "Max: " << dmax << std::endl;
    std::cout << "Mean: " << dmean << std::endl;
    std::cout << "Dim: " << dimension << std::endl;
    std::cout << "contour_level：" << contour_level << std::endl;
	std::cout << "IO_Statistics finished" << std::endl;
}
void VoxEM::Voxel_Prune_RangeZero(float upperbound, float lowerbound, const std::string& inputname, const std::string& outputname) {
    std::vector<std::vector<std::vector<float>>>& vox = this->voxel_workspace[inputname];

    for (size_t i = 0; i < vox.size(); ++i) {
        for (size_t j = 0; j < vox[i].size(); ++j) {
            for (size_t k = 0; k < vox[i][j].size(); ++k) {
                if (lowerbound != -1.0 && vox[i][j][k] <= lowerbound) {
                    vox[i][j][k] = 0.0;
                }
                if (upperbound != -1.0 && vox[i][j][k] >= upperbound) {
                    vox[i][j][k] = 0.0;
                }
            }
        }
    }

    this->voxel_workspace[outputname] = vox;

    std::cout << "Voxel_Prune_RangeZero finished" << std::endl;
}
std::tuple<std::vector<std::array<float, 3>>, std::vector<std::array<float, 3>>, std::vector<std::array<float, 1>>> VoxEM::load_sample_points(const std::string& file_path, bool density = false) {
    std::vector<std::array<float, 3>> point_list;
    std::vector<std::array<float, 3>> vector_list;
    std::vector<std::array<float, 1>> density_list;

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::string line;
    float sample = 0.0;
    std::array<float, 3> origin;

    std::getline(file, line);
    sample = std::stof(line);
	std::getline(file, line);
	std::getline(file, line);
    std::getline(file, line);
    std::istringstream origin_ss(line);
    origin_ss >> origin[0] >> origin[1] >> origin[2];
	std::getline(file, line);
    int line_num = 0;
    while (std::getline(file, line)) { 
		line_num++;
        std::istringstream iss(line);
        if (line_num % 2 == 1) {
			std::string ignore;
            float x = 0.0, y = 0.0, z = 0.0;
            iss >>ignore>>x >> y >> z;
            point_list.push_back({{x * sample + origin[0], y * sample + origin[1], z * sample + origin[2]}});
        } else {
            float vx = 0.0, vy = 0.0, vz = 0.0, d = 0.0;
            iss >> vx >> vy >> vz >> d;
            vector_list.push_back({{vx, vy, vz}});
            density_list.push_back({{d}});
        }
		
    }

    if (density) {
        return std::make_tuple(point_list, vector_list, density_list);
    }
    return std::make_tuple(point_list, vector_list, std::vector<std::array<float, 1>>());
}
float VoxEM::calculateRMS(const std::vector<float>& values) {
    float mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
	//std::cout << "mean："<<mean << std::endl;
    float sum = 0.0;
    for (float value : values) {
        sum += (value - mean) * (value - mean);
    }
    return std::sqrt(sum / (values.size() - 1));  // 注意这里我们用的是n-1来计算无偏估计的样本标准差
}

float VoxEM::calculateMin(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0;  // Handle the case where there are no values
    }
    float min_value = values[0];
    for (float value : values) {
        if (value < min_value) {
            min_value = value;
        }
    }
    return min_value;
}
float VoxEM::calculateMax(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0;  // Handle the case where there are no values
    }
    float max_value = values[0];
    for (float value : values) {
        if (value > max_value) {
            max_value = value;
        }
    }
    return max_value;
}
float VoxEM::calculateMean(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0;  // Handle the case where there are no values
    }
    float mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return mean;
}
std::tuple<int, int, int> VoxEM::unpack_ints(const char* data, int offset) {
    int value1, value2, value3;
    std::tie(value1, value2, value3) = std::make_tuple(
        *reinterpret_cast<const int*>(&data[offset]),
        *reinterpret_cast<const int*>(&data[offset + 4]),
        *reinterpret_cast<const int*>(&data[offset + 8])
    );
    return std::make_tuple(value1, value2, value3);
}
int VoxEM::unpack_int(const char* data, int offset) {
    return *reinterpret_cast<const int*>(data + offset);
}
std::tuple<float, float, float> VoxEM::unpack_floats(const char* data, int offset) {
    float value1, value2, value3;
    std::tie(value1, value2, value3) = std::make_tuple(
        *reinterpret_cast<const float*>(&data[offset]),
        *reinterpret_cast<const float*>(&data[offset + 4]),
        *reinterpret_cast<const float*>(&data[offset + 8])
    );
    return std::make_tuple(value1, value2, value3);
}
std::vector<float> VoxEM::unpack_float_array(const char* data, int offset, int count) {
    std::vector<float> result(count);
    for (int i = 0; i < count; i++) {
        result[i] = *reinterpret_cast<const float*>(&data[offset + i * 4]);
    }
    return result;
}
std::vector<std::vector<std::vector<float>>> VoxEM::getVoxelWorkspace(const std::string& key) {
    return this->voxel_workspace[key];
}

std::unordered_map<std::string, std::any> VoxEM::getDescriptionWorkspace(const std::string& key) {
    return this->description_workspace[key];
}
void VoxEM::setPointWorkspaceValue(const std::string& key, const std::vector<std::array<float, 3>>& value) {
	// 将 value 转换为 std::vector<float>，并存储在 point_workspace 中的 key 中
	std::vector<float> flattened_values;
	for (const auto& point : value) {
		flattened_values.push_back(point[0]);
		flattened_values.push_back(point[1]);
		flattened_values.push_back(point[2]);
	}
		this->point_workspace[key] = flattened_values;
}
std::vector<float> VoxEM::getPointWorkspaceValue(const std::string& key) const {
    auto it = point_workspace.find(key);
    if (it != point_workspace.end()) {
        return it->second;
    } else {
        return {};  // 返回空的 vector<float>，表示找不到对应的键名
    }
}
/*
std::vector<std::vector<float>> VoxEM::transpose(const std::vector<std::array<float, 3>>& matrix) {
	if (matrix.empty()) {
        return {};
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

std::vector<std::array<float, 3>> VoxEM::convertToArrays(const std::vector<std::vector<float>>& input) {
    std::vector<std::array<float, 3>> output;
    output.reserve(input.size());
    for (const auto& row : input) {
        std::array<float, 3> arr;
        for (size_t i = 0; i < 3; ++i) {
            arr[i] = row[i];
        }
        output.push_back(arr);
    }
    return output;
}
*/
