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
#include <torch/torch.h>
#include <numeric>
#include <omp.h>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
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

    // Create tensors from the flat vectors
    torch::Tensor Q = torch::from_blob(vox_flat.data(), {D, H, W}).cuda();
    Q = Q.repeat({3, 1, 1, 1});
    
    // Assuming grid is a flat vector with every 3 elements representing one point (x, y, z)
    // and that the total number of points is equal to the product of D, H, and W.
    torch::Tensor X = torch::from_blob(grid.data(), {D, H, W, 3}).cuda();

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
	auto mgrid = torch::meshgrid({torch::arange(0, window, torch::kFloat32),
                              torch::arange(0, window, torch::kFloat32),
                              torch::arange(0, window, torch::kFloat32)},"xy"
                              );

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
    W_tensor = W_tensor.view({1, 1, window, window, window}).to(torch::kCUDA);

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
    
    C = C.cpu(); 
    std::cout << "C.to CPU Completed." << std::endl;

    
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
	//std::cout << "Y_diff_magnitude_init:" << Y_diff_magnitude <<std::endl;
    int i = 0;
    std::vector<float> Y_diff(Y.size(), 0.0f);
	try {
		while (i <= iteration && Y_diff_magnitude >= convergence) {
			float sum_of_squares = 0.0f;

			//#pragma omp parallel for reduction(+:sum_of_squares) schedule(dynamic)
			for (size_t j = 0; j < num_points; ++j) {
				std::array<int, 3> Y_indice;
				std::array<float, 3> Y_proposed;
				std::array<float, 3> inv_interval = {1.0f / interval[0], 1.0f / interval[1], 1.0f / interval[2]};

				for (int k = 0; k < 3; ++k) {
					//std::cout << "Y[j * 3 + k]:" << Y[j * 3 + k] <<std::endl;
					Y_indice[k] = static_cast<int>(std::round((Y[j * 3 + k] - interval[k]) * inv_interval[k])) + 1;
				}
				/*
				std::cout << "Y_indice: \n";
				for (const auto& idx : Y_indice) std::cout << idx << " ";
				std::cout << std::endl;
				*/
				for (int dim = 0; dim < 3; ++dim) {
					size_t index = (dim * num_cols * num_rows * num_slices) + (Y_indice[0] * num_rows * num_slices) + (Y_indice[1] * num_slices) + Y_indice[2];
					if (C_vec[index]<0){
						Y_proposed[dim]=0;
					}else if(C_vec[index]>1000){
						Y_proposed[dim]=0;
					}else{
						Y_proposed[dim] = C_vec[index];
					}
				}
				/*
				std::cout << "Y_proposed: ";
				for (const auto& val : Y_proposed) std::cout << val << " ";
				std::cout << std::endl;
				*/
				float sum_squares = std::inner_product(std::begin(Y_proposed), std::end(Y_proposed), std::begin(Y_proposed), 0.0f);
				bool is_below_threshold = sum_squares < lower_bound*lower_bound;

				for (int k = 0; k < 3; ++k) {
					float diff = is_below_threshold ? 0.0f : (Y[j * 3 + k] - Y_proposed[k]);
					Y_diff[j * 3 + k] = diff;
					sum_of_squares += diff * diff;
				}
				
			}
			/*
			std::cout << "Y_diff: \n";
			for (size_t j = 0; j < num_points; ++j) {
				for (size_t i = 0; i < 3; ++i) {
					float value = Y_diff[j * 3 + i];
					std::cout << value;
					if (i < 2) {
						std::cout << "\t"; // 在同一行的元素间加入制表符
					}
				}
				std::cout << std::endl; // 每行结束后换行
			}
			*/
			//std::cout << "Y_diff_magnitude_init:" << Y_diff_magnitude <<std::endl;
			Y_diff_magnitude = sum_of_squares / (3*num_points);
			//std::cout << "Y_diff_magnitude:" << Y_diff_magnitude <<std::endl;
			//#pragma omp parallel for
			/*
			std::cout << "Y1: \n";
			for (size_t j = 0; j < num_points; ++j) {
				for (size_t i = 0; i < 3; ++i) {
					float value = Y[j * 3 + i];
					std::cout << value;
					if (i < 2) {
						std::cout << "\t"; // 在同一行的元素间加入制表符
					}
				}
				std::cout << std::endl; // 每行结束后换行
			}
			*/
			for (size_t idx = 0; idx < Y.size(); ++idx) {
				Y[idx] -= Y_diff[idx] * step_size;
			}
			/*
			std::cout << "Y2: \n";
			for (size_t j = 0; j < num_points; ++j) {
				for (size_t i = 0; i < 3; ++i) {
					float value = Y[j * 3 + i];
					std::cout << value;
					if (i < 2) {
						std::cout << "\t"; // 在同一行的元素间加入制表符
					}
				}
				std::cout << std::endl; // 每行结束后换行
			}
			*/
			if (i % 50 == 0) {
				std::cout << "Meanshift: iteration " << i << " convergence " << Y_diff_magnitude << std::endl;
			}
			i++;
		}
	}catch (const std::exception& e) {
	std::cerr << "An exception occurred: " << e.what() << std::endl;
	}
	
    this->point_workspace[point_outputname] = Y;
    std::cout << "Meanshift Completed." << std::endl;
	
	/*
	for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 8327; ++j) {
            float value = Y[j * 3 + i];
            std::cout << value;
            if (j < 8327 - 1) {
                std::cout << "\t"; // 在同一行的元素间加入制表符
            }
        }
        std::cout << std::endl; // 每行结束后换行
    }
	*/
}
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
