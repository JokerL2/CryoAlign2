#include "Registration.h"
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
#include <numeric>
#include <omp.h>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <teaser/registration.h>
#include <filesystem>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <Eigen/Core>
#include <cnpy.h>
#include <utility>
#include <flann/flann.hpp>
#include <future>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>

// Constructor
using namespace std;

Registration::Registration() {
    // Initialize Workspaces
}
Eigen::Matrix4d Registration::Registration_given_feature(const std::string& data_dir,
                                              const std::string& source_dir,
                                              const std::string& target_dir,
                                              const std::string& source_sample_dir,
                                              const std::string& target_sample_dir,
											  const std::string& source_pdb,
											  const std::string& sup_pdb,
											  //const std::string& res_path,
                                              float VOXEL_SIZE,
											  float feature_radius,
                        const std::string& outnum,
                                              bool visualize,
                                              std::optional<Eigen::Matrix4d> T,
                                              bool one_stage) {

    // Load samples and ignore density if not needed
    std::vector<std::array<double, 3>> sample_A_points, sample_A_normals;
    std::vector<std::array<double, 3>> sample_B_points, sample_B_normals;
    std::tie(sample_A_points, sample_A_normals, std::ignore) = load_sample(source_sample_dir);
    std::tie(sample_B_points, sample_B_normals, std::ignore) = load_sample(target_sample_dir);

    // Create an empty point cloud for A and B
    open3d::geometry::PointCloud A_pcd;
    open3d::geometry::PointCloud B_pcd;

    // Fill points and normals for A
    for (const auto& point : sample_A_points) {
        A_pcd.points_.push_back(Eigen::Vector3d(point[0], point[1], point[2]));
    }
    for (const auto& normal : sample_A_normals) {
        A_pcd.normals_.push_back(Eigen::Vector3d(normal[0], normal[1], normal[2]));
    }

    // Fill points and normals for B
    for (const auto& point : sample_B_points) {
        B_pcd.points_.push_back(Eigen::Vector3d(point[0], point[1], point[2]));
    }
    for (const auto& normal : sample_B_normals) {
        B_pcd.normals_.push_back(Eigen::Vector3d(normal[0], normal[1], normal[2]));
    }

    // Load keypoints for A and B
    open3d::geometry::PointCloud A_key_pcd = load_xyz(source_dir);
    open3d::geometry::PointCloud B_key_pcd = load_xyz(target_dir);

    // in C++ Open3D we use the points() method to access the points
    const std::vector<Eigen::Vector3d>& A_keypoint = A_key_pcd.points_;
    const std::vector<Eigen::Vector3d>& B_keypoint = B_key_pcd.points_;

    // Directory for temporary files
    std::string temp_dir = data_dir + "/temp";
    if (!std::filesystem::exists(temp_dir)) {
        std::filesystem::create_directory(temp_dir);
    }
	std::cout << "Registration PointCloud3d sussessfully" << std::endl;
	Eigen::MatrixXd A_key_feats = cal_SHOT(sample_A_points, sample_A_normals, temp_dir, A_keypoint, VOXEL_SIZE * feature_radius);
	Eigen::MatrixXd B_key_feats = cal_SHOT(sample_B_points, sample_B_normals, temp_dir, B_keypoint, VOXEL_SIZE * feature_radius);
	std::cout << "A_key_feats rows: " << A_key_feats.rows() << ", cols: " << A_key_feats.cols() << std::endl;
	std::cout << "B_key_feats rows: " << B_key_feats.rows() << ", cols: " << B_key_feats.cols() << std::endl;
	/*
	// 直接打印 A_key_feats
    std::cout << "A_key_feats:" << std::endl;
    for (int i = 0; i < A_key_feats.rows(); ++i) {
        for (int j = 0; j < A_key_feats.cols(); ++j) {
            std::cout << A_key_feats(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // 直接打印 B_key_feats
    std::cout << "\nB_key_feats:" << std::endl;
    for (int i = 0; i < B_key_feats.rows(); ++i) {
        for (int j = 0; j < B_key_feats.cols(); ++j) {
            std::cout << B_key_feats(i, j) << " ";
        }
        std::cout << std::endl;
    }
	*/
	std::cout << "cal_SHOT sussessfully" << std::endl;
	// 寻找对应点
    auto [corrs_A, corrs_B] = find_correspondences(A_key_feats, B_key_feats);
	// 计算余弦距离
    auto cosine_SHOT_distances = compute_cosine_distances(A_key_feats, B_key_feats, corrs_A, corrs_B);
    double shot_score=calc_SHOT_score(cosine_SHOT_distances);
	/*
	// 打印 corrs_A
    std::cout << "corrs_A: ";
    for (const auto& idx : corrs_A) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    // 打印 corrs_B
    std::cout << "corrs_B: ";
    for (const auto& idx : corrs_B) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
	*/
    
	// 根据索引提取对应点的坐标
    Eigen::MatrixXd A_corr(3, corrs_A.size());
    Eigen::MatrixXd B_corr(3, corrs_B.size());
    for (size_t i = 0; i < corrs_A.size(); ++i) {
        A_corr.col(i) = A_keypoint[corrs_A[i]];
		B_corr.col(i) = B_keypoint[corrs_B[i]];
    }

    // 转置以匹配Python代码中的格式
    //A_corr.transposeInPlace();
    //B_corr.transposeInPlace();
	/*
	// 打印 A_corr
	std::cout << "A_corr:" << std::endl;
	std::cout << A_corr << std::endl;

	// 打印 B_corr
	std::cout << "B_corr:" << std::endl;
	std::cout << B_corr << std::endl;
	*/
	Eigen::Matrix4d T_icp;
	if (!T){
		float NOISE_BOUND = VOXEL_SIZE;
		auto solver_params = getTeaserSolverParams(NOISE_BOUND);
		// 打印 solver_params 的所有参数
		/*
		std::cout << "Solver Parameters:" << std::endl;
		std::cout << "cbar2: " << solver_params.cbar2 << std::endl;
		std::cout << "noise_bound: " << solver_params.noise_bound << std::endl;
		std::cout << "estimate_scaling: " << (solver_params.estimate_scaling ? "true" : "false") << std::endl;
		std::cout << "inlier_selection_mode: " << static_cast<int>(solver_params.inlier_selection_mode) << std::endl;
		std::cout << "rotation_tim_graph: " << static_cast<int>(solver_params.rotation_tim_graph) << std::endl;
		std::cout << "rotation_estimation_algorithm: " << static_cast<int>(solver_params.rotation_estimation_algorithm) << std::endl;
		std::cout << "rotation_gnc_factor: " << solver_params.rotation_gnc_factor << std::endl;
		std::cout << "rotation_max_iterations: " << solver_params.rotation_max_iterations << std::endl;
		std::cout << "rotation_cost_threshold: " << solver_params.rotation_cost_threshold << std::endl;
		*/
		teaser::RobustRegistrationSolver solver(solver_params);
		
		solver.solve(A_corr, B_corr);
		auto solution = solver.getSolution();
		auto R_teaser = solution.rotation;
		auto t_teaser = solution.translation;
		/*
		// 输出旋转矩阵
		std::cout << "Rotation matrix:" << std::endl;
		std::cout << solution.rotation << std::endl;

		// 输出平移向量
		std::cout << "Translation vector:" << std::endl;
		std::cout << solution.translation << std::endl;
		*/
		Eigen::Matrix4d T_teaser = Rt2T(R_teaser, t_teaser);
		Eigen::Matrix4d init_transformation = T_teaser;
		
		
		if (one_stage){
			if (visualize){
				draw_registration_result(A_pcd, B_pcd, init_transformation);
			}
			return init_transformation;
		}
		// 使用Open3D的ICP方法
		auto icp_result = open3d::pipelines::registration::RegistrationICP(
			A_pcd, B_pcd, 10.0, init_transformation,
			open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
			open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, 1000)
		);
		T_icp = icp_result.transformation_;
	}else{
		T_icp = T.value();
	}
	
	
	if (visualize){
		draw_registration_result(A_pcd, B_pcd, T_icp);	
	}

    // 计算分数
    
	/*
	std::string res_file = res_path+"/res.txt";
	size_t lastSlash = data_dir.rfind('/');
	std::string res_id = data_dir.substr(lastSlash + 1);
	// 打开文件，并追加内容
    std::ofstream out_file(res_file, std::ios::app | std::ios::ate);

    // 检查文件是否为空或者新的
    bool is_new_file = out_file.tellp() == 0;
    if (is_new_file) {
        // 如果是新文件，添加表头
        //out_file << "Resource ID\tScore\tRmsd\n";
		out_file << "Resource ID\tScore\n";
    }

    // 将res_id和score写入文件，使用制表符分隔
    //out_file << res_id << "\t" << score <<"\t"<< rmsd << std::endl;
	out_file << res_id << "\t" << score << std::endl;

    out_file.close();
	*/
	//std::cout << "res_id:" <<res_id<<";"<< "direct_alignment score:" << score<<";"<< "rmsd:"<< rmsd <<";"<<std::endl;
	double score = cal_score(A_pcd, B_pcd,shot_score, 10.0, T_icp);
	std::cout << "direct_alignment score:" << score<<";"<<std::endl;
	if (!source_pdb.empty() && !sup_pdb.empty()){
		Eigen::MatrixXd source_Pdb = getPointsFromPDB(source_pdb);
		Eigen::MatrixXd source_sup = getPointsFromPDB(sup_pdb);
		double rmsd = calRMSD(source_Pdb, source_sup, T_icp);
		std::cout << "RMSD between estiamted transformed PDB and ground truth:" << rmsd<<";"<<std::endl;
	}
	
	Eigen::Matrix4d T_icp_transposed = T_icp.transpose();
	double* T_icp_ptr = T_icp_transposed.data();

    // 使用cnpy保存矩阵为.npy文件
    cnpy::npy_save(data_dir +"/"+outnum+"_"+"RT.npy", T_icp_ptr, {4, 4});
	//std::filesystem::remove_all(data_dir);
	return T_icp;
}
double Registration::calc_SHOT_score(const std::vector<double>& cosine_distances) {
    if (cosine_distances.empty()) return 0.0;
    
    double total = 0.0;
    for (const auto& d : cosine_distances) {
        total += (1.0 - d);  // 将距离转换为相似度
    }
    return total / cosine_distances.size();  // 范围[0,1] 1=完美匹配
}
// 计算余弦距离的函数
std::vector<double> Registration::compute_cosine_distances(
    const Eigen::MatrixXd& feats_A,
    const Eigen::MatrixXd& feats_B,
    const std::vector<int>& corrs_A,
    const std::vector<int>& corrs_B) 
{
    std::vector<double> distances;
    distances.reserve(corrs_A.size());

    for (size_t i = 0; i < corrs_A.size(); ++i) {
        // 获取对应索引
        const int idx_A = corrs_A[i];
        const int idx_B = corrs_B[i];

        // 提取特征向量
        const Eigen::VectorXd vec_A = feats_A.row(idx_A);
        const Eigen::VectorXd vec_B = feats_B.row(idx_B);

        // 计算点积和模长
        const double dot_product = vec_A.dot(vec_B);
        const double norm_A = vec_A.norm();
        const double norm_B = vec_B.norm();

        // 处理零向量特殊情况
        if (norm_A < 1e-8 || norm_B < 1e-8) {
            distances.push_back(2.0); // 最大可能距离
            continue;
        }

        // 计算余弦距离
        const double cosine_sim = dot_product / (norm_A * norm_B);
        const double cosine_distance = 1.0 - cosine_sim;
        
        distances.push_back(cosine_distance);
    }

    return distances;
}


void Registration::extract_top_K(const std::string& record_dir, const std::string& record_T_dir, int K, const std::string& save_dir, const std::string& source_pdb_dir, const std::string& source_sup_dir) {
    // 读取分数
    std::vector<double> score_list;
    std::ifstream record_file(record_dir);
    std::string line;
    std::getline(record_file, line); // 跳过第一行
    while (std::getline(record_file, line)) {
        double score = std::stod(line.substr(line.find_last_of(" \t") + 1));
        score_list.push_back(score);
    }

    // 对分数和索引进行排序
    std::vector<std::pair<size_t, double>> score_index_pairs(score_list.size());
    for (size_t i = 0; i < score_list.size(); ++i) {
        score_index_pairs[i] = {i, score_list[i]};
    }
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
              [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                  return a.second > b.second;
              });

    // 获取前 K 个索引
    std::vector<size_t> top_k_indices(K);
    for (int i = 0; i < K; ++i) {
        top_k_indices[i] = score_index_pairs[i].first;
    }
	// 打印 top_k_indices 中的值
	std::cout << "Top K indices: ";
	for (size_t i = 0; i < top_k_indices.size(); ++i) {
		std::cout << top_k_indices[i];
		if (i < top_k_indices.size() - 1) {
			std::cout << ", "; // 在元素之间添加逗号和空格
		}
	}
	std::cout << std::endl;
	// 打开文件
    std::ifstream inputFile(record_T_dir);

    if (!inputFile.is_open()) {
        std::cerr << "Failed to open file: " << record_T_dir << std::endl;
    }

    std::vector<Eigen::Matrix4d> matrices;
    std::string line1;
    Eigen::Matrix4d mat;
    int row = 0;
	int matrixIndex = 0;


	while (std::getline(inputFile, line)) {
		if (line.empty()) {
			continue;
		}
		std::istringstream lineStream(line);
		std::string cell;
		int col = 0;

		while (std::getline(lineStream, cell, ',')) {
			double value = std::stod(cell);
			mat(row, col) = value;
			col++;
		}

		row++;
		if (row == 4) {
			// 检查当前矩阵的索引是否在top_k_indices中
            if (std::find(top_k_indices.begin(), top_k_indices.end(), matrixIndex) != top_k_indices.end()) {
                matrices.push_back(mat); // 只有在top_k_indices中的矩阵才被添加
            }
            matrixIndex++; // 更新矩阵索引
            row = 0; // 重置行计数器
		}
	}

    // 关闭文件
    inputFile.close();

    // 打开文件准备写入
    std::ofstream save_file(save_dir);
    if (!save_file.is_open()) {
        std::cerr << "无法打开保存文件：" << save_dir << std::endl;
    }
	if (!source_pdb_dir.empty() && !source_sup_dir.empty()){
		// 写入表头
		save_file << "score\ttransformation matrix\tRMSD\n";

		// 遍历 matrices
		for (size_t i = 0; i < matrices.size(); ++i) {
			const Eigen::Matrix4d& T = matrices[i]; // 直接引用matrices中的矩阵
			double score = score_list[top_k_indices[i]]; // 使用top_k_indices索引score_list
			double rmsd = cal_pdb_RMSD(source_pdb_dir, source_sup_dir, T); // 计算 RMSD

			// 写入分数
			save_file << std::fixed << std::setprecision(2) << score << "\t";

			// 写入变换矩阵
			for (int col = 0; col < 4; ++col) {
				for (int row = 0; row < 4; ++row) {
					save_file << std::fixed << std::setprecision(4) << T.transpose()(row, col); 
					if (row < 3) save_file << ",";
				}
				if (col < 3) save_file << ";";
			}
			//save_file << "\t"  << "\n";

			// 写入 RMSD 值
			save_file << "\t" << std::fixed << std::setprecision(2) << rmsd << "\n";
		}
	}else{
		// 写入表头
		save_file << "score\ttransformation matrix\n";

		// 遍历 matrices
		for (size_t i = 0; i < matrices.size(); ++i) {
			const Eigen::Matrix4d& T = matrices[i]; // 直接引用matrices中的矩阵
			double score = score_list[top_k_indices[i]]; // 使用top_k_indices索引score_list
			//double rmsd = cal_pdb_RMSD(source_pdb_dir, source_sup_dir, T); // 计算 RMSD

			// 写入分数
			save_file << std::fixed << std::setprecision(2) << score << "\t";

			// 写入变换矩阵
			for (int col = 0; col < 4; ++col) {
				for (int row = 0; row < 4; ++row) {
					save_file << std::fixed << std::setprecision(4) << T.transpose()(row, col); 
					if (row < 3) save_file << ",";
				}
				if (col < 3) save_file << ";";
			}
			save_file << "\t"  << "\n";

			// 写入 RMSD 值
			//save_file << "\t" << std::fixed << std::setprecision(2) << rmsd << "\n";
		}
	}

    // 完成写入
    save_file.close();
    std::cout << "数据已成功写入文件：" << save_dir << std::endl;
	
    
}


double Registration::cal_pdb_RMSD(const std::string& source_pdb_dir,const std::string& source_sup_dir,Eigen::Matrix4d T){
	Eigen::MatrixXd source_pdb = getPointsFromPDB(source_pdb_dir);
    Eigen::MatrixXd source_sup = getPointsFromPDB(source_sup_dir);
	std::cout << "Read pdbfile success! \n" << std::endl;
	double rmsd = calRMSD(source_pdb, source_sup, T);
	return rmsd;
}
void Registration::Registration_mask_list(const std::string& data_dir,
                                              const std::string& source_dir,
                                              const std::string& target_dir,
                                              const std::string& source_sample_dir,
                                              const std::string& target_sample_dir,
                                              float VOXEL_SIZE,
											  float feature_radius,
                         const std::string& outnum,
                                              bool store_partial) {

    // Load samples and ignore density if not needed
    std::vector<std::array<double, 3>> sample_A_points, sample_A_normals;
    std::vector<std::array<double, 3>> sample_B_points, sample_B_normals;
    std::tie(sample_A_points, sample_A_normals, std::ignore) = load_sample(source_sample_dir);
    std::tie(sample_B_points, sample_B_normals, std::ignore) = load_sample(target_sample_dir);

    // Create an empty point cloud for A and B
    open3d::geometry::PointCloud A_pcd;
    open3d::geometry::PointCloud B_pcd;

    // Fill points and normals for A
    for (const auto& point : sample_A_points) {
        A_pcd.points_.push_back(Eigen::Vector3d(point[0], point[1], point[2]));
    }
    for (const auto& normal : sample_A_normals) {
        A_pcd.normals_.push_back(Eigen::Vector3d(normal[0], normal[1], normal[2]));
    }

    // Fill points and normals for B
    for (const auto& point : sample_B_points) {
        B_pcd.points_.push_back(Eigen::Vector3d(point[0], point[1], point[2]));
    }
    for (const auto& normal : sample_B_normals) {
        B_pcd.normals_.push_back(Eigen::Vector3d(normal[0], normal[1], normal[2]));
    }

    // Load keypoints for A and B
    open3d::geometry::PointCloud A_key_pcd = load_xyz(source_dir);
    open3d::geometry::PointCloud B_key_pcd = load_xyz(target_dir);

    // in C++ Open3D we use the points() method to access the points
    const std::vector<Eigen::Vector3d>& A_keypoint = A_key_pcd.points_;
    const std::vector<Eigen::Vector3d>& B_keypoint = B_key_pcd.points_;

    // Directory for temporary files
    std::string temp_dir = data_dir + "/temp";
    if (!std::filesystem::exists(temp_dir)) {
        std::filesystem::create_directory(temp_dir);
    }
	std::cout << "Registration PointCloud3d sussessfully" << std::endl;
	Eigen::MatrixXd A_key_feats = cal_SHOT(sample_A_points, sample_A_normals, temp_dir, A_keypoint, VOXEL_SIZE * feature_radius);
	Eigen::MatrixXd B_key_feats = cal_SHOT(sample_B_points, sample_B_normals, temp_dir, B_keypoint, VOXEL_SIZE * feature_radius);
	std::cout << "cal_SHOT sussessfully" << std::endl;
	Eigen::Vector3d A_min_bound = A_pcd.GetMinBound();
	Eigen::Vector3d A_max_bound = A_pcd.GetMaxBound();
	Eigen::Vector3d A_dist_cor = A_max_bound - A_min_bound;
	double A_mean_dis = A_dist_cor.sum() / A_dist_cor.size(); 

	Eigen::Vector3d B_min_bound = B_pcd.GetMinBound();
	Eigen::Vector3d B_max_bound = B_pcd.GetMaxBound();
	Eigen::Vector3d B_dist_cor = B_max_bound - B_min_bound;
	double B_mean_dis = B_dist_cor.sum() / B_dist_cor.size(); 
	
	if (std::abs(A_mean_dis - B_mean_dis) > 0) {
		double radius = *std::max_element(A_dist_cor.data(), A_dist_cor.data() + A_dist_cor.size()) * 1.1 / 2;
		Eigen::Vector3d center = B_min_bound + Eigen::Vector3d(radius / 10, radius / 10, radius / 10);
		Eigen::Vector3d terminal = B_max_bound;
		int step = static_cast<int>(std::floor(radius / 2)); // translation interval of mask

		double max_correspondence_dist = 10.0; // 未在Python代码中使用
		//std::string data_dir = "./emd_3661_6647"; // 修改为你的数据目录
		std::string record_dir = data_dir + "/"+outnum+"_record_cpp.txt";
		std::string record_T_dir = data_dir + "/"+outnum+"_record_T_cpp.npy";
		std::string record_T_dir_csv = data_dir + "/"+outnum+"_record_T_cpp.csv";
		std::vector<Eigen::Matrix4d> record_T; // 用于存储变换矩阵的向量

		std::ofstream f(record_dir, std::ios::out);
		if (!f.is_open()) {
			std::cerr << "Cannot open file " << record_dir << std::endl;
			return; // 或者处理错误
		}

		f << "t_x\tt_y\tt_z\tscore\n";
		//schedule(dynamic) schedule(static,3)
		omp_set_num_threads(4);
		// 预先分配 record_T 的大小
		int total_iterations = ((terminal[0] / step) + 1) * ((terminal[1] / step) + 1) * ((terminal[2] / step) + 1);
		record_T.reserve(total_iterations);
		#pragma omp parallel for schedule(static,4) collapse(3)
		for (int i = 0; i <= static_cast<int>(terminal[0]); i += step) {
			for (int j = 0; j <= static_cast<int>(terminal[1]); j += step) {
				for (int k = 0; k <= static_cast<int>(terminal[2]); k += step) {
					int thread_id = omp_get_thread_num(); // 获取当前线程ID
					double start_time = omp_get_wtime(); // 获取开始时间
					Eigen::Vector3d temp_center = center + Eigen::Vector3d(i, j, k);

					// 假设 this->mask 已经初始化
					this->mask["name"] = "sphere";
					this->mask["center"] = temp_center;
					this->mask["radius"] = radius;

					auto [final_T, score] = Registration_mask(temp_dir, A_pcd, B_pcd, A_key_pcd, B_key_pcd, A_key_feats, B_key_feats, this->mask, max_correspondence_dist, VOXEL_SIZE, store_partial);
					if (final_T) {
						#pragma omp critical
						{
							f << std::fixed << std::setprecision(2) << i << "\t" << j << "\t" << k << "\t" << score << std::endl;
							record_T.push_back(*final_T);
						}
					} else {
						#pragma omp critical
						{
							f << std::fixed << std::setprecision(2) << i << "\t" << j << "\t" << k << "\t" << "0.00" << std::endl;
							record_T.push_back(Eigen::Matrix4d::Identity());
						}
					}

					double end_time = omp_get_wtime(); // 获取结束时间

					std::cout << "线程 " << thread_id << " 开始于 " << start_time
							  << " 结束于 " << end_time << "，耗时 " << end_time - start_time
							  << "秒。处理的单元: i=" << i << ", j=" << j << ", k=" << k << std::endl;
				}
			}
		}
		f.close();
		std::vector<double> data;
		for (const auto& mat : record_T) {
			// 转置矩阵
			Eigen::Matrix4d transposed = mat.transpose();
			
			// 将转置后的矩阵数据添加到 data 数组中
			for (int i = 0; i < transposed.size(); ++i) {
				data.push_back(*(transposed.data() + i));
			}
		}
		// 计算矩阵的数量
		size_t num_matrices = record_T.size();
		// 保存为 .npy 文件
		cnpy::npy_save(record_T_dir, data.data(), {num_matrices, 4, 4}, "w");
		
		// 打开文件
		std::ofstream outputFile(record_T_dir_csv);

		if (!outputFile.is_open()) {
			std::cerr << "Failed to open file: " << record_T_dir_csv << std::endl;
			return;
		}

		// 遍历每个矩阵
		for (const auto& mat : record_T) {
			// 将矩阵数据写入CSV
			for (int i = 0; i < mat.rows(); ++i) {
				for (int j = 0; j < mat.cols(); ++j) {
					outputFile << mat(i, j);
					if (j < mat.cols() - 1) outputFile << ","; // 不是最后一列，添加逗号
				}
				outputFile << "\n"; // 完成一行的写入
			}
			// 如果你想在每个矩阵之间添加空行，取消注释下面的代码
			// outputFile << "\n";
		}
		outputFile.close();
	}else{
		std::cout << "please exchange the source map and target one" << std::endl;
	}
	std::cout << "masklist success!" << std::endl;
	//std::filesystem::remove_all(temp_dir);
	
}
/*
void Registration::Registration_mask_list(const std::string& data_dir,
                                              const std::string& source_dir,
                                              const std::string& target_dir,
                                              const std::string& source_sample_dir,
                                              const std::string& target_sample_dir,
                                              float VOXEL_SIZE,
                                              float feature_radius,
                                              const std::string& outnum,
                                              bool store_partial) {
    // MPI初始化
    MPI_Init(nullptr, nullptr);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 数据广播结构体
    struct BroadcastParams {
        Eigen::Vector3d center, terminal;
        double radius, step, max_correspondence_dist;
        int total_i, total_j, total_k;
        bool valid_processing;
        Eigen::MatrixXd A_key_feats, B_key_feats;
    } bcast_params;

    // 主进程加载数据
    if (world_rank == 0) {
        // 加载样本数据
        std::vector<std::array<double, 3>> sample_A_points, sample_A_normals;
        std::vector<std::array<double, 3>> sample_B_points, sample_B_normals;
        std::tie(sample_A_points, sample_A_normals, std::ignore) = load_sample(source_sample_dir);
        std::tie(sample_B_points, sample_B_normals, std::ignore) = load_sample(target_sample_dir);

        // 创建并填充点云A
        open3d::geometry::PointCloud A_pcd;
        for (const auto& point : sample_A_points) {
            A_pcd.points_.push_back(Eigen::Vector3d(point[0], point[1], point[2]));
        }
        for (const auto& normal : sample_A_normals) {
            A_pcd.normals_.push_back(Eigen::Vector3d(normal[0], normal[1], normal[2]));
        }

        // 创建并填充点云B
        open3d::geometry::PointCloud B_pcd;
        for (const auto& point : sample_B_points) {
            B_pcd.points_.push_back(Eigen::Vector3d(point[0], point[1], point[2]));
        }
        for (const auto& normal : sample_B_normals) {
            B_pcd.normals_.push_back(Eigen::Vector3d(normal[0], normal[1], normal[2]));
        }

        // 加载关键点
        open3d::geometry::PointCloud A_key_pcd = load_xyz(source_dir);
        open3d::geometry::PointCloud B_key_pcd = load_xyz(target_dir);

        // 创建临时目录
        std::string temp_dir = data_dir + "/temp";
        if (!std::filesystem::exists(temp_dir)) {
            std::filesystem::create_directory(temp_dir);
        }

        // 计算SHOT特征
        const auto& A_keypoint = A_key_pcd.points_;
        const auto& B_keypoint = B_key_pcd.points_;
        bcast_params.A_key_feats = cal_SHOT(sample_A_points, sample_A_normals, temp_dir, A_keypoint, VOXEL_SIZE * feature_radius);
        bcast_params.B_key_feats = cal_SHOT(sample_B_points, sample_B_normals, temp_dir, B_keypoint, VOXEL_SIZE * feature_radius);

        // 计算边界
        Eigen::Vector3d A_min_bound = A_pcd.GetMinBound();
        Eigen::Vector3d A_max_bound = A_pcd.GetMaxBound();
        Eigen::Vector3d B_min_bound = B_pcd.GetMinBound();
        Eigen::Vector3d B_max_bound = B_pcd.GetMaxBound();

        // 有效性检查
        double A_mean = (A_max_bound - A_min_bound).mean();
        double B_mean = (B_max_bound - B_min_bound).mean();
        if (std::abs(A_mean - B_mean) > 1e-6) {
            bcast_params.valid_processing = true;
            Eigen::Vector3d A_dist = A_max_bound - A_min_bound;
            bcast_params.radius = *std::max_element(A_dist.data(), A_dist.data()+3) * 1.1 / 2;
            bcast_params.center = B_min_bound + Eigen::Vector3d::Constant(bcast_params.radius/10);
            bcast_params.terminal = B_max_bound;
            bcast_params.step = static_cast<int>(std::floor(bcast_params.radius / 2));
            bcast_params.max_correspondence_dist = 10.0;

            // 计算迭代次数
            bcast_params.total_i = static_cast<int>(bcast_params.terminal[0]/bcast_params.step) + 1;
            bcast_params.total_j = static_cast<int>(bcast_params.terminal[1]/bcast_params.step) + 1;
            bcast_params.total_k = static_cast<int>(bcast_params.terminal[2]/bcast_params.step) + 1;
        } else {
            bcast_params.valid_processing = false;
        }
    }

    // 广播基本参数
    MPI_Bcast(&bcast_params.valid_processing, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bcast_params.valid_processing) {
        if (world_rank == 0) {
            std::cout << "please exchange the source map and target one" << std::endl;
        }
        MPI_Finalize();
        return;
    }

    // 广播几何参数
    if (world_rank != 0) {
        bcast_params.A_key_feats.resize(0,0);
        bcast_params.B_key_feats.resize(0,0);
    }
    
    // 自定义广播函数
    auto broadcast_matrix = [](Eigen::MatrixXd& mat, int root) {
        int rows, cols;
        if (MPI_Comm_rank(MPI_COMM_WORLD, &root) == root) {
            rows = mat.rows();
            cols = mat.cols();
        }
        MPI_Bcast(&rows, 1, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (world_rank != root) mat.resize(rows, cols);
        MPI_Bcast(mat.data(), rows*cols, MPI_DOUBLE, root, MPI_COMM_WORLD);
    };

    // 广播特征矩阵
    if (world_rank == 0) {
        MPI_Bcast(&bcast_params.center, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcast_params.terminal, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcast_params.radius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcast_params.step, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcast_params.max_correspondence_dist, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcast_params.total_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcast_params.total_j, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&bcast_params.total_k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    broadcast_matrix(bcast_params.A_key_feats, 0);
    broadcast_matrix(bcast_params.B_key_feats, 0);

    // 工作划分（按i维度）
    int chunk_size = bcast_params.total_i / world_size;
    int remainder = bcast_params.total_i % world_size;
    int start_i = world_rank * chunk_size + (world_rank < remainder ? world_rank : remainder);
    int end_i = start_i + chunk_size + (world_rank < remainder ? 1 : 0);
    start_i *= bcast_params.step;
    end_i = std::min(end_i * bcast_params.step, static_cast<int>(bcast_params.terminal[0]));

    // 本地存储
    std::vector<Eigen::Matrix4d> local_T;
    std::vector<std::tuple<int, int, int, double>> local_records;

    // OpenMP并行区域
    #pragma omp parallel
    {
        // 线程本地变量
        std::vector<Eigen::Matrix4d> thread_T;
        std::vector<std::tuple<int, int, int, double>> thread_records;
        auto local_mask = this->mask;  // 线程独立的mask副本

        #pragma omp for collapse(3) schedule(dynamic)
        for (int i = start_i; i <= end_i; i += bcast_params.step) {
            for (int j = 0; j <= static_cast<int>(bcast_params.terminal[1]); j += bcast_params.step) {
                for (int k = 0; k <= static_cast<int>(bcast_params.terminal[2]); k += bcast_params.step) {
                    // 设置mask参数
                    Eigen::Vector3d temp_center = bcast_params.center + Eigen::Vector3d(i, j, k);
                    local_mask["name"] = "sphere";
                    local_mask["center"] = temp_center;
                    local_mask["radius"] = bcast_params.radius;

                    // 执行配准
                    auto [final_T, score] = Registration_mask(
                        data_dir + "/temp_" + std::to_string(world_rank),
                        A_pcd, B_pcd, 
                        A_key_pcd, B_key_pcd,
                        bcast_params.A_key_feats, bcast_params.B_key_feats,
                        local_mask,
                        bcast_params.max_correspondence_dist,
                        VOXEL_SIZE,
                        store_partial
                    );

                    // 记录结果
                    thread_records.emplace_back(i, j, k, score);
                    thread_T.push_back(final_T ? *final_T : Eigen::Matrix4d::Identity());
                }
            }
        }

        // 合并线程结果
        #pragma omp critical
        {
            local_records.insert(local_records.end(), thread_records.begin(), thread_records.end());
            local_T.insert(local_T.end(), thread_T.begin(), thread_T.end());
        }
    }

    // 结果收集
    if (world_rank == 0) {
        // 主进程写入文件
        std::string record_dir = data_dir + "/"+outnum+"_record_cpp.txt";
        std::ofstream f(record_dir);
        f << "t_x\tt_y\tt_z\tscore\n";
        for (const auto& [i,j,k,score] : local_records) {
            f << i << "\t" << j << "\t" << k << "\t" << score << "\n";
        }

        // 接收其他进程结果
        for (int src = 1; src < world_size; ++src) {
            // 接收元数据
            int recv_count;
            MPI_Recv(&recv_count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 接收坐标和分数
            std::vector<int> coords(recv_count * 3);
            std::vector<double> scores(recv_count);
            MPI_Recv(coords.data(), recv_count*3, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(scores.data(), recv_count, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 写入文件
            for (int idx = 0; idx < recv_count; ++idx) {
                f << coords[idx*3] << "\t" << coords[idx*3+1] << "\t" 
                  << coords[idx*3+2] << "\t" << scores[idx] << "\n";
            }

            // 接收变换矩阵
            int matrix_count;
            MPI_Recv(&matrix_count, 1, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<double> matrix_data(matrix_count * 16);
            MPI_Recv(matrix_data.data(), matrix_count*16, MPI_DOUBLE, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // 转换为Matrix4d
            for (int i = 0; i < matrix_count; ++i) {
                Eigen::Matrix4d mat;
                std::memcpy(mat.data(), &matrix_data[i*16], 16*sizeof(double));
                local_T.push_back(mat);
            }
        }
        f.close();

        // 保存矩阵文件
        std::vector<double> all_T_data;
        for (const auto& mat : local_T) {
            Eigen::Matrix4d transposed = mat.transpose();
            all_T_data.insert(all_T_data.end(), transposed.data(), transposed.data() + 16);
        }
        cnpy::npy_save(data_dir + "/"+outnum+"_record_T_cpp.npy",
                      all_T_data.data(),
                      {all_T_data.size()/16, 4, 4},
                      "w");
    } else {
        // 从进程发送数据
        int send_count = local_records.size();
        MPI_Send(&send_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // 打包坐标和分数
        std::vector<int> coords(send_count * 3);
        std::vector<double> scores(send_count);
        for (size_t i = 0; i < local_records.size(); ++i) {
            auto [x,y,z,s] = local_records[i];
            coords[i*3] = x;
            coords[i*3+1] = y;
            coords[i*3+2] = z;
            scores[i] = s;
        }
        MPI_Send(coords.data(), send_count*3, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(scores.data(), send_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        // 发送变换矩阵
        int matrix_count = local_T.size();
        MPI_Send(&matrix_count, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        std::vector<double> matrix_data(matrix_count * 16);
        for (size_t i = 0; i < local_T.size(); ++i) {
            Eigen::Matrix4d transposed = local_T[i].transpose();
            std::memcpy(&matrix_data[i*16], transposed.data(), 16*sizeof(double));
        }
        MPI_Send(matrix_data.data(), matrix_count*16, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    if (world_rank == 0) {
        std::cout << "masklist success!" << std::endl;
    }
}
*/
std::tuple<std::optional<Eigen::Matrix4d>, double> Registration::Registration_mask(
							const std::string& data_dir,
							const open3d::geometry::PointCloud& A_pcd,
							const open3d::geometry::PointCloud& B_pcd,
							const open3d::geometry::PointCloud& A_key_pcd,
							const open3d::geometry::PointCloud& B_key_pcd,
							const Eigen::MatrixXd& A_key_feats, // 请根据实际特征的类型替换 Eigen::MatrixXd
							const Eigen::MatrixXd& B_key_feats,
							const std::unordered_map<std::string,std::variant<std::string, Eigen::Vector3d, double>>& mask,
							float max_correspondence_dist,
							float VOXEL_SIZE,
							bool store_partial) {
	//const std::vector<Eigen::Vector3d>& A_points = A_pcd.points_;
    const std::vector<Eigen::Vector3d>& B_points = B_pcd.points_;
	const std::vector<Eigen::Vector3d>& A_keypoint = A_key_pcd.points_;
    const std::vector<Eigen::Vector3d>& B_keypoint = B_key_pcd.points_;
	open3d::geometry::PointCloud mask_B_pcd;
	std::vector<size_t> mask_indices;
	open3d::geometry::PointCloud mask_B_key_pcd;
	std::vector<size_t> mask_key_indices;
	try {
		const std::string& name = std::get<std::string>(this->mask.at("name"));
		if (name == "sphere") {
			Eigen::Vector3d center = std::get<Eigen::Vector3d>(mask.at("center"));
			std::cout << "center: " << center.transpose() << std::endl;
			double radius = std::get<double>(mask.at("radius"));
			std::tie( mask_B_pcd,  mask_indices) = MaskSpherePoints(B_pcd, center, radius);
			std::tie( mask_B_key_pcd,  mask_key_indices) = MaskSpherePoints(B_key_pcd, center, radius);
		} else{
			// 如果掩码不是球形，则直接使用原始点云
			mask_B_pcd = B_pcd;
			mask_B_key_pcd = B_key_pcd;
			std::vector<size_t> mask_indices(B_pcd.points_.size());
			std::iota(mask_indices.begin(), mask_indices.end(), 0);
			std::vector<size_t> mask_key_indices(B_key_pcd.points_.size());
			std::iota(mask_key_indices.begin(), mask_key_indices.end(), 0);
		}
	}catch (const std::bad_any_cast& e) {
    std::cerr << "提取 mask 中的值时出错: " << e.what() << std::endl;
    // 适当的错误处理
	}

	if (mask_indices.size() <= 100||mask_indices.size() <=10) {
		std::cerr << "Mask content is small" << std::endl;
		return std::make_tuple(std::nullopt, 0.0);
	}
		
	std::cout << "sphere sussessfully" << std::endl;
	Eigen::MatrixXd B_mask_key_feats(mask_key_indices.size(), B_key_feats.cols());
	for (size_t i = 0; i < mask_key_indices.size(); ++i) {
		B_mask_key_feats.row(i) = B_key_feats.row(mask_key_indices[i]);
	}
	const std::vector<Eigen::Vector3d>& B_mask_key_points = mask_B_key_pcd.points_;
	auto [corrs_A, corrs_B] = find_correspondences(A_key_feats, B_mask_key_feats);
	auto cosine_SHOT_distances = compute_cosine_distances(A_key_feats, B_key_feats, corrs_A, corrs_B);
    double shot_score=calc_SHOT_score(cosine_SHOT_distances);
	// 根据索引提取对应点的坐标
    Eigen::MatrixXd A_corr(3, corrs_A.size());
    Eigen::MatrixXd B_corr(3, corrs_B.size());
    for (size_t i = 0; i < corrs_A.size(); ++i) {
        A_corr.col(i) = A_keypoint[corrs_A[i]];
		B_corr.col(i) = B_mask_key_points[corrs_B[i]];
    }
	float NOISE_BOUND = VOXEL_SIZE;
	auto solver_params = getTeaserSolverParams(NOISE_BOUND);
	teaser::RobustRegistrationSolver solver(solver_params);
	solver.solve(A_corr, B_corr);
	auto solution = solver.getSolution();
	auto R_teaser = solution.rotation;
	auto t_teaser = solution.translation;
	Eigen::Matrix4d T_teaser = Rt2T(R_teaser, t_teaser);
	Eigen::Matrix4d init_transformation = T_teaser;
	open3d::geometry::PointCloud source = A_pcd;
    open3d::geometry::PointCloud target;
    if(store_partial){
        target = mask_B_pcd;
    } else {
        target = B_pcd;
    }

    // 使用Open3D的ICP方法
    auto icp_result = open3d::pipelines::registration::RegistrationICP(
        source, target, max_correspondence_dist, init_transformation,
        open3d::pipelines::registration::TransformationEstimationPointToPoint(false),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, 1000)
    );

    Eigen::Matrix4d final_T = icp_result.transformation_;

    // 计算分数
    double score = cal_score(A_pcd, B_pcd,shot_score, max_correspondence_dist, final_T);

    return std::make_tuple(final_T, score); 
	
	
}

double Registration::cal_score(
    const open3d::geometry::PointCloud& A_pcd,
    const open3d::geometry::PointCloud& B_pcd,
	double shot_score,
    double max_correspondence_dist,
    const Eigen::Matrix4d& final_T) {

    // 使用Open3D执行评估
    auto eval_metric = open3d::pipelines::registration::EvaluateRegistration(
        A_pcd, B_pcd, max_correspondence_dist, final_T);

    // 检查对应点集的大小
    if (eval_metric.correspondence_set_.size() < A_pcd.points_.size() * 0.1) {
        std::cout << "correspondence_set is small" << std::endl;
        return 0.0;
    }

    // 变换A_pcd
    auto A_transformed = A_pcd;
    A_transformed.Transform(final_T);

    // 初始化改进评分指标
    double cosin_dist_sum = 0.0;
    double dist_sum = 0.0;
    double density_score = 0.0;
    double shot_similarity_score = 0.0;
    int valid_cosin_count = 0;

    // 计算密度
    auto A_kdtree = open3d::geometry::KDTreeFlann(A_transformed);
    auto B_kdtree = open3d::geometry::KDTreeFlann(B_pcd);

    for (const auto& corr : eval_metric.correspondence_set_) {
        // 计算法线之间的余弦距离
        double cosin_dist = A_transformed.normals_[corr[0]].dot(B_pcd.normals_[corr[1]]);
        if (cosin_dist >= 0.6) {
            cosin_dist_sum += cosin_dist;
            valid_cosin_count++;
        }

        // 计算对应点之间的欧氏距离
        double dist = (A_transformed.points_[corr[0]] - B_pcd.points_[corr[1]]).norm();
        dist_sum += exp(-dist);

        // 计算密度一致性
        std::vector<int> A_indices, B_indices;
        std::vector<double> A_distances, B_distances;
        A_kdtree.SearchRadius(A_transformed.points_[corr[0]], max_correspondence_dist, A_indices, A_distances);
        B_kdtree.SearchRadius(B_pcd.points_[corr[1]], max_correspondence_dist, B_indices, B_distances);

        double density_ratio = static_cast<double>(A_indices.size()) / static_cast<double>(B_indices.size());
        density_score += exp(-fabs(density_ratio - 1.0));

    }
	
    // 计算加权综合评分
    double normal_similarity_score = static_cast<double>(valid_cosin_count) / eval_metric.correspondence_set_.size();
    double distance_score = dist_sum / eval_metric.correspondence_set_.size();
    density_score /= eval_metric.correspondence_set_.size();
    double score = 0.3 * normal_similarity_score + 
                   0.1 * distance_score + 
                   0.3 * density_score + 
                   0.3 * shot_score;

    return score;
}



/*double Registration::cal_score(const open3d::geometry::PointCloud& A_pcd,
                 const open3d::geometry::PointCloud& B_pcd,
                 double max_correspondence_dist,
                 const Eigen::Matrix4d& final_T) {

    // 使用Open3D执行评估
    auto eval_metric = open3d::pipelines::registration::EvaluateRegistration(
        A_pcd, B_pcd, max_correspondence_dist, final_T);

    // 检查对应点集的大小
    if (eval_metric.correspondence_set_.size() < A_pcd.points_.size() * 0.1) {
        std::cout << "correspondence_set is small" << std::endl;
        return 0.0;
    }

    // 变换A_pcd
    auto A_transformed = A_pcd;
    A_transformed.Transform(final_T);

    // 计算余弦距离的总和和满足条件的数量
    double cosin_dist_sum = 0.0;
    int valid_cosin_count = 0;
    for (const auto& corr : eval_metric.correspondence_set_) {
        // 计算法线之间的余弦距离
        double cosin_dist = A_transformed.normals_[corr[0]].dot(B_pcd.normals_[corr[1]]);
        if (cosin_dist >= 0.6) {
            cosin_dist_sum += cosin_dist;
            valid_cosin_count++;
        }
    }
    double score = static_cast<double>(valid_cosin_count) / eval_metric.correspondence_set_.size();

    return score;
}
*/

std::pair<open3d::geometry::PointCloud, std::vector<size_t>> Registration::MaskSpherePoints(const open3d::geometry::PointCloud& pcd,
                 const Eigen::Vector3d& center,
                 double radius) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < pcd.points_.size(); ++i) {
        if ((pcd.points_[i] - center).norm() <= radius) {
            indices.push_back(i);
        }
    }
    open3d::geometry::PointCloud masked_pcd = *(pcd.SelectByIndex(indices));
    return std::make_pair(masked_pcd, indices);
}

double Registration::calRMSD(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target, const std::optional<Eigen::Matrix4d>& transformation) {
    int atom_num = source.rows();
    std::cout << "Number of atoms: " << atom_num << std::endl;

    Eigen::MatrixXd transformed_source;

    if (transformation) {
        //std::cout << "Transformation:\n" << transformation.value() << std::endl;

        // 创建齐次坐标
        Eigen::MatrixXd homo_source = Eigen::MatrixXd::Ones(atom_num, 4);
        homo_source.block(0, 0, atom_num, 3) = source;
		//std::cout << "homo_source (first two rows):\n" << homo_source.topRows(2) << std::endl;
        transformed_source = (transformation.value() * homo_source.transpose()).transpose().block(0, 0, atom_num, 3);
        // 应用变换
        //std::cout << "Transformed source (first two rows):\n" << transformed_source.topRows(2) << std::endl;
    } else {
        transformed_source = source;
    }

    // 计算均方偏差 (MSD)
    Eigen::MatrixXd diffs = transformed_source - target;
    double msd = (diffs.array().pow(2).rowwise().sum()).mean();
    
    // 计算均方根偏差 (RMSD)
    double rmsd = std::sqrt(msd);
    return rmsd;
}
/*
double Registration::calRMSD(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target, const std::optional<Eigen::Matrix4d>& transformation) {
    int atom_num = source.rows();
	std::cout << "atom_num:" << atom_num << std::endl;
    Eigen::MatrixXd transformed_source;

    // 如果提供了变换矩阵
    if (transformation) {
        Eigen::MatrixXd homo_source(atom_num, 4);
        homo_source << source, Eigen::VectorXd::Ones(atom_num);
        transformed_source = (transformation.value() * homo_source.transpose()).transpose();
		std::cout << "Transformed Source:\n" << transformed_source.topRows(3) << std::endl;
    } else {
        transformed_source = source;
    }

    // 计算均方根偏差
    double msd = 0.0;
    for (int i = 0; i < atom_num; ++i) {
        msd += (transformed_source.row(i).head(3) - target.row(i)).squaredNorm();
    }
    msd = std::sqrt(msd);

    return msd / static_cast<double>(atom_num);
}
*/
Eigen::MatrixXd Registration::getPointsFromPDB(const std::string& pdbFile, const std::string& flag) {
    std::vector<std::vector<double>> all_atom_coord;
    std::string useFlag = flag;

    if (useFlag.empty()) {
        useFlag = pdbFile.substr(pdbFile.length() - 3);
    }

    std::ifstream fr(pdbFile);
    std::string line;

    if (!fr.is_open()) {
        std::cerr << "Unable to open file: " << pdbFile << std::endl;
        return Eigen::MatrixXd(); // Return empty matrix
    }

    while (std::getline(fr, line)) {
        if (line.substr(0, 4) == "ATOM") {
            std::istringstream iss(line.substr(27, 27)); // For PDB format
            std::vector<double> atom_coord;
            double x, y, z;

            if (useFlag == "pdb" || useFlag == "ent") {
                if (iss >> x >> y >> z) {
                    atom_coord.push_back(x);
                    atom_coord.push_back(y);
                    atom_coord.push_back(z);
                    all_atom_coord.push_back(atom_coord);
                } else {
                    std::cerr << "throw away 3D coordinate " << line.substr(27, 27).c_str() << std::endl;
                    std::cerr << pdbFile << std::endl;
                }
            } else if (useFlag == "cif") {
                std::istringstream iss_cif(line); // For CIF format
                std::vector<std::string> temp_atom_coord;
                std::string item;
                while (iss_cif >> item) {
                    temp_atom_coord.push_back(item);
                }
                try {
                    if (temp_atom_coord.size() == 18) {
                        x = std::stod(temp_atom_coord[9]);
                        y = std::stod(temp_atom_coord[10]);
                        z = std::stod(temp_atom_coord[11]);
                        atom_coord.push_back(x);
                        atom_coord.push_back(y);
                        atom_coord.push_back(z);
                        all_atom_coord.push_back(atom_coord);
                    }else if (temp_atom_coord.size() == 21){
						x = std::stod(temp_atom_coord[10]);
                        y = std::stod(temp_atom_coord[11]);
                        z = std::stod(temp_atom_coord[12]);
                        atom_coord.push_back(x);
                        atom_coord.push_back(y);
                        atom_coord.push_back(z);
                        all_atom_coord.push_back(atom_coord);					
					}
                } catch (const std::exception& e) {
                    std::cerr << "throw away 3D coordinate " << temp_atom_coord[9] << " "
                              << temp_atom_coord[10] << " " << temp_atom_coord[11] << std::endl;
                    std::cerr << pdbFile << std::endl;
                }
            }
        }
    }

    if (all_atom_coord.empty()) {
        std::cerr << "Warning: point set is empty!" << std::endl;
    }

    // Convert points to Eigen matrix for return
    Eigen::MatrixXd points(all_atom_coord.size(), 3);
    for (size_t i = 0; i < all_atom_coord.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            points(i, j) = all_atom_coord[i][j];
        }
    }
    return points;
}
void Registration::WriteTrans(const Eigen::Matrix4d& transformation, const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件：" << file_path << std::endl;
        return;
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            file << std::fixed << std::setprecision(6) << transformation(i, j);
            if (j != 3) file << "\t";
        }
        file << std::endl;
    }
    file.close();
}

Eigen::Matrix4d Registration::ReadTrans(const std::string& file_path) {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件：" << file_path << std::endl;
        return transformation;
    }
    std::string line;
    for (int i = 0; i < 3; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        for (int j = 0; j < 4; ++j) {
            iss >> transformation(i, j);
        }
    }
    file.close();
    return transformation;
}
void Registration::WritePLY(const open3d::geometry::PointCloud& pcd, const std::string& file_path, bool normal) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "无法写入文件: " << file_path << std::endl;
        return;
    }

    // 写入PLY头部信息
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "comment Created by Open3D\n";
    file << "element vertex " << pcd.points_.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    
    if (normal && !pcd.normals_.empty()) {
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
    }
    
    file << "end_header\n";

    // 写入点云数据
    for (size_t i = 0; i < pcd.points_.size(); ++i) {
        file << pcd.points_[i](0) << " " << pcd.points_[i](1) << " " << pcd.points_[i](2);
        if (normal && !pcd.normals_.empty()) {
            file << " " << pcd.normals_[i](0) << " " << pcd.normals_[i](1) << " " << pcd.normals_[i](2);
        }
        file << "\n";
    }

    file.close();
}


void Registration::draw_registration_result(const open3d::geometry::PointCloud& source,
                              const open3d::geometry::PointCloud& target,
                              const std::optional<Eigen::Matrix4d>& transformation) {
    // 创建点云的拷贝
    auto source_temp = std::make_shared<open3d::geometry::PointCloud>(source);
    auto target_temp = std::make_shared<open3d::geometry::PointCloud>(target);

    // 上色
    source_temp->PaintUniformColor({1, 0.706, 0});
    target_temp->PaintUniformColor({0, 0.651, 0.929});

    // 如果提供了变换矩阵，则应用变换
    if (transformation.has_value()) {
        source_temp->Transform(transformation.value());
    }

    // 可视化
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Registration Result", 1600, 900);
    vis.AddGeometry(source_temp);
    vis.AddGeometry(target_temp);
    vis.GetViewControl().SetZoom(0.4559);
    vis.GetViewControl().SetFront({0.6452, -0.3036, -0.7011});
    vis.GetViewControl().SetLookat({1.9892, 2.0208, 1.8945});
    vis.GetViewControl().SetUp({-0.2779, -0.9482, 0.1556});
    vis.Run();
    vis.DestroyVisualizerWindow();
}
Eigen::Matrix4d Registration::Rt2T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

teaser::RobustRegistrationSolver::Params Registration::getTeaserSolverParams(float noise_bound) {
    teaser::RobustRegistrationSolver::Params params;
    params.cbar2 = 1.0;
    params.noise_bound = noise_bound;
    params.estimate_scaling = false;
    params.inlier_selection_mode = teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT;
    params.rotation_tim_graph = teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN;
    params.rotation_estimation_algorithm = teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_gnc_factor = 1.4;
    params.rotation_max_iterations = 10;
    params.rotation_cost_threshold = 1e-16;
    return params;
}
std::pair<std::vector<int>, std::vector<int>> Registration::find_correspondences(
    const Eigen::MatrixXd& feats0,
    const Eigen::MatrixXd& feats1) {

    int dim = feats1.cols();  // 特征的维度
    int nPts = feats1.rows(); // 特征点的数量

    flann::Matrix<double> dataset(new double[nPts*dim], nPts, dim);
    flann::Matrix<double> query(new double[feats0.rows()*dim], feats0.rows(), dim);

    for (int i = 0; i < nPts; ++i) {
        for (int j = 0; j < dim; ++j) {
            dataset[i][j] = feats1(i, j);
        }
    }

    for (int i = 0; i < feats0.rows(); ++i) {
        for (int j = 0; j < dim; ++j) {
            query[i][j] = feats0(i, j);
        }
    }

    std::vector<std::vector<int>> indices;
    std::vector<std::vector<double>> dists;

    flann::Index<flann::L2<double>> index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();
    index.knnSearch(query, indices, dists, 1, flann::SearchParams(128));

    std::vector<int> corres_idx0, corres_idx1;

    for (int i = 0; i < indices.size(); ++i) {
        corres_idx0.push_back(i);
        corres_idx1.push_back(indices[i][0]);
    }

    delete[] dataset.ptr();
    delete[] query.ptr();

    return {corres_idx0, corres_idx1};
}
open3d::geometry::PointCloud Registration::load_xyz(const std::string& file_path) {
    open3d::geometry::PointCloud pcd;
    if (!open3d::io::ReadPointCloud(file_path, pcd)) {
        throw std::runtime_error("Failed to read point cloud from file: " + file_path);
    }
    return pcd;
}
std::tuple<std::vector<std::array<double, 3>>, std::vector<std::array<double, 3>>, std::vector<std::array<double, 1>>> Registration::load_sample(const std::string& file_path, bool density) {
    std::vector<std::array<double, 3>> point_list;
    std::vector<std::array<double, 3>> vector_list;
    std::vector<std::array<double, 1>> density_list;

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::string line;
    double sample = 5.0d;
    std::array<double, 3> origin;

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
            double x = 5.0d, y = 5.0d, z = 5.0d;
            iss >>ignore>>x >> y >> z;
            point_list.push_back({{x * sample + origin[0], y * sample + origin[1], z * sample + origin[2]}});
        } else {
            double vx = 5.0d, vy = 5.0d, vz = 5.0d, d = 5.0d;
            iss >> vx >> vy >> vz >> d;
            vector_list.push_back({{vx, vy, vz}});
            density_list.push_back({{d}});
        }
		
    }

    if (density) {
        return std::make_tuple(point_list, vector_list, density_list);
    }
    return std::make_tuple(point_list, vector_list, std::vector<std::array<double, 1>>());
}
Eigen::MatrixXd Registration::cal_SHOT(const std::vector<std::array<double, 3>>& points,
                         const std::vector<std::array<double, 3>>& normals,
                         const std::string& temp_dir,
                         const std::vector<Eigen::Vector3d>& key_points,
                         float radius) {
    // Generate file paths
    std::string points_filename = temp_dir + "/points.pcd";
    std::string normals_filename = temp_dir + "/normals.txt";
    std::string key_points_filename = temp_dir + "/key_points.pcd";
    std::string feature_filename = temp_dir + "/SHOT_features.txt";

    // Save points to PCD file
    txt2pcd(points, points_filename);

    // Save normals to TXT file
    std::ofstream normals_out(normals_filename);
    if (!normals_out) {
        throw std::runtime_error("Cannot open file: " + normals_filename);
    }
	normals_out << std::fixed << std::setprecision(5);
    for (const auto& normal : normals) {
        normals_out << normal[0] << " " << normal[1] << " " << normal[2] << '\n';
    }
    normals_out.close();

    // Save key points to PCD file
    txt2pcd(key_points, key_points_filename);
	Eigen::MatrixXd shotFeatures = computeSHOTFeatures(points_filename, normals_filename, key_points_filename, radius);
    //std::cout << "SHOT Features Matrix: \n" << shotFeatures << std::endl;
	/*
    // Execute the external command to compute SHOT features
    std::string command = "./point_cloud_feature " + points_filename + " " +
                          normals_filename + " " + key_points_filename + " " +
                          std::to_string(radius) + " > " + feature_filename;
    int result = std::system(command.c_str());
    if (result != 0) {
        throw std::runtime_error("External command failed with code: " + std::to_string(result));
    }
	*/
    // Read and return features
    return shotFeatures;
}


Eigen::MatrixXd Registration::computeSHOTFeatures(const std::string& pointCloudFilename,
                                                  const std::string& normalFilename,
                                                  const std::string& keypointFilename,
                                                  float radius) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

    loadPointCloud(pointCloudFilename, cloud);
    loadNormals(normalFilename, normals);
    loadKeypoints(keypointFilename, keypoints);

    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    shot.setInputCloud(keypoints);
    shot.setSearchSurface(cloud);
    shot.setInputNormals(normals);
    shot.setRadiusSearch(radius);

    pcl::PointCloud<pcl::SHOT352>::Ptr features(new pcl::PointCloud<pcl::SHOT352>);
    shot.compute(*features);

    return convertFeaturesToMatrix(*features);
}

void Registration::loadPointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        throw std::runtime_error("Couldn't read file " + filename);
    }
}

void Registration::loadNormals(const std::string& filename, pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    Eigen::MatrixXd data = readMatrix(filename);
    normals->resize(data.rows());
    for (int i = 0; i < data.rows(); i++) {
        normals->points[i].normal_x = data(i, 0);
        normals->points[i].normal_y = data(i, 1);
        normals->points[i].normal_z = data(i, 2);
    }
}

void Registration::loadKeypoints(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *keypoints) == -1) {
        throw std::runtime_error("Couldn't read file " + filename);
    }
}

Eigen::MatrixXd Registration::convertFeaturesToMatrix(const pcl::PointCloud<pcl::SHOT352>& features) {
    Eigen::MatrixXd result(features.size(), 352);
    for (size_t i = 0; i < features.size(); ++i) {
        for (int j = 0; j < 352; ++j) {
            result(i, j) = features.points[i].descriptor[j];
        }
    }
    return result;
}

Eigen::MatrixXd Registration::readMatrix(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::vector<double> values;
    int rows = 0;
    std::string line;

    while (getline(infile, line)) {
        std::stringstream stream(line);
        double value;
        while (stream >> value) {
            values.push_back(value);
        }
        ++rows;
    }
    int cols = values.size() / rows;

    Eigen::MatrixXd result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = values[i * cols + j];
        }
    }
    return result;
}




// 用于处理 std::vector<std::array<float, 3>> 类型的点云数据
void Registration::txt2pcd(const std::vector<std::array<double, 3>>& points, const std::string& output_filename) {
    std::ofstream out_file(output_filename);
    if (!out_file.is_open()) {
        throw std::runtime_error("Cannot open the output file: " + output_filename);
    }
	out_file << std::fixed << std::setprecision(5);
    out_file << "# .PCD v0.7 - Point Cloud Data file format\n";
    out_file << "VERSION 0.7\n";
    out_file << "FIELDS x y z\n";
    out_file << "SIZE 4 4 4\n";
    out_file << "TYPE F F F\n";
    out_file << "COUNT 1 1 1\n";
    out_file << "WIDTH " << points.size() << "\n";
    out_file << "HEIGHT 1\n";
    out_file << "VIEWPOINT 0 0 0 1 0 0 0\n";
    out_file << "POINTS " << points.size() << "\n";
    out_file << "DATA ascii\n";

    for (const auto& point : points) {
        out_file << point[0] << " " << point[1] << " " << point[2] << "\n";
    }
}

// 用于处理 std::vector<Eigen::Vector3d> 类型的点云数据
void Registration::txt2pcd(const std::vector<Eigen::Vector3d>& points, const std::string& output_filename) {
    std::ofstream out_file(output_filename);
    if (!out_file.is_open()) {
        throw std::runtime_error("Cannot open the output file: " + output_filename);
    }
	out_file << std::fixed << std::setprecision(5);
    out_file << "# .PCD v0.7 - Point Cloud Data file format\n";
    out_file << "VERSION 0.7\n";
    out_file << "FIELDS x y z\n";
    out_file << "SIZE 4 4 4\n";
    out_file << "TYPE F F F\n";
    out_file << "COUNT 1 1 1\n";
    out_file << "WIDTH " << points.size() << "\n";
    out_file << "HEIGHT 1\n";
    out_file << "VIEWPOINT 0 0 0 1 0 0 0\n";
    out_file << "POINTS " << points.size() << "\n";
    out_file << "DATA ascii\n";

    for (const auto& point : points) {
        out_file << point.x() << " " << point.y() << " " << point.z() << "\n";
    }
}
Eigen::MatrixXd Registration::read_features(const std::string& feature_dir, const std::string& mode, bool key) {
    std::vector<std::vector<double>> features;
    std::ifstream file(feature_dir);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + feature_dir);
    }

    std::string line;
    while (std::getline(file, line)) {
        // 根据是否使用key和mode的值，选择性地提取特定数据
        std::string processedLine;
        if (!key && (mode == "SHOT" || mode == "3DSC" || mode == "USC")) {
            size_t firstCloseParen = line.find(')'); // 查找第一个闭合括号
            size_t secondOpenParen = line.find('(', firstCloseParen); // 查找第一个闭合括号之后的开括号
            if (secondOpenParen != std::string::npos) {
                size_t secondCloseParen = line.find(')', secondOpenParen); // 查找第二个闭合括号
                processedLine = line.substr(secondOpenParen + 1, secondCloseParen - secondOpenParen - 1);
            }
        } else if (key && mode == "SHOT") {
            size_t firstCloseParen = line.find(')'); // 查找第一个闭合括号
            size_t secondOpenParen = line.find('(', firstCloseParen); // 查找第一个闭合括号之后的开括号
            if (secondOpenParen != std::string::npos) {
                size_t secondCloseParen = line.find(')', secondOpenParen); // 查找第二个闭合括号
                processedLine = line.substr(secondOpenParen + 1, secondCloseParen - secondOpenParen - 1);
            }
        } else {
            processedLine = line.substr(1, line.size() - 2);
        }

        std::istringstream iss(processedLine);
        std::vector<double> feature_vector;
        std::string value;
        while (std::getline(iss, value, ',')) {
            feature_vector.push_back(std::stod(value)); // 应使用std::stod而不是std::stof
        }
        features.push_back(feature_vector);
    }

    Eigen::MatrixXd feature_matrix(features.size(), features.empty() ? 0 : features[0].size());
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = 0; j < features[i].size(); ++j) {
            feature_matrix(i, j) = features[i][j];
        }
    }

    return feature_matrix;
}
