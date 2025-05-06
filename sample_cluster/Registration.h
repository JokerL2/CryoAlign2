#ifndef REGISTRATION_H
#define REGISTRATION_H
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <map>
#include <any>
#include <sstream>
#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <teaser/registration.h>
#include <sstream>
#include <cstdlib>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
class Registration {
public:
    // Constructor
    Registration();
	Eigen::MatrixXd computeSHOTFeatures(const std::string& pointCloudFilename,
                                        const std::string& normalFilename,
                                        const std::string& keypointFilename,
                                        float radius);
	
	void loadPointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void loadNormals(const std::string& filename, pcl::PointCloud<pcl::Normal>::Ptr& normals);
    void loadKeypoints(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoints);
    Eigen::MatrixXd convertFeaturesToMatrix(const pcl::PointCloud<pcl::SHOT352>& features);
    Eigen::MatrixXd readMatrix(const std::string& filename);
	std::tuple<std::vector<std::array<double, 3>>, std::vector<std::array<double, 3>>, std::vector<std::array<double, 1>>>
    load_sample(const std::string& file_path, bool density = false);
    open3d::geometry::PointCloud load_xyz(const std::string& file_path);
    Eigen::Matrix4d Registration_given_feature(const std::string& data_dir,
                                    const std::string& source_dir,
                                    const std::string& target_dir,
                                    const std::string& source_sample_dir,
                                    const std::string& target_sample_dir,
									const std::string& source_pdb,
									const std::string& sup_pdb,
									//const std::string& res_path,									
                                    float VOXEL_SIZE = 5.0,
									float feature_radius = 7.0,
                  const std::string& outnum = "",
                                    bool visualize = false,
                                    std::optional<Eigen::Matrix4d> T = std::nullopt,
                                    bool one_stage = false);
	Eigen::MatrixXd read_features(const std::string& feature_dir, const std::string& mode = "SHOT", bool key = false);
	//void print_py_array(const py::array_t<double>& arr);
	void txt2pcd(const std::vector<Eigen::Vector3d>& points, const std::string& output_filename);
	void txt2pcd(const std::vector<std::array<double, 3>>& points, const std::string& output_filename);
	Eigen::MatrixXd cal_SHOT(const std::vector<std::array<double, 3>>& points,
                         const std::vector<std::array<double, 3>>& normals,
                         const std::string& temp_dir,
                         const std::vector<Eigen::Vector3d>& key_points,
                         float radius = 25.0f);
	
	std::pair<std::vector<int>, std::vector<int>>find_correspondences(
    const Eigen::MatrixXd& feats0, const Eigen::MatrixXd& feats1);
	Eigen::Matrix4d Rt2T(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	teaser::RobustRegistrationSolver::Params getTeaserSolverParams(float noise_bound);
	void draw_registration_result(const open3d::geometry::PointCloud& source,
                              const open3d::geometry::PointCloud& target,
                              const std::optional<Eigen::Matrix4d>& transformation = std::nullopt);
	void WritePLY(const open3d::geometry::PointCloud& pcd, const std::string& file_path, bool normal = false);
	void WriteTrans(const Eigen::Matrix4d& transformation, const std::string& file_path);
	Eigen::Matrix4d ReadTrans(const std::string& file_path);
	Eigen::MatrixXd getPointsFromPDB(const std::string& pdbFile, const std::string& flag = "");
	double calRMSD(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target, const std::optional<Eigen::Matrix4d>& transformation = std::nullopt);
	void Registration_mask_list(  const std::string& data_dir,
								  const std::string& source_dir,
								  const std::string& target_dir,
								  const std::string& source_sample_dir,
								  const std::string& target_sample_dir,
								  float VOXEL_SIZE = 5.0,
								  float feature_radius = 7.0,
                  const std::string& outnum = "",
								  bool store_partial = false);
	std::tuple<std::optional<Eigen::Matrix4d>, double> Registration_mask(
							const std::string& data_dir,
							const open3d::geometry::PointCloud& A_pcd,
							const open3d::geometry::PointCloud& B_pcd,
							const open3d::geometry::PointCloud& A_key_pcd,
							const open3d::geometry::PointCloud& B_key_pcd,
							const Eigen::MatrixXd& A_key_feats, // 请根据实际特征的类型替换 Eigen::MatrixXd
							const Eigen::MatrixXd& B_key_feats,
							const std::unordered_map<std::string,std::variant<std::string, Eigen::Vector3d, double>>& mask,
							float max_correspondence_dist = 10.0,
							float VOXEL_SIZE = 5.0,
							bool store_partial = false);
	std::pair<open3d::geometry::PointCloud, std::vector<size_t>> MaskSpherePoints(const open3d::geometry::PointCloud& pcd,
                 const Eigen::Vector3d& center,
                 double radius);
	double cal_score(const open3d::geometry::PointCloud& A_pcd,
                 const open3d::geometry::PointCloud& B_pcd,
				 double shot_score,
                 double max_correspondence_dist,
                 const Eigen::Matrix4d& final_T);
	std::vector<double> compute_cosine_distances(
    const Eigen::MatrixXd& feats_A,
    const Eigen::MatrixXd& feats_B,
    const std::vector<int>& corrs_A,
    const std::vector<int>& corrs_B);
	double calc_SHOT_score(const std::vector<double>& cosine_distances);
	void extract_top_K(const std::string& record_dir, const std::string& record_T_dir, int K, const std::string& save_dir, const std::string& source_pdb_dir = "", const std::string& source_sup_dir = "");
	double cal_pdb_RMSD(const std::string& source_pdb_dir,const std::string& source_sup_dir,Eigen::Matrix4d T);
private:
    std::unordered_map<std::string,std::variant<std::string, Eigen::Vector3d, double>> mask;
};

#endif
