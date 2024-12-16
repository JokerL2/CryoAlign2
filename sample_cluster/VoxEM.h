#ifndef VOXEM_H
#define VOXEM_H
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <map>
#include <any>
#include <sstream>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/core.hpp>
#include <armadillo>
class VoxEM {
public:
    // Constructor
    VoxEM();

    void IO_ReadMrc(const std::string& mrc_inputname = "XXX.mrc", const std::string& voxel_outputname = "Default", const std::string& grid_outputname = "", const std::string& description_outputname = "Default", const std::string& statistics_outputname = "Default", bool nonzerostatistics = true);

    void Point_Create_Grid(const std::string& voxel_inputname = "Default", const std::string& point_outputname = "Default", const std::string& description_inputname = "Default");

    void IO_Statistics(const std::string& voxel_inputname = "Voxel", const std::string& statistics_outputname = "Default", bool nonzerostatistics = true);
    
    //void PrintDescriptionWorkspace(const std::string& description_outputname);
    
    void Voxel_Prune_RangeZero(float upperbound =-1.0 , float lowerbound = -1.0, const std::string& inputname = "Voxel", const std::string& outputname = "Voxel");

    // Add getters for private members
    std::vector<std::vector<std::vector<float>>> getVoxelWorkspace(const std::string& key);
    std::unordered_map<std::string, std::any> getDescriptionWorkspace(const std::string& key);
	void setPointWorkspaceValue(const std::string& key, const std::vector<std::array<float, 3>>& value);
	std::tuple<std::vector<std::array<float, 3>>, std::vector<std::array<float, 3>>, std::vector<std::array<float, 1>>> load_sample_points(const std::string& file_path, bool density);
	//std::vector<std::vector<float>> transpose(const std::vector<std::array<float, 3>>& matrix);
	//std::vector<std::array<float, 3>> convertToArrays(const std::vector<std::vector<float>>& input);
	std::vector<float> getPointWorkspaceValue(const std::string& key) const;
	void Point_Create_Meanshift_sample(
        float lower_bound = 0.05,
        int window = 7,
        float bandwidth = 1.0,
        int iteration = 800,
        float step_size = 0.01,
        float convergence = 0.000187,
        const std::string& voxel_inputname = "Voxel",
        const std::string& point_outputname = "Meanshift",
        const std::string& point_inputname = "",
        const std::string& description_inputname = "Default"
    );
	void Point_Transform_DBSCAN(
		const double distance_tolerance = 1.9,
		const size_t clustersize_tolerance = 13,
		const std::string& inputname = "Voxel",
		const std::string& outputname = "DBSCAN",
		const bool centroid_only = false,
		const int n_job = 50
	);
	arma::colvec3 CalculateCentroid(const int cluster_id,
                                const arma::mat& points,
                                const arma::Row<size_t>& labels);
	void IO_WriteXYZ(const std::string& point_inputname,
                 const std::string& file_outputname,
                 const std::string& atom_name,
                 const std::array<float, 3>& shifting);
	void XYZ(const std::vector<float>& points, const std::string& atom_name, const std::string& filename);
private:
    std::unordered_map<std::string, std::vector<std::vector<std::vector<float>>>> voxel_workspace;
    std::unordered_map<std::string, std::vector<float>> point_workspace;
    std::unordered_map<std::string, std::vector<int>> graph_workspace;
    std::unordered_map<std::string, std::vector<int>> segment_workspace;
    std::unordered_map<std::string, std::unordered_map<std::string, std::any>> statistics_workspace;
    std::unordered_map<std::string, std::unordered_map<std::string, std::any>> description_workspace;
    //std::unordered_map<std::string, std::unordered_map<std::string, std::variant<int, std::array<int, 3>, std::array<float, 3>, std::array<int, 3>, std::vector<float>>> description_workspace;

    std::tuple<int, int, int> unpack_ints(const char* data, int offset);
    int unpack_int(const char* data, int offset);
    std::tuple<float, float, float> unpack_floats(const char* data, int offset);
    std::vector<float> unpack_float_array(const char* data, int offset, int count);

    float calculateRMS(const std::vector<float>& values);
    float calculateMin(const std::vector<float>& values);
    float calculateMax(const std::vector<float>& values);
    float calculateMean(const std::vector<float>& values);
};

#endif