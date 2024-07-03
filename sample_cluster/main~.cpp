#include "VoxEM.h"
#include "Registration.h"
int main(int argc, char* argv[]) {
	
	VoxEM voxem;
    std::string mrc_inputname = argv[1];
	std::string sample_file = argv[2];
	float threshold = std::stof(argv[3]);
	std::string file_outputname = argv[4];
    voxem.IO_ReadMrc(mrc_inputname, "Default", "", "Default", "Default", true);	
    voxem.Voxel_Prune_RangeZero(threshold, -1.0, "Default", "Voxel");

    std::vector<std::vector<std::vector<float>>> vox = voxem.getVoxelWorkspace("Voxel");
    std::array<int, 3> start;
    if (std::any start_any = voxem.getDescriptionWorkspace("Default")["Start"]; start_any.has_value() && start_any.type() == typeid(std::array<int, 3>)) {
        start = std::any_cast<std::array<int, 3>>(start_any);
    }
    std::array<float, 3> origin;
    if (std::any origin_any = voxem.getDescriptionWorkspace("Default")["Origin"]; origin_any.has_value() && origin_any.type() == typeid(std::array<float, 3>)) {
        origin = std::any_cast<std::array<float, 3>>(origin_any);
    }
    std::array<float, 3> vox_length;
    if (std::any vox_length_any = voxem.getDescriptionWorkspace("Default")["Angstrom"]; vox_length_any.has_value() && vox_length_any.type() == typeid(std::array<float, 3>)) {
        vox_length = std::any_cast<std::array<float, 3>>(vox_length_any);
    }
    std::array<float, 3> vox_ang = {vox_length[0] / static_cast<float>(vox.size()), vox_length[1] / static_cast<float>(vox[0].size()), vox_length[2] / static_cast<float>(vox[0][0].size())};

    
    std::vector<std::array<float, 3>> sample_points;
    std::vector<std::array<float, 3>> sample_vector;
    std::vector<std::array<float, 1>> density_list;
    std::tie(sample_points, sample_vector, density_list) = voxem.load_sample_points(sample_file, false);

    std::array<float, 3> shifting = {
        start[0] * vox_ang[0] + origin[0],
        start[1] * vox_ang[1] + origin[1],
        start[2] * vox_ang[2] + origin[2]
    };
	// 打印shifting数组
    std::cout << "Shifting: ["
              << shifting[0] << ", "
              << shifting[1] << ", "
              << shifting[2] << "]" << std::endl;
    for (auto& point : sample_points) {
        point[0] -= shifting[0];
        point[1] -= shifting[1];
        point[2] -= shifting[2];
    }
    voxem.setPointWorkspaceValue("sample", sample_points);
	std::vector<float> sample_values = voxem.getPointWorkspaceValue("sample");	
	
	std::cout << "sample_points_transposed shape: (" << sample_points.size() << ", " << sample_points[0].size() << ")" << std::endl;
	voxem.Point_Create_Meanshift_sample(
    threshold, // lower_bound
    17,        // window
    3.0,       // bandwidth
    2000,      // iteration
    0.05,      // step_size
    0.000187,  // convergence
    "Voxel",   // voxel_inputname
    "Meanshift", // point_outputname
    "sample"   // point_inputname
    // description_inputname 使用默认值 "Default"
);
	
	voxem.Point_Transform_DBSCAN(5.0 * 3, 3,"Meanshift", "Meanshift",false);
	voxem.Point_Transform_DBSCAN(5.0, 3, "Meanshift", "DBSCAN",true,100);
	
	
	voxem.IO_WriteXYZ("DBSCAN",file_outputname,"H",shifting);
	std::cout << "Sampling and clustering success" << std::endl;
	
	
	
    return 0;
}
