#include "sample_cluster/VoxEM.h"
#include <cstdlib> // 引入 system() 函数所需的头文件
#include <cstdio>  // 引入 sprintf() 函数所需的头文件
#include <cmath>
using namespace std;
void Sample_cluster(const std::string& data_dir,const std::string& map_name, float threshold,float voxel_size){
	// 使用字符串流来格式化浮点数
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << voxel_size; // 设置小数点后两位
    std::string voxel_size_str = oss.str();
	std::string mrc_inputname = data_dir + "/" + map_name;
	std::string before_map;
	if (map_name.find('-') == std::string::npos) {
		before_map = map_name.substr(map_name.rfind('_') + 1);
		before_map = before_map.substr(0, before_map.find(".map"));
	}else{
		std::string sub_map = map_name.substr(map_name.find('-') + 1, map_name.find('_') - map_name.find('-') - 1);
		before_map = sub_map.substr(0, sub_map.find(".map"));
	}
    std::string file_outputname = data_dir + "/" + "Points_" + before_map + "_" + voxel_size_str + "_Key.xyz";
    std::string sample_file = data_dir + "/" + map_name.substr(0, map_name.size() - 4) + "_" + voxel_size_str + ".txt";
	
	std::cout << "MRC file: " << mrc_inputname << std::endl;
    std::cout << "Sub map: " << before_map << std::endl;
    std::cout << "Output file: " << file_outputname << std::endl;
    std::cout << "Sample file: " << sample_file << std::endl;
	
	
	char sample_command[512];
	
    // 使用 sprintf 构建命令行字符串
    sprintf(sample_command, "/home/liu/croy_align_cpp/CryoAlign_cpp_v3/sample_cluster/Sample -a %s -t %.4f -s %.2f > %s", mrc_inputname.c_str(), threshold, voxel_size, sample_file.c_str());

    // 使用 system() 执行命令
    int ret_val = system(sample_command);
	if (ret_val != 0) {
		std::cerr << "Sample 命令执行失败" << std::endl;
	}
	
	
	VoxEM voxem;
	voxem.IO_ReadMrc(mrc_inputname , "Default", "", "Default", "Default", true);	
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
	
	voxem.Point_Transform_DBSCAN(voxel_size * 3, 3,"Meanshift", "Meanshift",false);
	voxem.Point_Transform_DBSCAN(voxel_size, 2, "Meanshift", "DBSCAN",true,100);
	
	
	voxem.IO_WriteXYZ("DBSCAN",file_outputname,"H",shifting);
	std::cout << "Sampling and clustering success" << std::endl;
}

void print_help() {
    std::cout << "Usage: CryoAlign_extract_keypoints [data dir] [source.map] [source contour level] [target.map] [target contour level] [voxel_size]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_dir: Map file path." << std::endl;
    std::cout << "  --map_name: Source emdb num." << std::endl;
    std::cout << "  --contour_level: Author recommend contour level." << std::endl;
	std::cout << "  --voxel_size: Sampling interval.(defaults 5.0)" << std::endl;
	std::cout << std::endl;
	std::cout << "Example:" << std::endl;
	std::cout << "  CryoAlign_extract_keypoints --data_dir ../../example_dataset/emd_3695_emd_3696/ --map_name EMD-3695.map --contour_level 0.008 --voxel_size 5.0" << std::endl;
}
int main(int argc, char* argv[]) {

    // Initialize variables with default values or empty strings
    string data_dir;
    string map_name;
    float threshold = 0.0f;
	float voxel_size = 0.0f;
	bool has_voxel_size = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else if (arg == "--data_dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--map_name" && i + 1 < argc) {
            map_name = argv[++i];
        } else if (arg == "--contour_level" && i + 1 < argc) {
            threshold = stof(argv[++i]);
        } else if (arg == "--voxel_size" && i + 1 < argc) {
	    has_voxel_size = true;
	    voxel_size = stof(argv[++i]);
	} else {
            cout << "Error: Invalid argument " << arg << endl;
            print_help();
            return 1;
        }
    }

    if (data_dir.empty() || map_name.empty()) {
        cout << "Error: Missing required arguments." << endl;
        print_help();
        return 1;
    }

	
    Sample_cluster(data_dir,map_name, threshold,has_voxel_size ? voxel_size : 5.0);
	
	
	
	
	
	
    return 0;
}
