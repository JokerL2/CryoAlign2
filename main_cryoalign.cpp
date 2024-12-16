#include "sample_cluster/VoxEM.h"
#include "alignment/Registration.h"
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
void direct_alignment(const std::string& temp_path, const std::string& source_key_dir,const std::string& target_key_dir,const std::string& source_sample_dir, const std::string& target_sample_dir,std::optional<std::string> source_pdb, std::optional<std::string> sup_pdb,float voxel_size,float feature_radius,const std::string& outnum){
	Registration dres;
	Eigen::Matrix4d T;

    if (source_pdb && sup_pdb) {
        T = dres.Registration_given_feature(temp_path, source_key_dir, target_key_dir, source_sample_dir, target_sample_dir, source_pdb.value(), sup_pdb.value(), voxel_size,feature_radius,outnum);
    } else {
        T = dres.Registration_given_feature(temp_path, source_key_dir, target_key_dir, source_sample_dir, target_sample_dir, "", "", voxel_size,feature_radius,outnum);
    }
	
	std::cout << "Estimated transformation: \n" << T << std::endl;
}
void mask_alignment(const std::string& temp_path, const std::string& source_key_dir,const std::string& target_key_dir,const std::string& source_sample_dir, const std::string& target_sample_dir,std::optional<std::string> source_pdb, std::optional<std::string> sup_pdb,float voxel_size,float feature_radius,const std::string& outnum){
	Registration mres;
	
	mres.Registration_mask_list(temp_path, source_key_dir,target_key_dir, source_sample_dir,target_sample_dir, voxel_size,feature_radius,outnum);
	std::string record_dir = temp_path+"/"+outnum+"_record_cpp.txt";
	std::string record_T_dir = temp_path+"/"+outnum+"_record_T_cpp.csv";
	int k=10;
	std::string save_dir = temp_path+"/"+outnum+"_extract_top_10.txt";
	if (source_pdb && sup_pdb) {
        mres.extract_top_K(record_dir, record_T_dir, k, save_dir, source_pdb.value(), sup_pdb.value());
    } else {
        mres.extract_top_K(record_dir, record_T_dir, k, save_dir, "", "");
    }
}
void print_help() {
    std::cout << "Usage: CryoAlign [data dir] [source.map] [source contour level] [target.map]" << std::endl;
    std::cout << "[target contour level] [source.pdb] [source sup.pdb] [voxel_size] [feature_radius] [alg_type]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_dir: Map file path." << std::endl;
    std::cout << "  --source_map: Source emdb num." << std::endl;
    std::cout << "  --source_contour_level: Author recommend contour level." << std::endl;
    std::cout << "  --target_map: Target emdb num." << std::endl;
    std::cout << "  --target_contour_level: Author recommend contour level." << std::endl;
    std::cout << "  --source_pdb(optional): Source pdb name." << std::endl;
    std::cout << "  --source_sup_pdb(optional): Transformed source pdb name (ground truth)." << std::endl;
	std::cout << "  --voxel_size: Sampling interval.(defaults 5.0)" << std::endl;
	std::cout << "  --feature_radius: Radius for feature construction.(defaults 7.0)" << std::endl;
    std::cout << "  --alg_type: Global_alignment or Mask_alignment." << std::endl;
	std::cout << std::endl;
	std::cout << "Example:" << std::endl;
	std::cout << "  For Global_alignment:" << std::endl;
	std::cout << "  CryoAlign --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_map EMD-3695.map --source_contour_level 0.008 --target_map EMD-3696.map --target_contour_level 0.002 --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type global" << std::endl;
	std::cout << "  For Mask_alignment:" << std::endl;
	std::cout << "  CryoAlign --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_map EMD-3695.map --source_contour_level 0.008 --target_map EMD-3696.map --target_contour_level 0.002 --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type mask" << std::endl;
}
int main(int argc, char* argv[]) {

    // Initialize variables with default values or empty strings
    string data_dir;
    string map_name1;
    string map_name2;
    float threshold1 = 0.0f;
    float threshold2 = 0.0f;
    optional<string> source_pdb;
    optional<string> sup_pdb;
    string alg_type;
	float voxel_size = 0.0f;
	float feature_radius = 0.0f;
	bool has_source_pdb = false;
    bool has_sup_pdb = false;
	bool has_voxel_size = false;
	bool has_feature_radius = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else if (arg == "--data_dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--source_map" && i + 1 < argc) {
            map_name1 = argv[++i];
        } else if (arg == "--source_contour_level" && i + 1 < argc) {
            threshold1 = stof(argv[++i]);
        } else if (arg == "--target_map" && i + 1 < argc) {
            map_name2 = argv[++i];
        } else if (arg == "--target_contour_level" && i + 1 < argc) {
            threshold2 = stof(argv[++i]);
        } else if (arg == "--source_pdb" && i + 1 < argc) {
            has_source_pdb = true;
            source_pdb = data_dir+"/"+argv[++i];
        } else if (arg == "--source_sup_pdb" && i + 1 < argc) {
            has_sup_pdb = true;
            sup_pdb = data_dir+"/"+argv[++i];
        } else if (arg == "--voxel_size" && i + 1 < argc) {
			has_voxel_size = true;
			voxel_size = stof(argv[++i]);
		} else if (arg == "--feature_radius" && i + 1 < argc) {
			has_feature_radius = true;
			feature_radius = stof(argv[++i]);
		} else if (arg == "--alg_type" && i + 1 < argc) {
            alg_type = argv[++i];
        } else {
            cout << "Error: Invalid argument " << arg << endl;
            print_help();
            return 1;
        }
    }

    if (data_dir.empty() || map_name1.empty() || map_name2.empty() || alg_type.empty()) {
        cout << "Error: Missing required arguments." << endl;
        print_help();
        return 1;
    }

  
    Sample_cluster(data_dir,map_name1, threshold1,has_voxel_size ? voxel_size : 5.0);
	Sample_cluster(data_dir,map_name2, threshold2,has_voxel_size ? voxel_size : 5.0);
	std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << voxel_size; // 设置小数点后两位
    std::string voxel_size_str = oss.str();
	std::string mrc_inputname1 = data_dir + "/" + map_name1;
	std::string before_map1;
	if (map_name1.find('-') == std::string::npos) {
		before_map1 = map_name1.substr(map_name1.rfind('_') + 1);
		before_map1 = before_map1.substr(0, before_map1.find(".map"));
	}else{
		std::string sub_map1 = map_name1.substr(map_name1.find('-') + 1, map_name1.find('_') - map_name1.find('-') - 1);
		before_map1 = sub_map1.substr(0, sub_map1.find(".map"));
	}

    std::string file_outputname1 = data_dir + "/" + "Points_" + before_map1 + "_" + voxel_size_str + "_Key.xyz";
    std::string sample_file1 = data_dir + "/" + map_name1.substr(0, map_name1.size() - 4) + "_" + voxel_size_str + ".txt";
	
	std::string mrc_inputname2 = data_dir + "/" + map_name2;
	std::string before_map2;
	if (map_name2.find('-') == std::string::npos) {
		before_map2 = map_name2.substr(map_name2.rfind('_') + 1);
		before_map2 = before_map2.substr(0, before_map2.find(".map"));
	}else{
		std::string sub_map2 = map_name2.substr(map_name2.find('-') + 1, map_name2.find('_') - map_name2.find('-') - 1);
		before_map2 = sub_map2.substr(0, sub_map2.find(".map"));
	}
    std::string file_outputname2 = data_dir + "/" + "Points_" + before_map2 + "_" + voxel_size_str +"_Key.xyz";
    std::string sample_file2 = data_dir + "/" + map_name2.substr(0, map_name2.size() - 4) + "_" + voxel_size_str + ".txt";
    std::string mapnum1 = map_name1.erase(map_name1.find_last_of("."));
    std::string mapnum2 = map_name2.erase(map_name2.find_last_of("."));
    std::string outnum = mapnum1+"_"+mapnum2;
	
	if (alg_type == "global") {
        direct_alignment(data_dir, file_outputname1,file_outputname2,sample_file1,sample_file2, has_source_pdb ? source_pdb : std::nullopt, has_sup_pdb ? sup_pdb : std::nullopt,has_voxel_size ? voxel_size : 5.0,has_feature_radius ? feature_radius : 7.0,outnum);
    } else if (alg_type == "mask") {
        mask_alignment(data_dir, file_outputname1,file_outputname2,sample_file1,sample_file2, has_source_pdb ? source_pdb : std::nullopt, has_sup_pdb ? sup_pdb : std::nullopt,has_voxel_size ? voxel_size : 5.0,has_feature_radius ? feature_radius : 7.0,outnum);
    }
	
    return 0;
}
