#include "sample_cluster/VoxEM.h"
#include "alignment/Registration.h"
#include <cstdlib> // 引入 system() 函数所需的头文件
#include <cstdio>  // 引入 sprintf() 函数所需的头文件
void Sample_cluster(const std::string& data_dir,const std::string& map_name, float threshold){
	// 使用字符串流来格式化浮点数
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << 5.0; // 设置小数点后两位
    std::string voxel_size_str = oss.str();
	std::string mrc_inputname = data_dir + "/" + map_name;
    std::string sub_map = map_name.substr(map_name.find('-') + 1, map_name.find('_') - map_name.find('-') - 1);
    std::string file_outputname = data_dir + "/" + "Points_" + sub_map + "_Key.xyz";
    std::string sample_file = data_dir + "/" + map_name.substr(0, map_name.size() - 4) + "_" + voxel_size_str + ".txt";
	
	std::cout << "MRC file: " << mrc_inputname << std::endl;
    std::cout << "Sub map: " << sub_map << std::endl;
    std::cout << "Output file: " << file_outputname << std::endl;
    std::cout << "Sample file: " << sample_file << std::endl;
	
	
	char sample_command[512];
	
    // 使用 sprintf 构建命令行字符串
    sprintf(sample_command, "../sample_cluster/Sample -a %s -t %.4f -s %.2f > %s", mrc_inputname.c_str(), threshold, 5.0, sample_file.c_str());

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
	
	voxem.Point_Transform_DBSCAN(5.0 * 3, 3,"Meanshift", "Meanshift",false);
	voxem.Point_Transform_DBSCAN(5.0, 2, "Meanshift", "DBSCAN",true,100);
	
	
	voxem.IO_WriteXYZ("DBSCAN",file_outputname,"H",shifting);
	std::cout << "Sampling and clustering success" << std::endl;
}
void direct_alignment(const std::string& temp_path, const std::string& source_key_dir,const std::string& target_key_dir,const std::string& source_sample_dir, const std::string& target_sample_dir,const std::string& source_pdb,const std::string& sup_pdb,bool visualize){
	Registration dres;
	Eigen::Matrix4d T = dres.Registration_given_feature(temp_path, source_key_dir,target_key_dir, source_sample_dir,target_sample_dir,source_pdb,sup_pdb, 5.0,visualize);
	
	std::cout << "Estimated transformation: \n" << T << std::endl;
}
void mask_alignment(const std::string& temp_path, const std::string& source_key_dir,const std::string& target_key_dir,const std::string& source_sample_dir, const std::string& target_sample_dir,const std::string& source_pdb, const std::string& sup_pdb){
	Registration mres;
	
	mres.Registration_mask_list(temp_path, source_key_dir,target_key_dir, source_sample_dir,target_sample_dir, 5.0);
	std::string record_dir = temp_path+"/record_cpp.txt";
	std::string record_T_dir = temp_path+"/record_T_cpp.csv";
	int k=10;
	std::string save_dir = temp_path+"/extract_top_10.txt";
	mres.extract_top_K(record_dir, record_T_dir, k, save_dir,source_pdb,sup_pdb);
}
int main(int argc, char* argv[]) {
	
	std::string data_dir = argv[1];
    std::string map_name1 = argv[2];
	
	float threshold1 = std::stof(argv[3]);
	
	std::string map_name2 = argv[4];
	
	float threshold2 = std::stof(argv[5]);
	
	std::string source_pdb = data_dir+"/"+ argv[6];
	
	std::string sup_pdb = data_dir+"/"+ argv[7];
	
	std::string flag = argv[8];
	
	bool visualize = argv[9];
	
    Sample_cluster(data_dir,map_name1, threshold1);
	Sample_cluster(data_dir,map_name2, threshold2);
	std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << 5.0; // 设置小数点后两位
    std::string voxel_size_str = oss.str();
	std::string mrc_inputname1 = data_dir + "/" + map_name1;
    std::string sub_map1 = map_name1.substr(map_name1.find('-') + 1, map_name1.find('_') - map_name1.find('-') - 1);
    std::string file_outputname1 = data_dir + "/" + "Points_" + sub_map1 + "_Key.xyz";
    std::string sample_file1 = data_dir + "/" + map_name1.substr(0, map_name1.size() - 4) + "_" + voxel_size_str + ".txt";
	
	std::string mrc_inputname2 = data_dir + "/" + map_name2;
    std::string sub_map2 = map_name2.substr(map_name2.find('-') + 1, map_name2.find('_') - map_name2.find('-') - 1);
    std::string file_outputname2 = data_dir + "/" + "Points_" + sub_map2 + "_Key.xyz";
    std::string sample_file2 = data_dir + "/" + map_name2.substr(0, map_name2.size() - 4) + "_" + voxel_size_str + ".txt";
	
	
	
	if (flag=="direct"){
		direct_alignment(data_dir, file_outputname1,file_outputname2,sample_file1,sample_file2,source_pdb,sup_pdb,visualize);
	}else if(flag=="mask"){
		mask_alignment(data_dir, file_outputname1,file_outputname2,sample_file1,sample_file2,source_pdb,sup_pdb);
		
	}
	
    return 0;
}
