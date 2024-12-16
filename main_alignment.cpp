#include "alignment/Registration.h"
#include <cstdlib> // 引入 system() 函数所需的头文件
#include <cstdio>  // 引入 sprintf() 函数所需的头文件
#include <cmath>
using namespace std;

void direct_alignment(const std::string& temp_path, const std::string& source_key_dir,const std::string& target_key_dir,const std::string& source_sample_dir, const std::string& target_sample_dir,std::optional<std::string> source_pdb, std::optional<std::string> sup_pdb,float voxel_size,float feature_radius,const std::string& outnum){
	Registration dres;
	Eigen::Matrix4d T;

    if (source_pdb && sup_pdb) {
        T = dres.Registration_given_feature(temp_path, source_key_dir, target_key_dir, source_sample_dir, target_sample_dir, source_pdb.value(), sup_pdb.value(), voxel_size,feature_radius, outnum);
    } else {
        T = dres.Registration_given_feature(temp_path, source_key_dir, target_key_dir, source_sample_dir, target_sample_dir, "", "", voxel_size,feature_radius, outnum);
    }
	
	std::cout << "Estimated transformation: \n" << T << std::endl;
}
void mask_alignment(const std::string& temp_path, const std::string& source_key_dir,const std::string& target_key_dir,const std::string& source_sample_dir, const std::string& target_sample_dir,std::optional<std::string> source_pdb, std::optional<std::string> sup_pdb,float voxel_size,float feature_radius, const std::string& outnum){
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
    std::cout << "Usage: CryoAlign_alignment [data dir] [source_xyz] [target_xyz] [source_sample]" << std::endl;
    std::cout << "[target_samplel] [source.pdb] [source sup.pdb] [voxel_size] [feature_radius] [alg_type]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_dir: Map file path." << std::endl;
    std::cout << "  --source_xyz: Source map keypoints file." << std::endl;
    std::cout << "  --target_xyz: Target map keypoints file" << std::endl;
    std::cout << "  --source_sample: Source map sample file." << std::endl;
    std::cout << "  --target_sample: Target map sample file." << std::endl;
    std::cout << "  --source_pdb(optional): Source pdb name." << std::endl;
    std::cout << "  --source_sup_pdb(optional): Transformed source pdb name (ground truth)." << std::endl;
	std::cout << "  --voxel_size: Sampling interval.(defaults 5.0)" << std::endl;
	std::cout << "  --feature_radius: Radius for feature construction.(defaults 7.0)" << std::endl;
    std::cout << "  --alg_type: Global_alignment or Mask_alignment." << std::endl;
	std::cout << std::endl;
	std::cout << "Example:" << std::endl;
	std::cout << "  For Global_alignment:" << std::endl;
	std::cout << "  CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type global" << std::endl;
	std::cout << "  For Mask_alignment:" << std::endl;
	std::cout << "  CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type mask" << std::endl;
}
int main(int argc, char* argv[]) {

    // Initialize variables with default values or empty strings
    string data_dir;
    string source_xyz;
    string target_xyz;
    string source_sample;
    string target_sample;
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
        } else if (arg == "--source_xyz" && i + 1 < argc) {
            source_xyz = argv[++i];
        } else if (arg == "--target_xyz" && i + 1 < argc) {
            target_xyz = argv[++i];
        } else if (arg == "--source_sample" && i + 1 < argc) {
            source_sample = argv[++i];
        } else if (arg == "--target_sample" && i + 1 < argc) {
            target_sample = argv[++i];
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

    if (data_dir.empty() || source_xyz.empty() || target_xyz.empty() || alg_type.empty()|| source_sample.empty()||target_sample.empty()) {
        cout << "Error: Missing required arguments." << endl;
        print_help();
        return 1;
    }
    std::string source_xyz1 = data_dir+"/"+source_xyz;
	  std::string target_xyz1 = data_dir+"/"+target_xyz;
	  std::string source_sample1 = data_dir+"/"+source_sample;
	  std::string target_sample1 = data_dir+"/"+target_sample;
    
    std::string mapnum1 = source_sample.erase(source_sample.find_last_of("."));
    std::string mapnum2 = target_sample.erase(target_sample.find_last_of("."));
    std::string outnum = mapnum1+"_"+mapnum2;
	
	
  
	
	
	if (alg_type == "global") {
        direct_alignment(data_dir, source_xyz1,target_xyz1,source_sample1,target_sample1, has_source_pdb ? source_pdb : std::nullopt, has_sup_pdb ? sup_pdb : std::nullopt,has_voxel_size ? voxel_size : 5.0,has_feature_radius ? feature_radius : 7.0,outnum);
    } else if (alg_type == "mask") {
        mask_alignment(data_dir, source_xyz1,target_xyz1,source_sample1,target_sample1, has_source_pdb ? source_pdb : std::nullopt, has_sup_pdb ? sup_pdb : std::nullopt,has_voxel_size ? voxel_size : 5.0,has_feature_radius ? feature_radius : 7.0,outnum);
    }
	
    return 0;
}
