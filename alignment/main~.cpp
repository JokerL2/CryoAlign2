#include "Registration.h"
int main(int argc, char* argv[]) {
	//py::scoped_interpreter guard{};
	
    Registration dres;
	std::string temp_path = argv[1];
	std::string source_key_dir = argv[2];
	std::string target_key_dir = argv[3];
	std::string source_sample_dir = argv[4];
	std::string target_sample_dir = argv[5];
	std::string res_path = argv[6];
	//Eigen::Matrix4d T = dres.Registration_given_feature("./emd_3695_3696", "./emd_3695_3696/test_95.xyz","./emd_3695_3696/test_96.xyz","./emd_3695_3696/emd_3695_5.00.txt", "./emd_3695_3696/emd_3696_5.00.txt", 5.0);
	//Eigen::Matrix4d T = dres.Registration_given_feature(temp_path, source_key_dir,target_key_dir,source_sample_dir, target_sample_dir,res_path,source_pdb,sup_pdb, 5.0);
	Eigen::Matrix4d T = dres.Registration_given_feature(temp_path, source_key_dir,target_key_dir,source_sample_dir, target_sample_dir,res_path, 5.0);
	
	std::cout << "Estimated transformation: \n" << T << std::endl;
	
	//Eigen::MatrixXd source_pdb = dres.getPointsFromPDB("./emd_3695_3696/5nsr.pdb");
    //Eigen::MatrixXd source_sup = dres.getPointsFromPDB("./emd_3695_3696/5nsr_sup.pdb");
	//std::cout << "Read pdbfile success! \n" << std::endl;
	//double rmsd = dres.calRMSD(source_pdb, source_sup, T);
	//std::cout << "RMSD between estiamted transformed PDB and ground truth:" << rmsd << std::endl;
	
	/*
	Registration mres;
	
	mres.Registration_mask_list("./emd_3661_6647", "./emd_3661_6647/test_61.xyz","./emd_3661_6647/test_47.xyz","./emd_3661_6647/emd_3661_5.00.txt", "./emd_3661_6647/emd_6647_5.00.txt", 5.0);
	
	
	std::string record_dir = "./emd_3661_6647/record_cpp.txt";
	std::string record_T_dir = "./emd_3661_6647/record_T_cpp.csv";
	int k=10;
	std::string save_dir = "./emd_3661_6647/extract_top_10.txt";
	std::string source_pdb_dir = "./emd_3661_6647/5no2.pdb";
    std::string source_sup_dir = "./emd_3661_6647/5no2_5juu_sup.pdb";
	mres.extract_top_K(record_dir, record_T_dir, k, save_dir);
	*/
    return 0;
}
