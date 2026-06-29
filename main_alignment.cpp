#include "alignment/Registration.h"
#include "MpiContext.h"

#include <iostream>
#include <optional>
#include <string>

using namespace std;

namespace {

std::string without_extension(const std::string& name) {
    size_t pos = name.find_last_of('.');
    return pos == std::string::npos ? name : name.substr(0, pos);
}

}

void direct_alignment(const std::string& temp_path,
                      const std::string& source_key_dir,
                      const std::string& target_key_dir,
                      const std::string& source_sample_dir,
                      const std::string& target_sample_dir,
                      std::optional<std::string> source_pdb,
                      std::optional<std::string> sup_pdb,
                      float voxel_size,
                      float feature_radius,
                      const std::string& outnum,
                      const ScoreConfig& score_config) {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank != 0) return;

    Registration dres(score_config);
    Eigen::Matrix4d T;
    if (source_pdb && sup_pdb) {
        T = dres.Registration_given_feature(temp_path, source_key_dir, target_key_dir, source_sample_dir, target_sample_dir, source_pdb.value(), sup_pdb.value(), voxel_size, feature_radius, outnum);
    } else {
        T = dres.Registration_given_feature(temp_path, source_key_dir, target_key_dir, source_sample_dir, target_sample_dir, "", "", voxel_size, feature_radius, outnum);
    }
    std::cout << "Estimated transformation: \n" << T << std::endl;
}

void mask_alignment(const std::string& temp_path,
                    const std::string& source_key_dir,
                    const std::string& target_key_dir,
                    const std::string& source_sample_dir,
                    const std::string& target_sample_dir,
                    std::optional<std::string> source_pdb,
                    std::optional<std::string> sup_pdb,
                    float voxel_size,
                    float feature_radius,
                    const std::string& outnum,
                    const ScoreConfig& score_config) {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Registration mres(score_config);
    mres.Registration_mask_list(temp_path, source_key_dir, target_key_dir, source_sample_dir, target_sample_dir, voxel_size, feature_radius, outnum);

    if (mpi_rank == 0) {
        std::string record_dir = temp_path + "/" + outnum + "_record_cpp.txt";
        std::string record_T_dir = temp_path + "/" + outnum + "_record_T_cpp.csv";
        std::string save_dir = temp_path + "/" + outnum + "_extract_top_10.txt";
        if (source_pdb && sup_pdb) {
            mres.extract_top_K(record_dir, record_T_dir, 10, save_dir, source_pdb.value(), sup_pdb.value());
        } else {
            mres.extract_top_K(record_dir, record_T_dir, 10, save_dir, "", "");
        }
    }
}

void print_help() {
    std::cout << "Usage: CryoAlign_alignment --data_dir DIR --source_xyz XYZ --target_xyz XYZ "
              << "--source_sample TXT --target_sample TXT [--source_pdb PDB] "
              << "[--source_sup_pdb PDB] [--voxel_size 5.0] [--feature_radius 7.0] "
              << "--alg_type global|mask [--score_mode single|multi] "
              << "[--normal_weight 0.25 --distance_weight 0.25 "
              << "--density_weight 0.25 --shot_weight 0.25]" << std::endl;
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
    std::cout << "  --score_mode: single or multi (default: single)." << std::endl;
    std::cout << "  --normal_weight: Normal consistency weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  --distance_weight: Point distance weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  --density_weight: Local geometric density weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  --shot_weight: SHOT similarity weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  Multi-mode weights must be in [0, 1] and sum to 1." << std::endl;
	std::cout << std::endl;
	std::cout << "Example:" << std::endl;
	std::cout << "  For Global_alignment:" << std::endl;
	std::cout << "  CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type global" << std::endl;
	std::cout << "  For Mask_alignment:" << std::endl;
	std::cout << "  CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type mask" << std::endl;
    std::cout << std::endl;
    std::cout << "Scoring:" << std::endl;
    std::cout << "  single: normal consistency only (default)." << std::endl;
    std::cout << "  multi: weighted normal consistency, point distance similarity, local geometric density similarity, and SHOT feature similarity." << std::endl;
    std::cout << "  Multi-mode weights must be in [0, 1] and sum to 1. Both score modes support global and mask." << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  Global alignment with the single score:" << std::endl;
    std::cout << "  ./CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt --voxel_size 5.0 --feature_radius 7.0 --alg_type global --score_mode single" << std::endl;
    std::cout << "  Mask alignment with the multidimensional score:" << std::endl;
    std::cout << "  ./CryoAlign_alignment --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_xyz Points_3695_5.00_Key.xyz --target_xyz Points_3696_5.00_Key.xyz --source_sample EMD-3695_5.00.txt --target_sample EMD-3696_5.00.txt --voxel_size 5.0 --feature_radius 7.0 --alg_type mask --score_mode multi --normal_weight 0.4 --distance_weight 0.2 --density_weight 0.2 --shot_weight 0.2" << std::endl;
    std::cout << "  Mask alignment with MPI (4 ranks):" << std::endl;
    std::cout << "  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -np 4 ./CryoAlign_alignment --data_dir ../../example_dataset/emd_3661_emd_6647/ --source_xyz Points_3661_Key.xyz --target_xyz Points_6647_5.00_Key.xyz --source_sample emd_3661_5.00.txt --target_sample emd_6647_5.00.txt --voxel_size 5.0 --feature_radius 7.0 --alg_type mask --score_mode single" << std::endl;
    std::cout << "GPU is used only by CryoAlign and CryoAlign_extract_keypoints, not by CryoAlign_alignment." << std::endl;
}

int main(int argc, char* argv[]) {
    MpiContext mpi(&argc, &argv);

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
    ScoreConfig score_config;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            if (mpi.is_root()) print_help();
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
            source_pdb = data_dir + "/" + argv[++i];
        } else if (arg == "--source_sup_pdb" && i + 1 < argc) {
            has_sup_pdb = true;
            sup_pdb = data_dir + "/" + argv[++i];
        } else if (arg == "--voxel_size" && i + 1 < argc) {
            has_voxel_size = true;
            voxel_size = stof(argv[++i]);
        } else if (arg == "--feature_radius" && i + 1 < argc) {
            has_feature_radius = true;
            feature_radius = stof(argv[++i]);
        } else if (arg == "--alg_type" && i + 1 < argc) {
            alg_type = argv[++i];
        } else if (arg == "--score_mode" && i + 1 < argc) {
            const string mode = argv[++i];
            if (mode == "single") {
                score_config.mode = ScoreMode::Single;
            } else if (mode == "multi") {
                score_config.mode = ScoreMode::Multi;
            } else {
                if (mpi.is_root()) {
                    cout << "Error: --score_mode must be single or multi." << endl;
                }
                return 1;
            }
        } else if (arg == "--normal_weight" && i + 1 < argc) {
            score_config.weights.normal = stod(argv[++i]);
        } else if (arg == "--distance_weight" && i + 1 < argc) {
            score_config.weights.distance = stod(argv[++i]);
        } else if (arg == "--density_weight" && i + 1 < argc) {
            score_config.weights.density = stod(argv[++i]);
        } else if (arg == "--shot_weight" && i + 1 < argc) {
            score_config.weights.shot = stod(argv[++i]);
        } else {
            if (mpi.is_root()) {
                cout << "Error: Invalid argument " << arg << endl;
                print_help();
            }
            return 1;
        }
    }

    if (data_dir.empty() || source_xyz.empty() || target_xyz.empty() || alg_type.empty() || source_sample.empty() || target_sample.empty()) {
        if (mpi.is_root()) {
            cout << "Error: Missing required arguments." << endl;
            print_help();
        }
        return 1;
    }
    std::string score_error;
    if (!score_config.Validate(&score_error)) {
        if (mpi.is_root()) {
            cout << "Error: " << score_error << endl;
        }
        return 1;
    }

    std::string source_xyz_path = data_dir + "/" + source_xyz;
    std::string target_xyz_path = data_dir + "/" + target_xyz;
    std::string source_sample_path = data_dir + "/" + source_sample;
    std::string target_sample_path = data_dir + "/" + target_sample;
    std::string outnum = without_extension(source_sample) + "_" + without_extension(target_sample);
    const float effective_voxel_size = has_voxel_size ? voxel_size : 5.0f;
    const float effective_feature_radius = has_feature_radius ? feature_radius : 7.0f;

    if (alg_type == "global") {
        direct_alignment(data_dir, source_xyz_path, target_xyz_path, source_sample_path, target_sample_path,
                         has_source_pdb ? source_pdb : std::nullopt,
                         has_sup_pdb ? sup_pdb : std::nullopt,
                         effective_voxel_size, effective_feature_radius, outnum, score_config);
    } else if (alg_type == "mask") {
        mask_alignment(data_dir, source_xyz_path, target_xyz_path, source_sample_path, target_sample_path,
                       has_source_pdb ? source_pdb : std::nullopt,
                       has_sup_pdb ? sup_pdb : std::nullopt,
                       effective_voxel_size, effective_feature_radius, outnum, score_config);
    } else {
        if (mpi.is_root()) {
            cout << "Error: Invalid alg_type " << alg_type << endl;
            print_help();
        }
        return 1;
    }

    return 0;
}
