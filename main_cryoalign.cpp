#include "sample_cluster/VoxEM.h"
#include "alignment/Registration.h"
#include "sample_cluster/vectorize/sample_vectorize.h"
#include "MpiContext.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

using namespace std;

namespace {

std::string map_prefix(const std::string& map_name) {
    if (map_name.find('-') == std::string::npos) {
        std::string before_map = map_name.substr(map_name.rfind('_') + 1);
        return before_map.substr(0, before_map.find(".map"));
    }
    std::string sub_map = map_name.substr(map_name.find('-') + 1, map_name.find('_') - map_name.find('-') - 1);
    return sub_map.substr(0, sub_map.find(".map"));
}

std::string without_extension(const std::string& name) {
    size_t pos = name.find_last_of('.');
    return pos == std::string::npos ? name : name.substr(0, pos);
}

std::string voxel_size_string(float voxel_size) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << voxel_size;
    return oss.str();
}

}

void Sample_cluster(const std::string& data_dir, const std::string& map_name, float threshold, float voxel_size) {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    VoxEM voxem;
    std::array<float, 3> shifting = {0.0f, 0.0f, 0.0f};
    std::string voxel_size_str = voxel_size_string(voxel_size);
    std::string mrc_inputname = data_dir + "/" + map_name;
    std::string before_map = map_prefix(map_name);
    std::string file_outputname = data_dir + "/" + "Points_" + before_map + "_" + voxel_size_str + "_Key.xyz";
    std::string sample_file = data_dir + "/" + without_extension(map_name) + "_" + voxel_size_str + ".txt";

    int sample_status = 0;
    if (mpi_rank == 0) {
        std::filesystem::remove(file_outputname);
        std::filesystem::remove(sample_file);

        std::cout << "MRC file: " << mrc_inputname << std::endl;
        std::cout << "Sub map: " << before_map << std::endl;
        std::cout << "Output file: " << file_outputname << std::endl;
        std::cout << "Sample file: " << sample_file << std::endl;

        sample_status = sample_vectorize_to_file(mrc_inputname.c_str(), threshold, voxel_size, sample_file.c_str(), 2);
        if (sample_status != 0) {
            std::cerr << "Built-in Sample failed" << std::endl;
        }
    }
    MPI_Bcast(&sample_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (sample_status != 0) {
        return;
    }

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
    std::array<float, 3> vox_ang = {
        vox_length[0] / static_cast<float>(vox.size()),
        vox_length[1] / static_cast<float>(vox[0].size()),
        vox_length[2] / static_cast<float>(vox[0][0].size())
    };

    std::vector<std::array<float, 3>> sample_points;
    std::vector<std::array<float, 3>> sample_vector;
    std::vector<std::array<float, 1>> density_list;
    std::tie(sample_points, sample_vector, density_list) = voxem.load_sample_points(sample_file, false);

    shifting = {
        start[0] * vox_ang[0] + origin[0],
        start[1] * vox_ang[1] + origin[1],
        start[2] * vox_ang[2] + origin[2]
    };
    if (mpi_rank == 0) {
        std::cout << "Shifting: [" << shifting[0] << ", " << shifting[1] << ", " << shifting[2] << "]" << std::endl;
    }
    for (auto& point : sample_points) {
        point[0] -= shifting[0];
        point[1] -= shifting[1];
        point[2] -= shifting[2];
    }
    voxem.setPointWorkspaceValue("sample", sample_points);
    if (mpi_rank == 0) {
        const size_t sample_cols = sample_points.empty() ? 0 : sample_points[0].size();
        std::cout << "sample_points_transposed shape: (" << sample_points.size() << ", " << sample_cols << ")" << std::endl;
    }

    voxem.Point_Create_Meanshift_sample(
        threshold,
        17,
        3.0,
        2000,
        0.05,
        0.000187,
        "Voxel",
        "Meanshift",
        "sample"
    );

    if (mpi_rank == 0) {
        voxem.Point_Transform_DBSCAN(voxel_size * 3, 3, "Meanshift", "Meanshift", false);
        voxem.Point_Transform_DBSCAN(voxel_size, 2, "Meanshift", "DBSCAN", true, 100);
        voxem.IO_WriteXYZ("DBSCAN", file_outputname, "H", shifting);
        std::cout << "Sampling and clustering success" << std::endl;
    } else {
        while (!std::filesystem::exists(file_outputname)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
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
    std::cout << "Usage: CryoAlign --data_dir DIR --source_map MAP --source_contour_level LEVEL "
              << "--target_map MAP --target_contour_level LEVEL [--source_pdb PDB] "
              << "[--source_sup_pdb PDB] [--voxel_size 5.0] [--feature_radius 7.0] "
              << "[--use_gpu|--cpu] --alg_type global|mask [--score_mode single|multi] "
              << "[--normal_weight 0.25 --distance_weight 0.25 "
              << "--density_weight 0.25 --shot_weight 0.25]" << std::endl;
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
    std::cout << "  --score_mode: single or multi (default: single)." << std::endl;
    std::cout << "  --normal_weight: Normal consistency weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  --distance_weight: Point distance weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  --density_weight: Local geometric density weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  --shot_weight: SHOT similarity weight for multi mode (default: 0.25)." << std::endl;
    std::cout << "  Multi-mode weights must be in [0, 1] and sum to 1." << std::endl;
	std::cout << std::endl;
	std::cout << "Example:" << std::endl;
	std::cout << "  For Global_alignment:" << std::endl;
	std::cout << "  CryoAlign --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_map EMD-3695.map --source_contour_level 0.008 --target_map EMD-3696.map --target_contour_level 0.002 --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type global" << std::endl;
	std::cout << "  For Mask_alignment:" << std::endl;
	std::cout << "  CryoAlign --data_dir ../../example_dataset/emd_3695_emd_3696/ --source_map EMD-3695.map --source_contour_level 0.008 --target_map EMD-3696.map --target_contour_level 0.002 --source_pdb 5nsr.pdb --source_sup_pdb 5nsr_sup.pdb --voxel_size 5.0 --feature_radius 7.0 --alg_type mask" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  CPU + MPI (4 ranks):" << std::endl;
    std::cout << "  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -np 4 ./CryoAlign --data_dir ../../example_dataset/emd_3661_emd_6647/ --source_map emd_3661.map --source_contour_level 0.07 --target_map emd_6647.map --target_contour_level 0.017 --voxel_size 5.0 --feature_radius 7.0 --alg_type mask --cpu" << std::endl;
    std::cout << "  GPU (single process):" << std::endl;
    std::cout << "  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./CryoAlign --data_dir ../../example_dataset/emd_3661_emd_6647/ --source_map emd_3661.map --source_contour_level 0.07 --target_map emd_6647.map --target_contour_level 0.017 --voxel_size 5.0 --feature_radius 7.0 --alg_type mask --use_gpu" << std::endl;
    std::cout << "  Do not launch GPU mode with multiple MPI ranks on one GPU." << std::endl;
}

int main(int argc, char* argv[]) {
    MpiContext mpi(&argc, &argv);

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
    bool request_gpu = false;
    bool force_cpu = false;
    ScoreConfig score_config;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            if (mpi.is_root()) print_help();
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
        } else if (arg == "--use_gpu" || arg == "--gpu") {
            request_gpu = true;
            force_cpu = false;
        } else if (arg == "--cpu" || arg == "--no_gpu") {
            force_cpu = true;
            request_gpu = false;
        } else {
            if (mpi.is_root()) {
                cout << "Error: Invalid argument " << arg << endl;
                print_help();
            }
            return 1;
        }
    }

    if (data_dir.empty() || map_name1.empty() || map_name2.empty() || alg_type.empty()) {
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

    const float effective_voxel_size = has_voxel_size ? voxel_size : 5.0f;
    const float effective_feature_radius = has_feature_radius ? feature_radius : 7.0f;
    const std::string voxel_size_str = voxel_size_string(effective_voxel_size);

    if (request_gpu) {
        setenv("CRYOALIGN_USE_GPU", "1", 1);
        unsetenv("CRYOALIGN_DISABLE_GPU");
    } else if (force_cpu) {
        setenv("CRYOALIGN_USE_GPU", "0", 1);
        setenv("CRYOALIGN_DISABLE_GPU", "1", 1);
    }

    Sample_cluster(data_dir, map_name1, threshold1, effective_voxel_size);
    Sample_cluster(data_dir, map_name2, threshold2, effective_voxel_size);

    std::string file_outputname1 = data_dir + "/" + "Points_" + map_prefix(map_name1) + "_" + voxel_size_str + "_Key.xyz";
    std::string sample_file1 = data_dir + "/" + without_extension(map_name1) + "_" + voxel_size_str + ".txt";
    std::string file_outputname2 = data_dir + "/" + "Points_" + map_prefix(map_name2) + "_" + voxel_size_str + "_Key.xyz";
    std::string sample_file2 = data_dir + "/" + without_extension(map_name2) + "_" + voxel_size_str + ".txt";
    std::string outnum = without_extension(map_name1) + "_" + without_extension(map_name2);

    if (alg_type == "global") {
        direct_alignment(data_dir, file_outputname1, file_outputname2, sample_file1, sample_file2,
                         has_source_pdb ? source_pdb : std::nullopt,
                         has_sup_pdb ? sup_pdb : std::nullopt,
                         effective_voxel_size, effective_feature_radius, outnum, score_config);
    } else if (alg_type == "mask") {
        mask_alignment(data_dir, file_outputname1, file_outputname2, sample_file1, sample_file2,
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
