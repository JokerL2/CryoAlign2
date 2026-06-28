#include "sample_cluster/VoxEM.h"
#include "sample_cluster/vectorize/sample_vectorize.h"
#include "MpiContext.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using namespace std;

void Sample_cluster(const std::string& data_dir, const std::string& map_name, float threshold, float voxel_size) {
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    VoxEM voxem;
    std::array<float, 3> shifting = {0.0f, 0.0f, 0.0f};
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << voxel_size;
    std::string voxel_size_str = oss.str();
    std::string mrc_inputname = data_dir + "/" + map_name;
    std::string before_map;
    if (map_name.find('-') == std::string::npos) {
        before_map = map_name.substr(map_name.rfind('_') + 1);
        before_map = before_map.substr(0, before_map.find(".map"));
    } else {
        std::string sub_map = map_name.substr(map_name.find('-') + 1, map_name.find('_') - map_name.find('-') - 1);
        before_map = sub_map.substr(0, sub_map.find(".map"));
    }
    std::string file_outputname = data_dir + "/" + "Points_" + before_map + "_" + voxel_size_str + "_Key.xyz";
    std::string sample_file = data_dir + "/" + map_name.substr(0, map_name.size() - 4) + "_" + voxel_size_str + ".txt";

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
    MpiContext mpi(&argc, &argv);

    string data_dir;
    string map_name;
    float threshold = 0.0f;
    float voxel_size = 0.0f;
    bool has_voxel_size = false;
    bool request_gpu = false;
    bool force_cpu = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            if (mpi.is_root()) print_help();
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

    if (data_dir.empty() || map_name.empty()) {
        if (mpi.is_root()) {
            cout << "Error: Missing required arguments." << endl;
            print_help();
        }
        return 1;
    }

    if (request_gpu) {
        setenv("CRYOALIGN_USE_GPU", "1", 1);
        unsetenv("CRYOALIGN_DISABLE_GPU");
    } else if (force_cpu) {
        setenv("CRYOALIGN_USE_GPU", "0", 1);
        setenv("CRYOALIGN_DISABLE_GPU", "1", 1);
    }

    Sample_cluster(data_dir, map_name, threshold, has_voxel_size ? voxel_size : 5.0f);
    return 0;
}
