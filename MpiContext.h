#ifndef CRYOALIGN_MPI_CONTEXT_H
#define CRYOALIGN_MPI_CONTEXT_H

#include <mpi.h>

class MpiContext {
public:
    MpiContext(int* argc, char*** argv) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(argc, argv);
            owns_mpi_ = true;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~MpiContext() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owns_mpi_ && !finalized) {
            MPI_Finalize();
        }
    }

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool is_root() const { return rank_ == 0; }

private:
    bool owns_mpi_ = false;
    int rank_ = 0;
    int size_ = 1;
};

#endif
