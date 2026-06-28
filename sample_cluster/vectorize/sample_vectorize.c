#include "sample_vectorize.h"

#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "mrc.h"

CMD cmd;

static void free_mrc_buffers(MRC *mrc) {
    if (mrc == NULL) {
        return;
    }

    const int ndata = mrc->xdim > 0 && mrc->ydim > 0 && mrc->zdim > 0
        ? mrc->xdim * mrc->ydim * mrc->zdim
        : 0;

    if (mrc->vec != NULL) {
        for (int i = 0; i < ndata; ++i) {
            free(mrc->vec[i]);
        }
        free(mrc->vec);
        mrc->vec = NULL;
    }

    if (mrc->xyz != NULL) {
        for (int i = 0; i < ndata; ++i) {
            free(mrc->xyz[i]);
        }
        free(mrc->xyz);
        mrc->xyz = NULL;
    }

    free(mrc->dens);
    mrc->dens = NULL;
}

static int run_sample_vectorize(const char *map_file,
                                double threshold,
                                double voxel_size,
                                int num_threads) {
    if (map_file == NULL || voxel_size <= 0.0) {
        return 1;
    }

    memset(&cmd, 0, sizeof(cmd));
    strncpy(cmd.file, map_file, LIN - 1);
    cmd.Nthr = num_threads > 0 ? num_threads : 2;
    cmd.dreso = 16.0;
    cmd.th1 = threshold;
    cmd.ssize = voxel_size;
    cmd.Mode = 0;

    const int max_threads = omp_get_num_procs();
    omp_set_num_threads(cmd.Nthr < max_threads ? cmd.Nthr : max_threads);

    MRC source;
    MRC sampled;
    memset(&source, 0, sizeof(source));
    memset(&sampled, 0, sizeof(sampled));

    int ret = 0;
    if (readmrc(&source, cmd.file)) {
        ret = 1;
        goto cleanup;
    }

    SetUpVoxSize(&source, &sampled, cmd.th1, cmd.ssize);
    if (fastVEC(&source, &sampled)) {
        ret = 1;
        goto cleanup;
    }

cleanup:
    free_mrc_buffers(&sampled);
    free_mrc_buffers(&source);
    malloc_trim(0);
    return ret;
}

int sample_vectorize_to_file(const char *map_file,
                             double threshold,
                             double voxel_size,
                             const char *output_file,
                             int num_threads) {
    if (output_file == NULL) {
        return 1;
    }

    FILE *out = fopen(output_file, "w");
    if (out == NULL) {
        fprintf(stderr, "Cannot open sample output file %s: %s\n", output_file, strerror(errno));
        return 1;
    }

    fflush(stdout);
    const int saved_stdout = dup(fileno(stdout));
    if (saved_stdout == -1) {
        fprintf(stderr, "Cannot duplicate stdout: %s\n", strerror(errno));
        fclose(out);
        return 1;
    }

    if (dup2(fileno(out), fileno(stdout)) == -1) {
        fprintf(stderr, "Cannot redirect stdout to %s: %s\n", output_file, strerror(errno));
        close(saved_stdout);
        fclose(out);
        return 1;
    }

    const int ret = run_sample_vectorize(map_file, threshold, voxel_size, num_threads);

    fflush(stdout);
    if (dup2(saved_stdout, fileno(stdout)) == -1) {
        fprintf(stderr, "Cannot restore stdout: %s\n", strerror(errno));
    }
    close(saved_stdout);
    fclose(out);

    return ret;
}
