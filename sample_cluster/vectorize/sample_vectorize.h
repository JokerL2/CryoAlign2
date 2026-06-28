#ifndef SAMPLE_VECTORIZE_H
#define SAMPLE_VECTORIZE_H

#ifdef __cplusplus
extern "C" {
#endif

int sample_vectorize_to_file(const char *map_file,
                             double threshold,
                             double voxel_size,
                             const char *output_file,
                             int num_threads);

#ifdef __cplusplus
}
#endif

#endif
