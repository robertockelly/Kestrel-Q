#ifndef KQ_CUDA_H
#define KQ_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Runs the CUDA device, transfer, and kernel-launch smoke validation. */
int kq_cuda_smoke(void);

#ifdef __cplusplus
}
#endif

#endif
