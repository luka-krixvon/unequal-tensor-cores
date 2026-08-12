// cublaslt_int8_probe.cu — does NVIDIA's own GEMM library expose an INT8
// path on this GPU, and at what attained rate?
//
// Probes cublasLtMatmul with CUDA_R_8I inputs / CUDA_R_32I output
// (CUBLAS_COMPUTE_32I). Three possible outcomes, all paper-relevant:
//   SUPPORTED + rate   -> NVIDIA's library serves INT8 here (record TOPS)
//   NOT_SUPPORTED      -> NVIDIA's own library declines INT8 on this arch
//   heuristic empty    -> no algorithm available (effectively unsupported)
//
// Build: nvcc -O3 -arch=native cublaslt_int8_probe.cu -lcublasLt -o lt_probe

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("RESULT cublaslt_int8=CUDA_ERROR (%s)\n",cudaGetErrorString(e)); exit(0);} }while(0)
#define CKL(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("RESULT cublaslt_int8=UNSUPPORTED (status=%d at line %d)\n",(int)s,__LINE__); exit(0);} }while(0)

int main() {
  const int M=4096, N=4096, K=4096;   // healthy square GEMM
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device=%s cc=%d.%d\n", p.name, p.major, p.minor);

  cublasLtHandle_t lt; CKL(cublasLtCreate(&lt));

  int8_t *A,*B; int32_t *C;
  CK(cudaMalloc(&A,(size_t)M*K)); CK(cudaMalloc(&B,(size_t)K*N));
  CK(cudaMalloc(&C,(size_t)M*N*4));
  CK(cudaMemset(A,1,(size_t)M*K)); CK(cudaMemset(B,1,(size_t)K*N));

  cublasLtMatmulDesc_t op;
  CKL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  cublasOperation_t tA=CUBLAS_OP_T, tB=CUBLAS_OP_N;  // int8 GEMM canonical TN
  CKL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tA,sizeof(tA)));
  CKL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tB,sizeof(tB)));

  cublasLtMatrixLayout_t la,lb,lc;
  CKL(cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K));   // A^T: K x M, ld=K
  CKL(cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,K,N,K));
  CKL(cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,M,N,M));

  cublasLtMatmulPreference_t pref; CKL(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws=64<<20; void* dws; CK(cudaMalloc(&dws,ws));
  CKL(cublasLtMatmulPreferenceSetAttribute(pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));

  cublasLtMatmulHeuristicResult_t h; int nres=0;
  cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pref,1,&h,&nres);
  if (hs!=CUBLAS_STATUS_SUCCESS || nres==0) {
    printf("RESULT cublaslt_int8=NO_ALGORITHM (status=%d nres=%d)\n",(int)hs,nres);
    return 0;
  }

  int32_t alpha=1, beta=0;
  // warmup + timing
  for(int i=0;i<3;i++)
    CKL(cublasLtMatmul(lt,op,&alpha,A,la,B,lb,&beta,C,lc,C,lc,&h.algo,dws,ws,0));
  CK(cudaDeviceSynchronize());
  cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  const int reps=50;
  CK(cudaEventRecord(t0));
  for(int i=0;i<reps;i++)
    CKL(cublasLtMatmul(lt,op,&alpha,A,la,B,lb,&beta,C,lc,C,lc,&h.algo,dws,ws,0));
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  float ms=0; CK(cudaEventElapsedTime(&ms,t0,t1));
  double tops = 2.0*M*N*K*reps/(ms*1e-3)/1e12;
  printf("RESULT cublaslt_int8=SUPPORTED attained_TOPS=%.1f (M=N=K=%d, %d reps)\n",tops,M,reps);
  return 0;
}
