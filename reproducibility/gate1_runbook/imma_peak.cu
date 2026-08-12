// imma_peak.cu — warp-level tensor-core peak-rate microbenchmark.
// Measures attained INT8 IMMA throughput (mma.m16n8k32.s8) with a BF16 arm
// (mma.m16n8k16.bf16) as an on-die reference ratio.
//
// Register-resident by design: no memory traffic, so the number approximates
// the instruction-issue ceiling of the integer tensor pipe. Four independent
// accumulator chains per warp provide enough ILP to saturate the pipe rather
// than measure dependent-issue latency.
//
// Ops accounting (per warp-level instruction):
//   s8  m16n8k32: 16*8*32 MACs = 4096 MACs = 8192 int ops
//   bf16 m16n8k16: 16*8*16 MACs = 2048 MACs = 4096 flops
//
// Build: nvcc -O3 -arch=<sm_89|sm_103a|native> imma_peak.cu -o imma_peak
// Run:   ./imma_peak [iters_per_thread=200000] [blocks_per_sm=8]

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} } while(0)

// ---------------- INT8: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ---
__global__ void imma_s8_kernel(int iters, int* sink) {
  unsigned a0=threadIdx.x+1, a1=threadIdx.x+2, a2=threadIdx.x+3, a3=threadIdx.x+4;
  unsigned b0=threadIdx.x+5, b1=threadIdx.x+6;
  int c0[4]={0,1,2,3}, c1[4]={1,2,3,4}, c2[4]={2,3,4,5}, c3[4]={3,4,5,6};
  #pragma unroll 4
  for (int i=0;i<iters;i++) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c0[0]),"+r"(c0[1]),"+r"(c0[2]),"+r"(c0[3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c1[0]),"+r"(c1[1]),"+r"(c1[2]),"+r"(c1[3])
      : "r"(a1),"r"(a2),"r"(a3),"r"(a0),"r"(b1),"r"(b0));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c2[0]),"+r"(c2[1]),"+r"(c2[2]),"+r"(c2[3])
      : "r"(a2),"r"(a3),"r"(a0),"r"(a1),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c3[0]),"+r"(c3[1]),"+r"(c3[2]),"+r"(c3[3])
      : "r"(a3),"r"(a0),"r"(a1),"r"(a2),"r"(b1),"r"(b0));
  }
  if (threadIdx.x==0 && blockIdx.x==0)
    sink[0]=c0[0]+c1[1]+c2[2]+c3[3];
}

// ---------------- BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
__global__ void hmma_bf16_kernel(int iters, float* sink) {
  unsigned a0=threadIdx.x+1, a1=threadIdx.x+2, a2=threadIdx.x+3, a3=threadIdx.x+4;
  unsigned b0=threadIdx.x+5, b1=threadIdx.x+6;
  float c0[4]={0,1,2,3}, c1[4]={1,2,3,4}, c2[4]={2,3,4,5}, c3[4]={3,4,5,6};
  #pragma unroll 4
  for (int i=0;i<iters;i++) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c0[0]),"+f"(c0[1]),"+f"(c0[2]),"+f"(c0[3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c1[0]),"+f"(c1[1]),"+f"(c1[2]),"+f"(c1[3])
      : "r"(a1),"r"(a2),"r"(a3),"r"(a0),"r"(b1),"r"(b0));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c2[0]),"+f"(c2[1]),"+f"(c2[2]),"+f"(c2[3])
      : "r"(a2),"r"(a3),"r"(a0),"r"(a1),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c3[0]),"+f"(c3[1]),"+f"(c3[2]),"+f"(c3[3])
      : "r"(a3),"r"(a0),"r"(a1),"r"(a2),"r"(b1),"r"(b0));
  }
  if (threadIdx.x==0 && blockIdx.x==0)
    sink[0]=c0[0]+c1[1]+c2[2]+c3[3];
}

template <typename K, typename S>
static double run_kernel(K kernel, int iters, int blocks, int threads,
                         double ops_per_mma, S* sink, int repeats) {
  // warmup
  kernel<<<blocks,threads>>>(iters/10, sink);
  CHECK(cudaDeviceSynchronize());
  double best_tops=0;
  for (int r=0;r<repeats;r++) {
    cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    kernel<<<blocks,threads>>>(iters, sink);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));
    double warps=(double)blocks*threads/32.0;
    double mmas=warps*(double)iters*4.0;           // 4 independent chains
    double tops=mmas*ops_per_mma/(ms*1e-3)/1e12;
    if (tops>best_tops) best_tops=tops;
    CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
  }
  return best_tops;
}

int main(int argc, char** argv) {
  int iters = argc>1 ? atoi(argv[1]) : 200000;
  int bps   = argc>2 ? atoi(argv[2]) : 8;

  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p,0));
  int blocks = p.multiProcessorCount*bps, threads = 256;
  int clk_khz=0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  printf("device=%s sm_count=%d cc=%d.%d clock_khz=%d\n",
         p.name,p.multiProcessorCount,p.major,p.minor,clk_khz);
  printf("config: blocks=%d threads=%d iters=%d chains=4 repeats=5\n",
         blocks,threads,iters);

  int* dsink_i;  CHECK(cudaMalloc(&dsink_i,sizeof(int)));
  float* dsink_f; CHECK(cudaMalloc(&dsink_f,sizeof(float)));

  double int8_tops = run_kernel(imma_s8_kernel,  iters, blocks, threads, 8192.0, dsink_i, 5);
  double bf16_tflops = run_kernel(hmma_bf16_kernel, iters, blocks, threads, 4096.0, dsink_f, 5);

  printf("RESULT int8_imma_attained_TOPS=%.1f\n", int8_tops);
  printf("RESULT bf16_hmma_attained_TFLOPS=%.1f\n", bf16_tflops);
  printf("RESULT int8_to_bf16_ratio=%.3f\n", int8_tops/bf16_tflops);
  // Interpretation key (spec, dense): H200 int8:bf16 = 2.0 ; B200 = 2.05 ;
  // B300 spec int8:bf16 = 150/2200 = 0.068. A measured ratio near 2 on sm_89/sm_90
  // validates the harness; the B300 value is the paper's target measurement.
  return 0;
}
