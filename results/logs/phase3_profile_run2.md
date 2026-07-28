(.venv) akashg7171_com@instance-20260718-143824:~/gpu-attention$ python3 scripts/03_profile.py
=== Phase 3 profiling driver — shape {'batch': 2, 'heads': 8, 'seq_len': 2048, 'head_dim': 64}, dtype float16 ===

ncu: /usr/local/cuda/bin/ncu
nsys: /usr/local/cuda/bin/nsys

$ sudo /usr/local/cuda/bin/ncu --set roofline --csv /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target naive
Wrote results/traces/naive_ncu.csv (==PROF== noise stripped)

$ sudo /usr/local/cuda/bin/ncu --set roofline --csv /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target triton
[diagnostic] autotuner cache: {(2048, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float16'): <triton.runtime.autotuner.Config object at 0x70f3ea86aa40>}
[diagnostic] autotuner best_config: BLOCK_M: 32, BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
Wrote results/traces/triton_ncu.csv (==PROF== noise stripped)

naive: 606 metric-rows -> 27 distinct kernel launches (3 excluded as RNG/fill noise).
Attention-relevant kernels (name, grid size, block size):
     4x  grid=(8, 16, 16)     block=(256, 1, 1)  turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_stages_32x1_tn
     4x  grid=(65536, 1, 1)   block=(128, 1, 1)  void at::vectorized_elementwise_kernel<4, at::BUnaryFunctor<c10::Half,
     4x  grid=(131072, 1, 1)  block=(128, 1, 1)  void at::unrolled_elementwise_kernel<at::direct_copy_kernel_cuda(at::T
     4x  grid=(8192, 1, 1)    block=(32, 4, 1)   void <unnamed>::softmax_warp_forward<float, float, float, 11, 0, 0, 32
     4x  grid=(65536, 1, 1)   block=(128, 1, 1)  void at::vectorized_elementwise_kernel<4, at::float16_copy_kernel_cuda
     4x  grid=(1, 16, 16)     block=(128, 1, 1)  turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn

Per-launch detail, in launch order (['Duration', 'DRAM Throughput', 'Compute (SM) Throughput', 'Memory Throughput']):
  id=   3  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1231008ns  DRAM Throughput=41.34%  Compute (SM) Throughput=33.48%  Memory Throughput=41.59%
  id=   4  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1076576ns  DRAM Throughput=86.24%  Compute (SM) Throughput=16.71%  Memory Throughput=86.24%
  id=   5  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2444928ns  DRAM Throughput=62.87%  Compute (SM) Throughput=45.25%  Memory Throughput=62.87%
  id=   6  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2385728ns  DRAM Throughput=82.98%  Compute (SM) Throughput=35.41%  Memory Throughput=82.98%
  id=   7  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1570496ns  DRAM Throughput=88.81%  Compute (SM) Throughput=11.45%  Memory Throughput=88.81%
  id=   8  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=996640ns  DRAM Throughput=66.47%  Compute (SM) Throughput=72.06%  Memory Throughput=66.47%
  id=   9  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1229888ns  DRAM Throughput=41.32%  Compute (SM) Throughput=33.47%  Memory Throughput=41.59%
  id=  10  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1076864ns  DRAM Throughput=86.37%  Compute (SM) Throughput=16.73%  Memory Throughput=86.37%
  id=  11  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2452640ns  DRAM Throughput=62.85%  Compute (SM) Throughput=45.40%  Memory Throughput=62.85%
  id=  12  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2395936ns  DRAM Throughput=83.92%  Compute (SM) Throughput=35.50%  Memory Throughput=83.92%
  id=  13  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1568192ns  DRAM Throughput=88.71%  Compute (SM) Throughput=11.46%  Memory Throughput=88.71%
  id=  14  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=992000ns  DRAM Throughput=66.58%  Compute (SM) Throughput=72.30%  Memory Throughput=66.58%
  id=  15  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1230624ns  DRAM Throughput=41.35%  Compute (SM) Throughput=33.47%  Memory Throughput=41.59%
  id=  16  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1075744ns  DRAM Throughput=86.27%  Compute (SM) Throughput=16.72%  Memory Throughput=86.27%
  id=  17  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2457632ns  DRAM Throughput=62.81%  Compute (SM) Throughput=46.02%  Memory Throughput=62.81%
  id=  18  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2377216ns  DRAM Throughput=83.79%  Compute (SM) Throughput=35.70%  Memory Throughput=83.79%
  id=  19  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1569248ns  DRAM Throughput=88.83%  Compute (SM) Throughput=11.45%  Memory Throughput=88.83%
  id=  20  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=997664ns  DRAM Throughput=66.71%  Compute (SM) Throughput=72.19%  Memory Throughput=66.71%
  id=  21  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1232384ns  DRAM Throughput=41.43%  Compute (SM) Throughput=33.46%  Memory Throughput=41.57%
  id=  22  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1076832ns  DRAM Throughput=86.27%  Compute (SM) Throughput=16.73%  Memory Throughput=86.27%
  id=  23  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2429408ns  DRAM Throughput=62.21%  Compute (SM) Throughput=45.23%  Memory Throughput=62.21%
  id=  24  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2387200ns  DRAM Throughput=83.07%  Compute (SM) Throughput=35.53%  Memory Throughput=83.07%
  id=  25  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1570304ns  DRAM Throughput=88.77%  Compute (SM) Throughput=11.45%  Memory Throughput=88.77%
  id=  26  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=999200ns  DRAM Throughput=66.70%  Compute (SM) Throughput=72.37%  Memory Throughput=66.70%

triton: 2378 metric-rows -> 105 distinct kernel launches (45 excluded as RNG/fill noise).
Attention-relevant kernels (name, grid size, block size):
    24x  grid=(32, 16, 1)     block=(128, 1, 1)  _fwd_kernel
    20x  grid=(64, 16, 1)     block=(128, 1, 1)  _fwd_kernel
     8x  grid=(16, 16, 1)     block=(256, 1, 1)  _fwd_kernel
     8x  grid=(16, 16, 1)     block=(128, 1, 1)  _fwd_kernel

Per-launch detail, in launch order (['Duration', 'DRAM Throughput', 'Compute (SM) Throughput', 'Memory Throughput']):
  id=   3  grid=(16, 16, 1)     _fwd_kernel                   Duration=53578688ns  DRAM Throughput=55.32%  Compute (SM) Throughput=14.02%  Memory Throughput=55.32%
  id=   5  grid=(16, 16, 1)     _fwd_kernel                   Duration=53613920ns  DRAM Throughput=55.32%  Compute (SM) Throughput=14.02%  Memory Throughput=55.32%
  id=   7  grid=(16, 16, 1)     _fwd_kernel                   Duration=53637376ns  DRAM Throughput=55.35%  Compute (SM) Throughput=14.04%  Memory Throughput=55.35%
  id=   9  grid=(16, 16, 1)     _fwd_kernel                   Duration=53646464ns  DRAM Throughput=55.49%  Compute (SM) Throughput=14.03%  Memory Throughput=55.49%
  id=  11  grid=(16, 16, 1)     _fwd_kernel                   Duration=53592480ns  DRAM Throughput=55.40%  Compute (SM) Throughput=14.04%  Memory Throughput=55.40%
  id=  13  grid=(16, 16, 1)     _fwd_kernel                   Duration=53539328ns  DRAM Throughput=55.25%  Compute (SM) Throughput=14.03%  Memory Throughput=55.25%
  id=  14  grid=(16, 16, 1)     _fwd_kernel                   Duration=53667744ns  DRAM Throughput=55.47%  Compute (SM) Throughput=14.05%  Memory Throughput=55.47%
  id=  16  grid=(16, 16, 1)     _fwd_kernel                   Duration=53681056ns  DRAM Throughput=55.54%  Compute (SM) Throughput=14.03%  Memory Throughput=55.54%
  id=  17  grid=(32, 16, 1)     _fwd_kernel                   Duration=41618080ns  DRAM Throughput=49.47%  Compute (SM) Throughput=16.09%  Memory Throughput=49.47%
  id=  19  grid=(32, 16, 1)     _fwd_kernel                   Duration=41685312ns  DRAM Throughput=49.54%  Compute (SM) Throughput=16.15%  Memory Throughput=49.54%
  id=  21  grid=(32, 16, 1)     _fwd_kernel                   Duration=41527200ns  DRAM Throughput=49.32%  Compute (SM) Throughput=16.10%  Memory Throughput=49.32%
  id=  23  grid=(32, 16, 1)     _fwd_kernel                   Duration=42157664ns  DRAM Throughput=49.12%  Compute (SM) Throughput=16.08%  Memory Throughput=49.12%
  id=  25  grid=(32, 16, 1)     _fwd_kernel                   Duration=42126816ns  DRAM Throughput=49.00%  Compute (SM) Throughput=16.10%  Memory Throughput=49.00%
  id=  27  grid=(32, 16, 1)     _fwd_kernel                   Duration=41646400ns  DRAM Throughput=49.58%  Compute (SM) Throughput=16.13%  Memory Throughput=49.58%
  id=  28  grid=(32, 16, 1)     _fwd_kernel                   Duration=41533408ns  DRAM Throughput=49.40%  Compute (SM) Throughput=16.09%  Memory Throughput=49.40%
  id=  30  grid=(32, 16, 1)     _fwd_kernel                   Duration=41575456ns  DRAM Throughput=49.52%  Compute (SM) Throughput=16.08%  Memory Throughput=49.52%
  id=  31  grid=(32, 16, 1)     _fwd_kernel                   Duration=41659392ns  DRAM Throughput=49.44%  Compute (SM) Throughput=16.08%  Memory Throughput=49.44%
  id=  33  grid=(32, 16, 1)     _fwd_kernel                   Duration=41988480ns  DRAM Throughput=49.60%  Compute (SM) Throughput=15.89%  Memory Throughput=49.60%
  id=  35  grid=(32, 16, 1)     _fwd_kernel                   Duration=41655712ns  DRAM Throughput=49.32%  Compute (SM) Throughput=16.09%  Memory Throughput=49.32%
  id=  37  grid=(32, 16, 1)     _fwd_kernel                   Duration=41686208ns  DRAM Throughput=49.44%  Compute (SM) Throughput=16.11%  Memory Throughput=49.44%
  id=  39  grid=(32, 16, 1)     _fwd_kernel                   Duration=41606656ns  DRAM Throughput=48.82%  Compute (SM) Throughput=15.88%  Memory Throughput=48.82%
  id=  41  grid=(32, 16, 1)     _fwd_kernel                   Duration=41656736ns  DRAM Throughput=49.57%  Compute (SM) Throughput=16.06%  Memory Throughput=49.57%
  id=  42  grid=(32, 16, 1)     _fwd_kernel                   Duration=41613280ns  DRAM Throughput=49.50%  Compute (SM) Throughput=16.07%  Memory Throughput=49.50%
  id=  44  grid=(32, 16, 1)     _fwd_kernel                   Duration=41735296ns  DRAM Throughput=49.50%  Compute (SM) Throughput=16.08%  Memory Throughput=49.50%
  id=  45  grid=(32, 16, 1)     _fwd_kernel                   Duration=34098944ns  DRAM Throughput=13.08%  Compute (SM) Throughput=18.17%  Memory Throughput=47.94%
  id=  47  grid=(32, 16, 1)     _fwd_kernel                   Duration=34093056ns  DRAM Throughput=13.07%  Compute (SM) Throughput=18.15%  Memory Throughput=47.87%
  id=  49  grid=(32, 16, 1)     _fwd_kernel                   Duration=34077760ns  DRAM Throughput=13.14%  Compute (SM) Throughput=18.15%  Memory Throughput=47.88%
  id=  51  grid=(32, 16, 1)     _fwd_kernel                   Duration=34076864ns  DRAM Throughput=13.09%  Compute (SM) Throughput=18.17%  Memory Throughput=47.94%
  id=  53  grid=(32, 16, 1)     _fwd_kernel                   Duration=34140896ns  DRAM Throughput=13.08%  Compute (SM) Throughput=18.16%  Memory Throughput=47.89%
  id=  55  grid=(32, 16, 1)     _fwd_kernel                   Duration=34131392ns  DRAM Throughput=13.07%  Compute (SM) Throughput=18.15%  Memory Throughput=47.87%
  id=  56  grid=(32, 16, 1)     _fwd_kernel                   Duration=34102272ns  DRAM Throughput=13.10%  Compute (SM) Throughput=18.16%  Memory Throughput=47.91%
  id=  58  grid=(32, 16, 1)     _fwd_kernel                   Duration=34089504ns  DRAM Throughput=13.12%  Compute (SM) Throughput=18.17%  Memory Throughput=47.94%
  id=  59  grid=(64, 16, 1)     _fwd_kernel                   Duration=34063296ns  DRAM Throughput=9.93%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id=  61  grid=(64, 16, 1)     _fwd_kernel                   Duration=34057344ns  DRAM Throughput=9.94%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id=  63  grid=(64, 16, 1)     _fwd_kernel                   Duration=34057888ns  DRAM Throughput=9.90%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id=  65  grid=(64, 16, 1)     _fwd_kernel                   Duration=34068320ns  DRAM Throughput=9.93%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id=  67  grid=(64, 16, 1)     _fwd_kernel                   Duration=34045824ns  DRAM Throughput=9.95%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id=  69  grid=(64, 16, 1)     _fwd_kernel                   Duration=34051168ns  DRAM Throughput=9.94%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id=  70  grid=(64, 16, 1)     _fwd_kernel                   Duration=34056928ns  DRAM Throughput=9.92%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id=  72  grid=(64, 16, 1)     _fwd_kernel                   Duration=34052256ns  DRAM Throughput=9.93%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id=  73  grid=(64, 16, 1)     _fwd_kernel                   Duration=33746656ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  75  grid=(64, 16, 1)     _fwd_kernel                   Duration=33736320ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.04%
  id=  77  grid=(64, 16, 1)     _fwd_kernel                   Duration=33739488ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.04%
  id=  79  grid=(64, 16, 1)     _fwd_kernel                   Duration=33741632ns  DRAM Throughput=0.19%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  81  grid=(64, 16, 1)     _fwd_kernel                   Duration=33739520ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  83  grid=(64, 16, 1)     _fwd_kernel                   Duration=33736736ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  84  grid=(64, 16, 1)     _fwd_kernel                   Duration=33748544ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  86  grid=(64, 16, 1)     _fwd_kernel                   Duration=33743296ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  87  grid=(16, 16, 1)     _fwd_kernel                   Duration=52202912ns  DRAM Throughput=39.12%  Compute (SM) Throughput=12.79%  Memory Throughput=39.12%
  id=  89  grid=(16, 16, 1)     _fwd_kernel                   Duration=52200512ns  DRAM Throughput=39.21%  Compute (SM) Throughput=12.78%  Memory Throughput=39.21%
  id=  91  grid=(16, 16, 1)     _fwd_kernel                   Duration=52173504ns  DRAM Throughput=39.14%  Compute (SM) Throughput=12.80%  Memory Throughput=39.14%
  id=  93  grid=(16, 16, 1)     _fwd_kernel                   Duration=52152704ns  DRAM Throughput=39.12%  Compute (SM) Throughput=12.79%  Memory Throughput=39.12%
  id=  95  grid=(16, 16, 1)     _fwd_kernel                   Duration=52275584ns  DRAM Throughput=39.09%  Compute (SM) Throughput=12.79%  Memory Throughput=39.09%
  id=  97  grid=(16, 16, 1)     _fwd_kernel                   Duration=52188032ns  DRAM Throughput=39.10%  Compute (SM) Throughput=12.79%  Memory Throughput=39.10%
  id=  98  grid=(16, 16, 1)     _fwd_kernel                   Duration=52220032ns  DRAM Throughput=39.11%  Compute (SM) Throughput=12.79%  Memory Throughput=39.11%
  id= 100  grid=(16, 16, 1)     _fwd_kernel                   Duration=52210848ns  DRAM Throughput=39.15%  Compute (SM) Throughput=12.79%  Memory Throughput=39.15%
  id= 101  grid=(64, 16, 1)     _fwd_kernel                   Duration=34051392ns  DRAM Throughput=9.94%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id= 102  grid=(64, 16, 1)     _fwd_kernel                   Duration=34053088ns  DRAM Throughput=9.91%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id= 103  grid=(64, 16, 1)     _fwd_kernel                   Duration=34054336ns  DRAM Throughput=9.92%  Compute (SM) Throughput=18.10%  Memory Throughput=48.94%
  id= 104  grid=(64, 16, 1)     _fwd_kernel                   Duration=34077184ns  DRAM Throughput=9.88%  Compute (SM) Throughput=18.09%  Memory Throughput=48.91%

=== Computing bytes moved, arithmetic intensity, roofline ===
naive: 6 representative kernels (1 per distinct kernel in the pipeline)
triton: 4 representative launches (tail steady-state cluster)
naive total HBM traffic (one call):  2.098 GB
triton total HBM traffic (one call): 1.013 GB (averaged over 4 steady-state launches)
naive latency (fresh, unprofiled):  8.588ms
triton latency (fresh, unprofiled): 12.487ms
naive arithmetic intensity:  8.187 FLOPs/byte, 2.000 TFLOP/s achieved
triton arithmetic intensity: 16.962 FLOPs/byte, 1.376 TFLOP/s achieved

Wrote results/figures/roofline.png

=== For manual/qualitative review (warp-stall reasons, full detail) ===
sudo /usr/local/cuda/bin/ncu --set full -o results/traces/naive_full /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target naive
sudo /usr/local/cuda/bin/ncu --set full -o results/traces/triton_full /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target triton
Then: /usr/local/cuda/bin/ncu -i results/traces/triton_full.ncu-rep   (opens the text report)

=== Timeline (Nsight Systems) ===
sudo /usr/local/cuda/bin/nsys profile -o results/traces/timeline /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target triton
(.venv) akashg7171_com@instance-20260718-143824:~/gpu-attention$ 