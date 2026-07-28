# Phase 3 profiling run 1 — `scripts/03_profile.py`

Raw terminal output, unedited, from the GPU box. Shape: batch=2, heads=8,
seq_len=2048, head_dim=64, fp16, non-causal. See `instructions.md`'s
Progress Log for the analyzed/explained version of these numbers.

```
(.venv) akashg7171_com@instance-20260718-143824:~/gpu-attention$ python3 scripts/03_profile.py
=== Phase 3 profiling driver — shape {'batch': 2, 'heads': 8, 'seq_len': 2048, 'head_dim': 64}, dtype float16 ===

ncu: /usr/local/cuda/bin/ncu
nsys: /usr/local/cuda/bin/nsys

$ sudo /usr/local/cuda/bin/ncu --set roofline --csv /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target naive
Wrote results/traces/naive_ncu.csv (==PROF== noise stripped)

$ sudo /usr/local/cuda/bin/ncu --set roofline --csv /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target triton
[diagnostic] autotuner cache: {(2048, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float16'): <triton.runtime.autotuner.Config object at 0x7541de4629e0>}
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
  id=   3  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1231616ns  DRAM Throughput=41.35%  Compute (SM) Throughput=33.48%  Memory Throughput=41.59%
  id=   4  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1076640ns  DRAM Throughput=86.33%  Compute (SM) Throughput=16.76%  Memory Throughput=86.33%
  id=   5  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2452608ns  DRAM Throughput=63.08%  Compute (SM) Throughput=45.50%  Memory Throughput=63.08%
  id=   6  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2381440ns  DRAM Throughput=83.31%  Compute (SM) Throughput=35.51%  Memory Throughput=83.31%
  id=   7  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1570880ns  DRAM Throughput=88.77%  Compute (SM) Throughput=11.46%  Memory Throughput=88.77%
  id=   8  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=997568ns  DRAM Throughput=66.75%  Compute (SM) Throughput=72.24%  Memory Throughput=66.75%
  id=   9  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1230528ns  DRAM Throughput=41.30%  Compute (SM) Throughput=33.47%  Memory Throughput=41.59%
  id=  10  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1078720ns  DRAM Throughput=86.41%  Compute (SM) Throughput=16.69%  Memory Throughput=86.41%
  id=  11  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2464000ns  DRAM Throughput=63.12%  Compute (SM) Throughput=45.27%  Memory Throughput=63.12%
  id=  12  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2388416ns  DRAM Throughput=83.03%  Compute (SM) Throughput=35.42%  Memory Throughput=83.03%
  id=  13  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1570176ns  DRAM Throughput=88.91%  Compute (SM) Throughput=11.46%  Memory Throughput=88.91%
  id=  14  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=998720ns  DRAM Throughput=66.52%  Compute (SM) Throughput=72.37%  Memory Throughput=66.52%
  id=  15  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1229920ns  DRAM Throughput=41.35%  Compute (SM) Throughput=33.49%  Memory Throughput=41.61%
  id=  16  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1076992ns  DRAM Throughput=86.23%  Compute (SM) Throughput=16.72%  Memory Throughput=86.23%
  id=  17  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2451680ns  DRAM Throughput=62.80%  Compute (SM) Throughput=45.62%  Memory Throughput=62.80%
  id=  18  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2386752ns  DRAM Throughput=83.58%  Compute (SM) Throughput=35.55%  Memory Throughput=83.58%
  id=  19  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1569152ns  DRAM Throughput=88.78%  Compute (SM) Throughput=11.46%  Memory Throughput=88.78%
  id=  20  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=995328ns  DRAM Throughput=66.53%  Compute (SM) Throughput=72.31%  Memory Throughput=66.53%
  id=  21  grid=(8, 16, 16)     turing_fp16_s1688gemm_fp16_2  Duration=1230080ns  DRAM Throughput=41.38%  Compute (SM) Throughput=33.47%  Memory Throughput=41.58%
  id=  22  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1079360ns  DRAM Throughput=86.27%  Compute (SM) Throughput=16.71%  Memory Throughput=86.27%
  id=  23  grid=(131072, 1, 1)  void at::unrolled_elementwis  Duration=2452032ns  DRAM Throughput=63.52%  Compute (SM) Throughput=45.32%  Memory Throughput=63.52%
  id=  24  grid=(8192, 1, 1)    void <unnamed>::softmax_warp  Duration=2375808ns  DRAM Throughput=83.23%  Compute (SM) Throughput=35.61%  Memory Throughput=83.23%
  id=  25  grid=(65536, 1, 1)   void at::vectorized_elementw  Duration=1570624ns  DRAM Throughput=88.86%  Compute (SM) Throughput=11.45%  Memory Throughput=88.86%
  id=  26  grid=(1, 16, 16)     turing_fp16_s1688gemm_fp16_1  Duration=998048ns  DRAM Throughput=66.53%  Compute (SM) Throughput=72.17%  Memory Throughput=66.53%

triton: 2379 metric-rows -> 105 distinct kernel launches (45 excluded as RNG/fill noise).
Attention-relevant kernels (name, grid size, block size):
    24x  grid=(32, 16, 1)     block=(128, 1, 1)  _fwd_kernel
    20x  grid=(64, 16, 1)     block=(128, 1, 1)  _fwd_kernel
     8x  grid=(16, 16, 1)     block=(256, 1, 1)  _fwd_kernel
     8x  grid=(16, 16, 1)     block=(128, 1, 1)  _fwd_kernel

Per-launch detail, in launch order (['Duration', 'DRAM Throughput', 'Compute (SM) Throughput', 'Memory Throughput']):
  id=   3  grid=(16, 16, 1)     _fwd_kernel                   Duration=53603264ns  DRAM Throughput=55.26%  Compute (SM) Throughput=14.02%  Memory Throughput=55.26%
  id=   5  grid=(16, 16, 1)     _fwd_kernel                   Duration=53610400ns  DRAM Throughput=55.33%  Compute (SM) Throughput=14.05%  Memory Throughput=55.33%
  id=   7  grid=(16, 16, 1)     _fwd_kernel                   Duration=53592864ns  DRAM Throughput=55.31%  Compute (SM) Throughput=14.03%  Memory Throughput=55.31%
  id=   9  grid=(16, 16, 1)     _fwd_kernel                   Duration=53668864ns  DRAM Throughput=55.34%  Compute (SM) Throughput=14.02%  Memory Throughput=55.34%
  id=  11  grid=(16, 16, 1)     _fwd_kernel                   Duration=53551680ns  DRAM Throughput=55.59%  Compute (SM) Throughput=14.03%  Memory Throughput=55.59%
  id=  13  grid=(16, 16, 1)     _fwd_kernel                   Duration=53661632ns  DRAM Throughput=55.48%  Compute (SM) Throughput=14.06%  Memory Throughput=55.48%
  id=  14  grid=(16, 16, 1)     _fwd_kernel                   Duration=53675328ns  DRAM Throughput=55.49%  Compute (SM) Throughput=14.03%  Memory Throughput=55.49%
  id=  16  grid=(16, 16, 1)     _fwd_kernel                   Duration=53533888ns  DRAM Throughput=55.29%  Compute (SM) Throughput=14.04%  Memory Throughput=55.29%
  id=  17  grid=(32, 16, 1)     _fwd_kernel                   Duration=41633248ns  DRAM Throughput=49.44%  Compute (SM) Throughput=15.92%  Memory Throughput=49.44%
  id=  19  grid=(32, 16, 1)     _fwd_kernel                   Duration=41483776ns  DRAM Throughput=49.48%  Compute (SM) Throughput=15.92%  Memory Throughput=49.48%
  id=  21  grid=(32, 16, 1)     _fwd_kernel                   Duration=41707712ns  DRAM Throughput=49.61%  Compute (SM) Throughput=16.12%  Memory Throughput=49.61%
  id=  23  grid=(32, 16, 1)     _fwd_kernel                   Duration=41622016ns  DRAM Throughput=49.66%  Compute (SM) Throughput=16.12%  Memory Throughput=49.66%
  id=  25  grid=(32, 16, 1)     _fwd_kernel                   Duration=41658240ns  DRAM Throughput=49.59%  Compute (SM) Throughput=16.11%  Memory Throughput=49.59%
  id=  27  grid=(32, 16, 1)     _fwd_kernel                   Duration=41644448ns  DRAM Throughput=49.48%  Compute (SM) Throughput=16.10%  Memory Throughput=49.48%
  id=  28  grid=(32, 16, 1)     _fwd_kernel                   Duration=42160544ns  DRAM Throughput=49.04%  Compute (SM) Throughput=16.14%  Memory Throughput=49.04%
  id=  30  grid=(32, 16, 1)     _fwd_kernel                   Duration=42119264ns  DRAM Throughput=49.36%  Compute (SM) Throughput=16.09%  Memory Throughput=49.36%
  id=  31  grid=(32, 16, 1)     _fwd_kernel                   Duration=41602336ns  DRAM Throughput=49.28%  Compute (SM) Throughput=16.14%  Memory Throughput=49.28%
  id=  33  grid=(32, 16, 1)     _fwd_kernel                   Duration=41794400ns  DRAM Throughput=48.86%  Compute (SM) Throughput=16.13%  Memory Throughput=48.86%
  id=  35  grid=(32, 16, 1)     _fwd_kernel                   Duration=41809600ns  DRAM Throughput=48.92%  Compute (SM) Throughput=16.11%  Memory Throughput=48.92%
  id=  37  grid=(32, 16, 1)     _fwd_kernel                   Duration=41739936ns  DRAM Throughput=49.49%  Compute (SM) Throughput=16.10%  Memory Throughput=49.49%
  id=  39  grid=(32, 16, 1)     _fwd_kernel                   Duration=41733312ns  DRAM Throughput=49.58%  Compute (SM) Throughput=16.12%  Memory Throughput=49.58%
  id=  41  grid=(32, 16, 1)     _fwd_kernel                   Duration=41609120ns  DRAM Throughput=49.39%  Compute (SM) Throughput=16.07%  Memory Throughput=49.39%
  id=  42  grid=(32, 16, 1)     _fwd_kernel                   Duration=41711680ns  DRAM Throughput=49.43%  Compute (SM) Throughput=16.06%  Memory Throughput=49.43%
  id=  44  grid=(32, 16, 1)     _fwd_kernel                   Duration=41671744ns  DRAM Throughput=49.59%  Compute (SM) Throughput=16.11%  Memory Throughput=49.59%
  id=  45  grid=(32, 16, 1)     _fwd_kernel                   Duration=34092448ns  DRAM Throughput=13.07%  Compute (SM) Throughput=18.15%  Memory Throughput=47.88%
  id=  47  grid=(32, 16, 1)     _fwd_kernel                   Duration=34063808ns  DRAM Throughput=13.08%  Compute (SM) Throughput=18.15%  Memory Throughput=47.89%
  id=  49  grid=(32, 16, 1)     _fwd_kernel                   Duration=34073344ns  DRAM Throughput=13.05%  Compute (SM) Throughput=18.16%  Memory Throughput=47.90%
  id=  51  grid=(32, 16, 1)     _fwd_kernel                   Duration=34065312ns  DRAM Throughput=13.10%  Compute (SM) Throughput=18.17%  Memory Throughput=47.94%
  id=  53  grid=(32, 16, 1)     _fwd_kernel                   Duration=34114048ns  DRAM Throughput=13.08%  Compute (SM) Throughput=18.16%  Memory Throughput=47.91%
  id=  55  grid=(32, 16, 1)     _fwd_kernel                   Duration=34159648ns  DRAM Throughput=13.12%  Compute (SM) Throughput=18.16%  Memory Throughput=47.89%
  id=  56  grid=(32, 16, 1)     _fwd_kernel                   Duration=34101280ns  DRAM Throughput=13.09%  Compute (SM) Throughput=18.15%  Memory Throughput=47.88%
  id=  58  grid=(32, 16, 1)     _fwd_kernel                   Duration=34057408ns  DRAM Throughput=13.12%  Compute (SM) Throughput=18.13%  Memory Throughput=47.83%
  id=  59  grid=(64, 16, 1)     _fwd_kernel                   Duration=34045504ns  DRAM Throughput=9.93%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id=  61  grid=(64, 16, 1)     _fwd_kernel                   Duration=34055040ns  DRAM Throughput=9.93%  Compute (SM) Throughput=18.10%  Memory Throughput=48.93%
  id=  63  grid=(64, 16, 1)     _fwd_kernel                   Duration=34062144ns  DRAM Throughput=9.92%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id=  65  grid=(64, 16, 1)     _fwd_kernel                   Duration=34043232ns  DRAM Throughput=9.91%  Compute (SM) Throughput=18.11%  Memory Throughput=48.94%
  id=  67  grid=(64, 16, 1)     _fwd_kernel                   Duration=34054400ns  DRAM Throughput=9.95%  Compute (SM) Throughput=18.10%  Memory Throughput=48.93%
  id=  69  grid=(64, 16, 1)     _fwd_kernel                   Duration=34056416ns  DRAM Throughput=9.90%  Compute (SM) Throughput=18.10%  Memory Throughput=48.94%
  id=  70  grid=(64, 16, 1)     _fwd_kernel                   Duration=34068992ns  DRAM Throughput=9.92%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id=  72  grid=(64, 16, 1)     _fwd_kernel                   Duration=34060000ns  DRAM Throughput=9.89%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id=  73  grid=(64, 16, 1)     _fwd_kernel                   Duration=33740224ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.04%
  id=  75  grid=(64, 16, 1)     _fwd_kernel                   Duration=33746784ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  77  grid=(64, 16, 1)     _fwd_kernel                   Duration=33745856ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  79  grid=(64, 16, 1)     _fwd_kernel                   Duration=33740480ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.04%
  id=  81  grid=(64, 16, 1)     _fwd_kernel                   Duration=33737056ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  83  grid=(64, 16, 1)     _fwd_kernel                   Duration=33748576ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  84  grid=(64, 16, 1)     _fwd_kernel                   Duration=33742048ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.67%  Memory Throughput=49.04%
  id=  86  grid=(64, 16, 1)     _fwd_kernel                   Duration=33745472ns  DRAM Throughput=0.18%  Compute (SM) Throughput=18.68%  Memory Throughput=49.05%
  id=  87  grid=(16, 16, 1)     _fwd_kernel                   Duration=52188448ns  DRAM Throughput=39.06%  Compute (SM) Throughput=12.78%  Memory Throughput=39.06%
  id=  89  grid=(16, 16, 1)     _fwd_kernel                   Duration=52151840ns  DRAM Throughput=39.12%  Compute (SM) Throughput=12.79%  Memory Throughput=39.12%
  id=  91  grid=(16, 16, 1)     _fwd_kernel                   Duration=52168864ns  DRAM Throughput=39.18%  Compute (SM) Throughput=12.78%  Memory Throughput=39.18%
  id=  93  grid=(16, 16, 1)     _fwd_kernel                   Duration=52213888ns  DRAM Throughput=39.17%  Compute (SM) Throughput=12.80%  Memory Throughput=39.17%
  id=  95  grid=(16, 16, 1)     _fwd_kernel                   Duration=52188000ns  DRAM Throughput=39.15%  Compute (SM) Throughput=12.79%  Memory Throughput=39.15%
  id=  97  grid=(16, 16, 1)     _fwd_kernel                   Duration=52217728ns  DRAM Throughput=39.17%  Compute (SM) Throughput=12.80%  Memory Throughput=39.17%
  id=  98  grid=(16, 16, 1)     _fwd_kernel                   Duration=52186592ns  DRAM Throughput=39.15%  Compute (SM) Throughput=12.79%  Memory Throughput=39.15%
  id= 100  grid=(16, 16, 1)     _fwd_kernel                   Duration=52192800ns  DRAM Throughput=39.11%  Compute (SM) Throughput=12.79%  Memory Throughput=39.11%
  id= 101  grid=(64, 16, 1)     _fwd_kernel                   Duration=34050560ns  DRAM Throughput=9.90%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id= 102  grid=(64, 16, 1)     _fwd_kernel                   Duration=34055328ns  DRAM Throughput=9.90%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%
  id= 103  grid=(64, 16, 1)     _fwd_kernel                   Duration=34054432ns  DRAM Throughput=9.95%  Compute (SM) Throughput=18.11%  Memory Throughput=48.96%
  id= 104  grid=(64, 16, 1)     _fwd_kernel                   Duration=34054432ns  DRAM Throughput=9.93%  Compute (SM) Throughput=18.11%  Memory Throughput=48.95%

=== Computing bytes moved, arithmetic intensity, roofline ===
naive: 6 representative kernels (1 per distinct kernel in the pipeline)
triton: 4 representative launches (tail steady-state cluster)
naive total HBM traffic:  2.110 GB
triton total HBM traffic: 4.054 GB
naive latency (fresh, unprofiled):  8.585ms
triton latency (fresh, unprofiled): 12.491ms
naive arithmetic intensity:  8.140 FLOPs/byte, 2.001 TFLOP/s achieved
triton arithmetic intensity: 4.238 FLOPs/byte, 1.375 TFLOP/s achieved

Wrote results/figures/roofline.png

=== For manual/qualitative review (warp-stall reasons, full detail) ===
sudo /usr/local/cuda/bin/ncu --set full -o results/traces/naive_full /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target naive
sudo /usr/local/cuda/bin/ncu --set full -o results/traces/triton_full /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target triton
Then: /usr/local/cuda/bin/ncu -i results/traces/triton_full.ncu-rep   (opens the text report)

=== Timeline (Nsight Systems) ===
sudo /usr/local/cuda/bin/nsys profile -o results/traces/timeline /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target triton
(.venv) akashg7171_com@instance-20260718-143824:~/gpu-attention$
```
