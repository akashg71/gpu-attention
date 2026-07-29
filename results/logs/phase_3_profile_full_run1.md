(.venv) akashg7171_com@instance-20260718-143824:~/gpu-attention$ sudo /usr/local/cuda/bin/ncu --set full -o results/traces/triton_full /home/akashg7171_com/gpu-attention/.venv/bin/python3 /home/akashg7171_com/gpu-attention/scripts/03_profile.py --target triton
==PROF== Connected to process 25311 (/usr/bin/python3.10)
==PROF== Profiling "distribution_elementwise_grid..." - 0: 0%....50%....100% - 31 passes
==PROF== Profiling "distribution_elementwise_grid..." - 1: 0%....50%....100% - 31 passes
==PROF== Profiling "distribution_elementwise_grid..." - 2: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 3: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 4: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 5: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 6: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 7: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 8: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 9: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 10: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 11: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 12: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 13: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 14: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 15: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 16: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 17: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 18: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 19: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 20: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 21: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 22: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 23: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 24: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 25: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 26: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 27: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 28: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 29: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 30: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 31: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 32: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 33: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 34: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 35: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 36: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 37: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 38: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 39: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 40: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 41: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 42: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 43: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 44: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 45: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 46: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 47: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 48: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 49: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 50: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 51: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 52: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 53: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 54: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 55: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 56: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 57: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 58: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 59: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 60: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 61: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 62: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 63: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 64: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 65: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 66: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 67: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 68: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 69: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 70: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 71: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 72: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 73: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 74: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 75: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 76: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 77: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 78: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 79: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 80: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 81: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 82: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 83: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 84: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 85: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 86: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 87: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 88: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 89: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 90: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 91: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 92: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 93: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 94: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 95: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 96: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 97: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 98: 0%....50%....100% - 31 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 99: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 100: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 101: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 102: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 103: 0%....50%....100% - 31 passes
==PROF== Profiling "_fwd_kernel" - 104: 0%....50%....100% - 31 passes
[diagnostic] autotuner cache: {(2048, 'torch.float16', 'torch.float16', 'torch.float16', 'torch.float16'): <triton.runtime.autotuner.Config object at 0x7a3d3795ed40>}
[diagnostic] autotuner best_config: BLOCK_M: 32, BLOCK_N: 32, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
==PROF== Disconnected from process 25311
==PROF== Report: /home/akashg7171_com/gpu-attention/results/traces/triton_full.ncu-rep
(.venv) akashg7171_com@instance-20260718-143824:~/gpu-attention$ git status
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
(.venv) akashg7171_com@instance-20260718-143824:~/gpu-attention$ 