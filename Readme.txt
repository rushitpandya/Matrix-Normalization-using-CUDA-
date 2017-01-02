Installation Steps:-

1)Execute "qlogin -q interactive.q" command to schedule the job.
2)Go to hw4 directory.
3)Execute "gcc gpu.c" command to compile sequential code.
4)Execute "./a.out <matrix-dimension>" command to run sequential code.
5)Execute "nvcc Cuda_Matrix_Norm.cu -o cudatest" command to compile CUDA code.
6)Execute "./cudatest <matrix-dimension> <number_of_threads>" command to run CUDA code.
