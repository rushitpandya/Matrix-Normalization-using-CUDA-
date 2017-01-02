#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>


#define MAXN 8000  /* Max value of N */
int N;  /* Matrix Dimension*/
int numThreads;  /* Number of Threads */

/*Random*/
#define randm() 4|2[uid]&3

/*CUDA Function for calculating mean column-wise and then reducing each column's totals*/
/*This Function will be called Number of blocks times*/
__global__ void Mean_SD_Norm(float* input,float* output ,float* mean_out,float* sd_out, int dim1, int numThread,int eval_ceil)
{
  extern __shared__ float mean[];//shared 1D-matrix for storing temporary results for mean of each threads 
  extern __shared__ float sd[];//shared 1D-matrix for storing temporary results for sd of each threads 
  __shared__ float meansum;//shared 1D-matrix for storing mean total of each threads
  __shared__ float sdsum;//shared 1D-matrix for storing SD total of each threads
  
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;//Getting Thread X Index for Particular Block
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;//Getting Thread Y Index for Particular Block
  int eva_block,index; 	 
  
  unsigned int thread_id = threadIdx.y;//Getting Id of thread
  unsigned int j = idx_y * dim1 + idx_x;//calculating index for input matrix
  
	__syncthreads();//waiting for all threads

	mean[thread_id]=input[j];//Assigned each column element of matrix to each thread

	/*If Dimension is more than Threads then reduce the remaining elements to assigned elements*/	
  for(int i=0;i<dim1;i+=numThread)
  {
		index=dim1*(numThread+thread_id+i);//calculating index of remaining element
		eva_block=index+blockIdx.x;
		if(eva_block < dim1*dim1)
		{
			mean[thread_id]+=input[index];
		}	
  }
  
  /*Reducing sum of each thread to final block sum*/
 if(thread_id==0)
 {
	for(int i=0;i<numThread;i++)
	{
		meansum+=mean[thread_id+i];
	}
	mean_out[blockIdx.x]=meansum/dim1;//Mean of block
 } 

 __syncthreads();
 sd[thread_id] = powf(input[j] - mean_out[blockIdx.x], 2.0);//evaluating SD for each thread for particular block
 
 
 	/*If Dimension is more than Threads then reduce the remaining elements to assigned elements*/	
  for(int i=0;i<dim1;i+=numThread)
  {
		index=dim1*(numThread+thread_id+i);
		eva_block=index+blockIdx.x;
		if(eva_block < dim1*dim1)
		{
			sd[thread_id]+=powf(input[index] - mean_out[blockIdx.x], 2.0);
		}	
  }
  
    /*Reducing SD Sum of each thread to final block SD sum*/
  if(thread_id==0)
 {
	 sdsum=0;
	for(int i=0;i<numThread;i++)
	{
		sdsum+=sd[thread_id+i];//calculating index of remaining element
	}
	sd_out[blockIdx.x]=sdsum/dim1;//SD of block
 } 
 
 __syncthreads();//waiting for threads

	/*Normalization of each block data on basis of mean and sd of each block*/
	output[blockIdx.x*dim1+thread_id] = (input[thread_id+blockIdx.x*dim1] - mean_out[blockIdx.x]) / sd_out[blockIdx.x];	
	
	/*Reducing Normalized Sum for remaining elements*/
	for(int i=0;i<eval_ceil;i++){
		if((numThread+thread_id)+blockIdx.x*dim1 < dim1*dim1)
		{
			output[(numThread+thread_id)+blockIdx.x*dim1] = (input[(numThread+thread_id)+blockIdx.x*dim1] - mean_out[blockIdx.x])/sd_out[blockIdx.x];//Normalizing the Matrix Indexes
		}	
  	}
}

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 4) {	
    seed = atoi(argv[3]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 3) {
    N = atoi(argv[1]);
    numThreads = atoi(argv[2]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
	
	/*Number of Threads should be less than or equal to 1024 else exit*/
	if (numThreads > 1024) 
	{
      printf("Number of threads cannot be more than %i.\n", 1024);
      exit(0);
    }
  }
  else 
  {
    printf("Usage: %s <matrix_dimension> <Number of Threads> [random seed]\n",argv[0]);    
    exit(0);
  }

  printf("\nMatrix dimension N = %i.\n", N);
}


int main(int argc, char **argv) 
{
  /* Timing variables */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);
 
  float* Host_Input = new float [N * N];//Input Matrix
  float* Host_Output = new float [N * N];//Output Matrix

  int i,j;
/*Initializing Input Matrix with random values*/ 
 printf("\nInitializing...\n");
  for(i=0;i<N;i++)
  {
    for(j=0;j<N;j++)
    {
//      Host_Input[j* N + i] = j+1;
 Host_Input[j* N + i] = (float)rand() / 32768.0;
    }
  }
  /* Displaying Input Matrix */
   /* printf("\nA1 =\n\t");
    for (i = 0; i < N; i++) 
	{
      for (j = 0; j < N; j++) 
	  {
		printf("%5.2f%s", Host_Input[i* N + j], (j < N-1) ? ", " : ";\n\t");
      }
    }
  */

  float* input;//Device Input Matrix
  float* output;//Device Output Matrix
  float* mean_out;//Device Mean Matrix
  float* sd_out;//Device SD Matrix
  size_t matrix_size_2d = N * N * sizeof(float);//Size of 2D Matrix
  size_t matrix_size_1d = N * sizeof(float);//Size of 1D Matrix

  //allocated the device memory for source array
  cudaMalloc(&input, matrix_size_2d);
  cudaMemcpy(input, Host_Input, matrix_size_2d, cudaMemcpyHostToDevice);

  //allocate the device memory for destination array
  cudaMalloc(&output, matrix_size_2d);

  //allocate the device memory for mean array
  cudaMalloc(&mean_out, matrix_size_1d);

  //allocate the device memory for sd array
  cudaMalloc(&sd_out, matrix_size_1d);

  dim3 dimBlock;
  dim3 dimGrid;

  /* Designing Decisions for number of blocks and number of threads in each block */
  if( N < numThreads)
  {
    dimBlock.x = 1;
    dimBlock.y = N;
    dimGrid.x = N;
    dimGrid.y = 1;
  }
  else
  {
    dimBlock.x = 1;
    dimBlock.y = numThreads;
    dimGrid.x = N;
    dimGrid.y = 1;
  }

  /* Start Clock */
  printf("\nStarting clock.\n");
  cudaEventRecord(start);
  gettimeofday(&etstart,&tzdummy);
  etstart2 = times(&cputstart);

	double d_ceil=(double)N/(double)numThreads;
	int c=ceil(d_ceil);

//printf("nt=%d\t c1=%ld\tc=%d\n",nt,c1,c);

  //Calling CUDA Kernel Function For Normalizing Matrix
  Mean_SD_Norm<<<dimGrid, dimBlock, matrix_size_1d>>>(input,output,mean_out,sd_out,N,numThreads,c);
  cudaDeviceSynchronize();
  
  /* Stop Clock */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");

  /*Copying Output Device Matrix to Output Host Matrix*/
  cudaMemcpy(Host_Output, output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
 /* if (N < 10) {
  printf("\nB1 =\n\t");
    for (i= 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%1.10f%s", Host_Output[i* N + j], (j < N-1) ? ", " : ";\n\t");
        }
    }
  }
*/
  /* Display timing results */
  printf("\nElapsed time CPU Time = %g ms.\n", (float)(usecstop - usecstart)/(float)1000);
  printf("Elapsed time Cuda Time = %g ms \n",milliseconds);
  printf("Effective Bandwidth (GB/s): %f \n", (2*matrix_size_2d/milliseconds)/1e6);
  float mean = N * log2((float)N) + N;
  float sd = N * log2((float)N) + (2*N) + (2*N*N);
  float norm = 2 * N * N;
  printf("Effective Throughput (GFLOPS/s): %f \n", ((mean+sd+norm)*1e-9)/(milliseconds*1e-3)); 
  printf("--------------------------------------------\n");

  //deallocate device memory
  cudaFree(input);
  cudaFree(output);
  cudaFree(mean_out);
  cudaFree(sd_out);

  //deallocate Host Input and Host Output Matrix
  free(Host_Input);
  free(Host_Output);

  exit(0);
}
