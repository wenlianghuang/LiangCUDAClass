#include<stdio.h>
#include<stdlib.h>




__global__ void VecAdd(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;//col
    int j = blockIdx.y * blockDim.y + threadIdx.y;//row

    int indexOfMatrix = i + j * N;

    if(i < N && j < N)
        C[indexOfMatrix] = A[indexOfMatrix] + B[indexOfMatrix];

}


void RandomInit(float *data, int n)
{
    for(int i = 0; i < n*n; i++)
        data[i] = rand()/(float)RAND_MAX;
}

void readValue(int *value, char * msg, int lowerBound, int upperBound)
{
      while(true)
      {
          printf("%s(%d-%d): ", msg, lowerBound, upperBound);
          scanf("%d", value);
 
          if(*value <= upperBound && *value >= lowerBound)
              return;
      }
}

int main()
{

   //Have some variables required for loop counters.
   int i;

   //have variables for threads per block, number of blocks.
   int threadsPerBlock = 0, blocksInGrid = 0;

   //create cuda event variables
   cudaEvent_t hostStart, hostStop, deviceStart, deviceStop;
   float timeDifferenceOnHost, timeDifferenceOnDevice;

   //program variables
   int N = 0;
   size_t size;                     //variable to have the size of arrays on device
   //int *matA, *matB, *matC, *matCFromGPU;   //matrices for host
   float *h_A; 
   float *h_B; 
   float *h_C; 
   float *h_D;
   float *d_A;
   float *d_B;
   float *d_C;            //matrices for Device

   //initialize cuda timing variables
   cudaEventCreate(&hostStart);
   cudaEventCreate(&hostStop);
   cudaEventCreate(&deviceStart);
   cudaEventCreate(&deviceStop);
  
   
   printf("Enter the size: ");
   scanf("%d",&N);
   //calculate the size required on GPU
   size = N * N * sizeof(float);

   h_A = (float*)malloc(size);
   h_B = (float*)malloc(size);
   h_C = (float*)malloc(size);
   RandomInit(h_A,N);
   RandomInit(h_B,N);
   printf("Adding matrices on CPU...\n");
   cudaEventRecord(hostStart, 0);
   for(i = 0 ; i < N * N; i ++)
           h_C[i] = h_A[i] + h_B[i];
   cudaEventRecord(hostStop, 0);
   cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
   /**printf("Matrix addition over. Time taken on CPU: %5.5f\n", timeDifferenceOnHost);**/
   printf("Processing time for CPU: %5.5f (ms)\n",timeDifferenceOnHost); 
   printf("CPU: %fGflops\n",3*N/(1000000*timeDifferenceOnHost)); 
   
   
   cudaMalloc((void**)&d_A,size);
   cudaMalloc((void**)&d_B,size);
   cudaMalloc((void**)&d_C,size);

   cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
   

   bool done = false;
 
   while(!done)   
   {
       h_D = (float *)malloc(size);

       //create a proper grid block using dim3
       readValue(&threadsPerBlock, "Enter no. of threads per block(input of 'P' will construct PxP threads in block)", 4, 32);
       readValue(&blocksInGrid, "Enter no. of blocks in grid(input of 'P' will construct PxP blocks)", (N + threadsPerBlock -1)/threadsPerBlock, 65535);
       printf("Threads Per block: %d, Blocks in grid: %d\n", threadsPerBlock, blocksInGrid); 
       printf("Adding matrices on GPU..\n");
       dim3 blocks(threadsPerBlock, threadsPerBlock);                                                   
       dim3 grid(blocksInGrid, blocksInGrid); //(matrixSize + threadsPerBlock - 1/blocks.x), (matrixSize + blocks.y - 1/blocks.y));
    
       //call the kernels to execute
       cudaEventRecord(deviceStart, 0);
       printf("Total linear threads: %d\n", blocksInGrid*threadsPerBlock);
       VecAdd<<<grid, blocks>>>(d_A, d_B, d_C, N);
       cudaEventRecord(deviceStop, 0);
       cudaEventSynchronize(deviceStop);
    
       cudaEventElapsedTime(&timeDifferenceOnDevice, deviceStart, deviceStop);
      
       printf("Processing time for GPU: %5.5f (ms)\n",timeDifferenceOnDevice); 
       printf("GPU: %fGflops\n",3*N/(1000000*timeDifferenceOnDevice)); 
       //copy the result back into host memory
       cudaMemcpy(h_D, d_C, size, cudaMemcpyDeviceToHost);
    
       printf("Speedup: %5.5f\n", (float)timeDifferenceOnHost/timeDifferenceOnDevice);
       double sum = 0.;
       double diff;
       for(int i = 0; i < N * N ; i++){
            diff = abs(h_D[i]-h_C[i]);
            sum += diff*diff;
        }
        sum = sqrt(sum);
        printf("norm(h_C - h_D)=%20.15e\n\n",sum);
       char c = 'n';
       printf("Again?(y/n): ");
       while(true)
          {
             c = getchar();
             if(c == 'y' || c == 'n')
           break;
          }
       if(c == 'n')
         break;
    
       free(h_D);
   }

   free(h_A);
   free(h_B);
   free(h_C);
    
   cudaEventDestroy(deviceStart);
   cudaEventDestroy(deviceStop);
   cudaEventDestroy(hostStart);
   cudaEventDestroy(hostStop);
 
   return 0;
   
}
