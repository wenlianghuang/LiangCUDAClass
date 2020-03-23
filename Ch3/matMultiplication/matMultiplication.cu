#include<iostream>


void RandomInit(float *mat,int N)
{
   for(int i = 0; i < N*N; i++)
      mat[i] = rand()/(float) RAND_MAX;
}

__global__ void matMul(float *matA, float *matB, float *matC, int N)
{
   int column = threadIdx.x + blockIdx.x*blockDim.x;
   int row = threadIdx.y + blockIdx.x*blockDim.y;
   
   int sum = 0;
   if(column < N && row < N)
   {
      for(int k = 0; k < N*N; k++)
      {
         sum += matA[N*column + k] * matB[k*N + column];
      }
      
      matC[row*N+column] = sum;
   }
} 
int main()
{
   int matrixSize;
   printf("Input the matrix size: ");
   scanf("%d",&matrixSize);
   
   size_t size = matrixSize * matrixSize * sizeof(int);
   float *matA,*matB,*matC,*matD;
   float *gpuMatA,*gpuMatB,*gpuMatC;
  
   matA = (float*)malloc(size);
   matB = (float*)malloc(size);
   matC = (float*)malloc(size);

   RandomInit(matA,matrixSize);
   RandomInit(matB,matrixSize);

   cudaEvent_t gpuStart,gpuStop,cpuStart,cpuStop;
   cudaEventCreate(&gpuStart);
   cudaEventCreate(&gpuStop);
   cudaEventCreate(&cpuStart);
   cudaEventCreate(&cpuStop);

   float cpu_tottime,gpu_tottime;
   cudaEventRecord(cpuStart,0);
   
   for(int i = 0; i < matrixSize; i++)
   {
      for(int j = 0; j < matrixSize; j++)
      {
         int sum = 0;
         for(int k = 0; k < matrixSize; k++)
         {
            sum += matA[i*matrixSize + k] * matB[k*matrixSize + j];
         }
         matC[i*matrixSize + j] = sum;
      }
   }

   cudaEventRecord(cpuStop,0);
   cudaEventSynchronize(cpuStop);
   cudaEventElapsedTime(&cpu_tottime,cpuStart,cpuStop);
   printf("CPU time %5.5f (ms) by matrix multiplication\n",cpu_tottime);

   int threadsPerblock = 0;
   int blocksPergrid = 0;
   cudaEventRecord(gpuStart,0);
   cudaMalloc((void**)&gpuMatA,size);
   cudaMalloc((void**)&gpuMatB,size);
   cudaMalloc((void**)&gpuMatC,size);

   cudaMemcpy((void **)gpuMatA,matA,size,cudaMemcpyHostToDevice);
   cudaMemcpy((void **)gpuMatB,matB,size,cudaMemcpyHostToDevice);

   cudaEventRecord(gpuStop,0);
   cudaEventSynchronize(gpuStop);

   float Inittime;
   cudaEventElapsedTime(&Inittime,gpuStart,gpuStop);
   printf("Input time: %5.5f (ms)\n",Inittime);

   printf("Input the threads per block: ");
   scanf("%d",&threadsPerblock);
   printf("\nInput the blocks per grid: ");
   scanf("%d",&blocksPergrid);
   printf("\n");
   
   matD = (float*)malloc(size);
   dim3 blocks(threadsPerblock,threadsPerblock);
   dim3 grid(blocksPergrid,blocksPergrid);
   
   cudaEventRecord(gpuStart,0);
   matMul<<<grid,blocks>>>(gpuMatA,gpuMatB,gpuMatC,matrixSize);
   cudaEventRecord(gpuStop,0);
   cudaEventSynchronize(gpuStop);

   float pro_time;
   cudaEventElapsedTime(&pro_time,gpuStart,gpuStop);
   printf("GPU Processing time: %5.5f (ms)\n",pro_time);

   cudaEventRecord(gpuStart,0); 
   cudaMemcpy(matD,gpuMatC,size,cudaMemcpyDeviceToHost);
   cudaFree(gpuMatA);
   cudaFree(gpuMatB);
   cudaFree(gpuMatC);
   cudaEventRecord(gpuStop,0);
   cudaEventSynchronize(gpuStop);
   float Outtime;
   cudaEventElapsedTime(&Outtime,gpuStart,gpuStop);
   printf("Output time: %5.5f (ms)\n",Outtime);

   printf("Total time of GPU: %5.5f (ms)\n",(Inittime+pro_time+Outtime));
   
}



   
    
