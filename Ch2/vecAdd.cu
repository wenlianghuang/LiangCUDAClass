// Vector addition: C = 1/A + 1/B.
// compile with the following command:
//
// (for GTX970)
// nvcc -arch=compute_52 -code=sm_52,sm_52 -O2 -m64 -o vecAdd vecAdd.cu
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o vecAdd vecAdd.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/** Variables 2D 
float** h_A;   // host vectors
float** h_B;
float** h_C;
float** h_D;
float** d_A;   // device vectors
float** d_B;
float** d_C;
**/

// Functions
void RandomInit(float*data, int n)
{
    for(int i = 0; i < n*n; i++)
        data[i] = rand()/(float)RAND_MAX;
}
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    
    int threadCol = threadIdx.x + blockIdx.x * blockDim.x; //column
    int threadRow = threadIdx.y + blockIdx.y * blockDim.y; //row
    int indexOfMatrix = threadCol + threadRow * N;
    /**
    if(i <= N && j <= N)
        C[i][j] = A[j][i] + B[j][i];
    **/
    if(threadCol < N && threadRow < N){
        C[indexOfMatrix] = A[indexOfMatrix] + B[indexOfMatrix];
    }

    __syncthreads();

}

// Host code

int main( )
{
    float *h_A, *h_B, *h_C, *h_D;
    float *d_A, *d_B, *d_C;

    int gid;   

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    scanf("%d",&gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("Vector Addition: C = A + B\n");
    int mem = 1024*1024*1024;     // Giga    
    int N;

    printf("Enter the size of the vectors: ");
    scanf("%d",&N);        
    printf("%d\n",N);        
    if( 3*N > mem ) {     // each real number takes 4 bytes
      printf("The size of these 3 vectors cannot be fitted into 4 Gbyte\n");
      exit(2);
    }
    
    long size = N*N*sizeof(float);

   
    // Allocate input vectors h_A and h_B in host memory

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    /** 2D array
    h_A = new float*[N];
    h_B = new float*[N];
    h_C = new float*[N];
    for(int i = 0; i <N; i++)
    {
        h_A[i] = new float[N];
        h_B[i] = new float[N];
        h_C[i] = new float[N];
    }
    */
    
    /*     
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            h_A[i][j] = 0.;
            h_B[i][j] = 0.;
            h_C[i][j] = 0.;
        }
    }*/
        
     
    // Initialize the input vectors with random numbers

    RandomInit(h_A, N);
    RandomInit(h_B, N);
    // Set the sizes of threads and blocks

    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
   
    /**I am not sure, but the below lines are not need, the threadsPerBlock = (N,N) and blocksPerGrid = (4,4),(8,8),(10,10)...**/ 
    while(1)
    {
        printf("Enter the number of threads per block: ");
        scanf("%d",&threadsPerBlock);
        printf("%d\n",threadsPerBlock);
        if( threadsPerBlock > 1024) {
            printf("%d, The number of threads per block must be 1024!\n",threadsPerBlock);
            continue;
        }
        blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
        if( blocksPerGrid > 2147483647){
            printf("%d, The number of blocks per grid must be 2147483647!\n",blocksPerGrid);
            continue;
        }
        break;
    }

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory

    /** cudaMemcpy 1D**/
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    /** cudaMemcpy 2D 
    cudaMemcpy2D(d_A,sizeof(float)*N,h_A,sizeof(float)*N,sizeof(float)*N,N,cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B,sizeof(float)*N,h_B,sizeof(float)*N,sizeof(float)*N,N,cudaMemcpyHostToDevice);
    **/
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);
    
    dim3 blocksPerGrid2D(4,4);
    dim3 threadsPerBlock2D(threadsPerBlock,threadsPerBlock); 
    /**VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);**/
    VecAdd<<<blocksPerGrid2D,threadsPerBlock2D>>>(d_A, d_B, d_C,N);
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",3*N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    /**1D cudaMemcpy**/
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    /**2D cudaMemcpy**
    cudaMemcpy2D(h_C,sizeof(float)*N,d_C,sizeof(float)*N,sizeof(float)*N,N,cudaMemcpyDeviceToHost);
    **/
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    h_D = (float*)malloc(size);       // to compute the reference solution
    /**2D array
    h_D = new float*[N];
    for(int i = 0; i < N; i++){
        h_D[i] = new float[N];
    }
    
    for(int i = 0; i<N; i++){
        for(int j = 0; j<N; j++){
            h_D[i][j] = 0.;
        }
    }
        
    for (int i = 0; i < N; ++i)
        for(int j = 0; j < N ; j++) 
            h_D[i][j] = h_A[i][j] + h_B[i][j];
    **/

    /**
    for (int i = 0; i<N; i++)
        for(int j = 0; j<N; j++)
            h_D[i*N+j] = h_A[i*N+j] + h_B[i*N+j];
    **/

    for (int i = 0; i<N*N;i++)
        h_D[i] = h_A[i] + h_B[i];

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result

    printf("Check result:\n");
    
    /** 2D array
    double sum=0.; 
    double diff;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j){
        //printf("%f , %f\n",h_D[i][j],h_C[i][j]);
        diff = abs(h_D[i][j] - h_C[i][j]);
        sum += diff*diff;
        //printf("Test sum: %f\n",sum);
      }
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n",sum);
    **/
    double sum = 0.;
    double diff;
    for(int i = 0; i < N*N; i++){
        diff = abs(h_D[i]-h_C[i]);
        sum += diff;
    }
    /**
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            diff = abs(h_D[i*N+j] - h_C[i*N+j]);
            sum += diff * diff;
        }
    }**/
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n",sum);
    cudaDeviceReset();
}


