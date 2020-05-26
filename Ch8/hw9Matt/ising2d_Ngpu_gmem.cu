//  Monte Carlo simulation of Ising model on 2D lattice
//  using Metropolis algorithm
//  using checkerboard (even-odd) update 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void exact_2d(double, double, double*, double*);
void rng_MT(float*, int);

double ellf(double phi, double ak);
double rf(double x, double y, double z);
double min(double x, double y, double z);
double max(double x, double y, double z);

int *spin;            // host spin variables
float *h_rng;         // host random numbers

__constant__ int fw[1000],bw[1000];     // declare constant memory for fw, bw 

__global__ void metro_gmem_odd(int* spin, float *ranf, int *s_L, int *s_R, int *s_T, int *s_B, const float B, const float T)
{
    int    x, y, parity;
    int    i, io;
    int    old_spin, new_spin, spins;
    float l,r,t,b;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(Nx,Ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
  
    int Nx = 2*blockDim.x;             // block size before even-odd reduction
    int nx = 2*blockDim.x*gridDim.x;   // number of sites in x-axis of the entire lattice 
    int ny = blockDim.y*gridDim.y; // go over to the odd sites
    // next, go over the odd sites 

    io = threadIdx.x + threadIdx.y*blockDim.x;   
    x = (2*io)%Nx;
    y = ((2*io)/Nx)%Nx;
    parity=(x+y+1)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice

    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*nx;
    old_spin = spin[i];
    new_spin = -old_spin;
    //for each neighboring spin
    if(x == 0){
        l = s_L[(nx-1)];
        r = spin[i+1];
    }
    else if(x == nx-1){
        l = spin[i-1];
        r = s_R[y*nx];
    }
    else{
        l = spin[i-1];
        r = spin[i+1];
    }
    if(y == 0){
        t = spin[i+nx];
        b = s_B[x+y*(nx-1)];
    }
    else if(y == ny-1){
        t = s_T[x];
        b = spin[i-nx];
    }
    else{
        t = spin[i+nx];
        b = spin[i-nx];
    }
    spins = l+r+t+b;

    de = -(new_spin - old_spin)*(spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }

    __syncthreads();

}


__global__ void metro_gmem_even(int* spin, float *ranf, int *s_L, int *s_R, int *s_T, int *s_B, const float B, const float T)
{
    int    x, y, parity;
    int    i, ie;
    int    old_spin, new_spin, spins;
    float l,r,t,b;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(Nx,Ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
  
    int Nx = 2*blockDim.x;             // block size before even-odd reduction
    int nx = 2*blockDim.x*gridDim.x;   // number of sites in x-axis of the entire lattice 
    int ny = blockDim.x*gridDim.y; //for even sites

    // first, go over the even sites 

    ie = threadIdx.x + threadIdx.y*blockDim.x;  
    x = (2*ie)%Nx;
    y = ((2*ie)/Nx)%Nx;
    parity=(x+y)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice

    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*nx;
    old_spin = spin[i];
    new_spin = -old_spin;
    if(x == 0){
        l = s_L[(nx-1)+y*nx];
        r = spin[i+1];
    }
    else if(x == nx-1){
        l = spin[i-1];
        r = s_R[y*nx];
    }
    else{
        l = spin[i-1];
        r = spin[i+1];
    }

    if(y == 0){
        t = spin[i+nx];
        b = s_B[x+y*(nx-1)];
    }
    else if(y == ny-1){
        t = s_T[x];
        b = spin[i-nx];
    }
    else{
        t = spin[i+nx];
        b = spin[i-nx];
    }

    spins = l+r+t+b;
    de = -(new_spin - old_spin)*(spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }
    
    __syncthreads();
 
}   

int main(void) {
  int NGPU;
  int cpu_thread_id;
  int *Dev; //GPU device number
  int Lx,Ly; //latice size in each GPU
  int NGx,NGy; //(NGx*NGy) = NGPU
  int nx,ny; 		// # of sites in x and y directions respectively
  int ns; 		// ns = nx*ny, total # of sites
  int *ffw;      	// forward index
  int *bbw; 	        // backward index
  int nt; 		// # of sweeps for thermalization
  int nm; 		// # of measurements
  int im; 		// interval between successive measurements
  int nd; 		// # of sweeps between displaying results
  int nb; 		// # of sweeps before saving spin configurations
  int sweeps; 		// total # of sweeps at each temperature
  int k1, k2;           // right, top
  int istart; 		// istart = (0: cold start/1: hot start)
  double T; 		// temperature
  double B; 		// external magnetic field
  double energy; 	// total energy of the system
  double mag; 		// total magnetization of the system
  double te; 		// accumulator for energy
  double tm; 		// accumulator for mag
  double count; 	// counter for # of measurements
  double M; 		// magnetization per site, < M >
  double E; 		// energy per site, < E >
  double E_ex; 		// exact solution of < E >
  double M_ex; 		// exact solution of < M >

  int gid;              // GPU_ID
  float gputime;
  float flops;

  printf("Enter the Number of GPU (NGx,NGy): ");
  scanf("%d %d",&NGx,&NGy);
  printf("%d %d\n",NGx,NGy);
  NGPU = NGx * NGy;
  Dev = (int*)malloc(NGPU * sizeof(int));
  for(int i = 0; i < NGPU; i++){
      printf("Enter the GPU ID(0/1..)\n");
      scanf("%d",&Dev[i]);
      printf("%d\n",Dev[i]);
  }
  
  // Error code to check return values for CUDA calls
  /*cudaError_t err = cudaSuccess;
  err = cudaSetDevice(gid);
  if(err != cudaSuccess) {
    printf("!!! Cannot select GPU with device ID = %d\n", gid);
    exit(1);
  }
  printf("Select GPU with device ID = %d\n", gid);
  cudaSetDevice(gid);
  */

  printf("Ising Model on 2D Square Lattice with p.b.c.\n");
  printf("============================================\n");
  printf("Enter the number of sites in each dimension (<= 1000)\n");
  scanf("%d",&nx);
  printf("%d\n",nx);
  ny=nx;
  ns=nx*ny;
  ffw = (int*)malloc(nx*sizeof(int));
  bbw = (int*)malloc(nx*sizeof(int));
  for(int i=0; i<nx; i++) {
    ffw[i]=(i+1)%nx;
    bbw[i]=(i-1+nx)%nx;
  }
    
  if(nx % NGx){
      printf("!!! Invalid partition of lattice: Nx %% NGx != 0\n");
      exit(1);
  }

  if(ny % NGy){
      printf("!!! Invalid partition of latice: Ny %% NGy != 0\n");
      exit(1);
  }
  Lx = nx/NGx;
  Ly = ny/NGy;


  spin = (int*)malloc(ns*sizeof(int));          // host spin variables
  h_rng = (float*)malloc(ns*sizeof(float));     // host random numbers

  printf("Enter the # of sweeps for thermalization\n");
  scanf("%d",&nt);
  printf("%d\n",nt);
  printf("Enter the # of measurements\n");
  scanf("%d",&nm);
  printf("%d\n",nm);
  printf("Enter the interval between successive measurements\n");
  scanf("%d",&im);
  printf("%d\n",im);
  printf("Enter the display interval\n");
  scanf("%d",&nd);
  printf("%d\n",nd);
  printf("Enter the interval for saving spin configuration\n");
  scanf("%d",&nb);
  printf("%d\n",nb);
  printf("Enter the temperature (in units of J/k)\n");
  scanf("%lf",&T);
  printf("%lf\n",T);
  printf("Enter the external magnetization\n");
  scanf("%lf",&B);
  printf("%lf\n",B);
  printf("Initialize spins configurations :\n");
  printf(" 0: cold start \n");
  printf(" 1: hot start \n");
  scanf("%d",&istart);
  printf("%d\n",istart);
 
  // Set the number of threads (tx,ty) per block

  int tx,ty;
  printf("Enter the number of threads (tx,ty) per block: ");
  printf("For even/odd updating, tx=ty/2 is assumed: ");
  scanf("%d %d",&tx, &ty);
  printf("%d %d\n",tx, ty);
  if(2*tx != ty) exit(0);
  if(tx*ty > 1024) {
    printf("The number of threads per block must be less than 1024 ! \n");
    exit(0);
  }
  dim3 threads(tx,ty);

  // The total number of threads in the grid is equal to (nx/2)*ny = ns/2 

  int bx = nx/tx/2;
  if(bx*tx*2 != nx) {
    printf("The block size in x is incorrect\n");
    exit(0);
  }
  int by = ny/ty;
  if(by*ty != ny) {
    printf("The block size in y is incorrect\n");
    exit(0);
  }
  if((bx > 65535)||(by > 65535)) {
    printf("The grid size exceeds the limit ! \n");
    exit(0);
  }
  dim3 blocks(bx,by);
  printf("The dimension of the grid is (%d, %d)\n",bx,by);

  if(istart == 0) {
    for(int j=0; j<ns; j++) {       // cold start
      spin[j] = 1;
    }
  }
  else {
    for(int j=0; j<ns; j++) {     // hot start
      if(rand()/(float)RAND_MAX > 0.5) { 
        spin[j] = 1;
      }
      else {
        spin[j] = -1;
      }
    }
  }

  FILE *output;            
  output = fopen("ising2d_Ngpu_gmem.dat","w");
  /*
  FILE *output3;
  output3 = fopen("spin_1gpu_gmem.dat","w");   
  */


  if(B == 0.0) {
    exact_2d(T,B,&E_ex,&M_ex);
    fprintf(output,"T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    fprintf(output,"T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }
  fprintf(output,"     E           M        \n");
  fprintf(output,"--------------------------\n");

  printf("Thermalizing\n");
  printf("sweeps   < E >     < M >\n");
  printf("---------------------------------\n");
  fflush(stdout);

  te=0.0;                          //  initialize the accumulators
  tm=0.0;
  count=0.0;
  sweeps=nt+nm*im;                 //  total # of sweeps


  int **d_1;
  d_1 = (int**)malloc(NGPU*sizeof(int*));
  
  omp_set_num_threads(NGPU);

  //Enable peer to peer communication
  #pragma omp parallel private(cpu_thread_id)
  {
      int cpuid_x,cpuid_y;
      cpu_thread_id = omp_get_thread_num();
      cpuid_x = cpu_thread_id % NGx;
      cpuid_y = cpu_thread_id / NGx;
      cudaSetDevice(Dev[cpu_thread_id]);

      int cpuid_r = ((cpuid_x + 1) % NGx) + cpuid_y*NGx; // GPU on the right
      cudaDeviceEnablePeerAccess(Dev[cpuid_r],0);
      int cpuid_l = ((cpuid_x + NGx-1)%NGx) + cpuid_y*NGx; // GPU on the left
      cudaDeviceEnablePeerAccess(Dev[cpuid_l],0);
      int cpuid_t = cpuid_x + ((cpuid_y+1)%NGy)*NGx;
      cudaDeviceEnablePeerAccess(Dev[cpuid_t],0);
      int cpuid_b = cpuid_x + ((cpuid_y+NGy-1)%NGy)*NGx;
      cudaDeviceEnablePeerAccess(Dev[cpuid_b],0);
      
      //Allocate vectors in device memory
      cudaMalloc((void**)&d_1[cpu_thread_id], ns*sizeof(int)/NGPU);

      for(int i = 0; i < Ly; i++)
      {
          int *h,*d;
          h = spin + cpuid_x*Lx + (cpuid_y*Ly+i)*nx;
          d = d_1[cpu_thread_id] + i*Lx;
          cudaMemcpy(d,h,Lx*sizeof(int),cudaMemcpyHostToDevice);
      }
  }

  // create the timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //start the timer
  cudaEventRecord(start,0);

  for(int swp=0; swp<nt; swp++) {      // thermalization
    rng_MT(h_rng, ns);                                  // generate ns random numbers 
    #pragma omp parallel private(cpu_thread_id)
    {
        int cpuid_x,cpuid_y;
        cpu_thread_id = omp_get_thread_num();
        cpuid_x = cpu_thread_id % NGx;
        cpuid_y = cpu_thread_id / NGx;
        cudaSetDevice(Dev[cpu_thread_id]);
        int *dL,*dR,*dT,*dB;
        dL = d_1[(cpuid_x-1+NGx)%NGx+cpuid_y*NGx];
        dR = d_1[(cpuid_x+1)%NGx+cpuid_y*NGx];
        dB = d_1[cpuid_x+((cpuid_y-1+NGy)%NGy)*NGx];
        dT = d_1[cpuid_x+(cpuid_y+1)%NGy*NGx];

        float *d_rng;
        cudaMalloc((void**)&d_rng,ns/NGPU*sizeof(float));
        cudaMemcpy(d_rng,h_rng+nx*cpu_thread_id/NGPU,ns/NGPU*sizeof(float),cudaMemcpyHostToDevice);

        metro_gmem_even<<<blocks,threads>>>(d_1[cpu_thread_id], d_rng,dL,dR,dT,dB, B, T);    // updating with Metropolis algorithm
        metro_gmem_odd<<<blocks,threads>>>(d_1[cpu_thread_id], d_rng,dL,dR,dT,dB, B, T);     // updating with Metropolis algorithm
        cudaFree(d_rng);
    }
  }
  cudaDeviceSynchronize();


  for(int swp=nt; swp<sweeps; swp++) {

    rng_MT(h_rng, ns);                                  // generate ns random numbers 
    #pragma omp parallel private(cpu_thread_id)
    {
        int cpuid_x,cpuid_y;
        cpu_thread_id = omp_get_thread_num();
        cpuid_x = cpu_thread_id % NGx;
        cpuid_y = cpu_thread_id / NGx;
        cudaSetDevice(Dev[cpu_thread_id]);
        int *dL,*dR,*dT,*dB;
        dL = d_1[(cpuid_x-1+NGx)%NGx+cpuid_y*NGx];
        dR = d_1[(cpuid_x+1)%NGx+cpuid_y*NGx];
        dB = d_1[cpuid_x+((cpuid_y-1+NGy)%NGy)*NGx];
        dT = d_1[cpuid_x+(cpuid_y+1)%NGy*NGx];

        float *d_rng;
        cudaMalloc((void**)&d_rng,ns/NGPU*sizeof(float));
        cudaMemcpy(d_rng,h_rng+nx*cpu_thread_id/NGPU,ns/NGPU*sizeof(float),cudaMemcpyHostToDevice);

        metro_gmem_even<<<blocks,threads>>>(d_1[cpu_thread_id], d_rng,dL,dR,dT,dB, B, T);    // updating with Metropolis algorithm
        metro_gmem_odd<<<blocks,threads>>>(d_1[cpu_thread_id], d_rng,dL,dR,dT,dB, B, T);     // updating with Metropolis algorithm
        cudaDeviceSynchronize();
        cudaFree(d_rng);
    }

    int k; 
    if(swp%im == 0) {
      #pragma omp parallel private(cpu_thread_id)
        {
            int cpuid_x,cpuid_y;
            cpu_thread_id = omp_get_thread_num();
            cpuid_x = cpu_thread_id % NGx;
            cpuid_y = cpu_thread_id / NGx;
            cudaSetDevice(Dev[cpu_thread_id]);

            for(int i = 0; i < Ly; i++){
                int *g, *d;
                g = spin + cpuid_x*Lx + (cpuid_y*Ly+i)*nx;
                d = d_1[cpu_thread_id] + i*Lx;
                cudaMemcpy(g,d,Lx*sizeof(int),cudaMemcpyDeviceToHost);
            }
        }
      mag=0.0;
      energy=0.0;
      for(int j=0; j<ny; j++) {
        for(int i=0; i<nx; i++) {
          k = i + j*nx;
          k1 = ffw[i] + j*nx;
          k2 = i + ffw[j]*nx;
          mag = mag + spin[k]; // total magnetization;
          energy = energy - spin[k]*(spin[k1] + spin[k2]);  // total bond energy;
        }
      }
      energy = energy - B*mag;
      te = te + energy;
      tm = tm + mag;
      count = count + 1.0;
      fprintf(output, "%.5e  %.5e\n", energy/(double)ns, mag/(double)ns);  // save the raw data 
    }
    if(swp%nd == 0) {
      E = te/(count*(double)(ns));
      M = tm/(count*(double)(ns));
      printf("%d  %.5e  %.5e\n", swp, E, M);
    }
  }
  fclose(output);      
  printf("---------------------------------\n");
  if(B == 0.0) {
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }

  // stop the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&gputime, start, stop);
  printf("Processing time for GPU: %f (ms) \n",gputime);
  flops = 7.0*nx*nx*sweeps;
  printf("GPU Gflops: %lf\n",flops/(1000000.0*gputime));

  // destroy the timer
  cudaEventDestroy(start);
  cudaEventDestroy(stop);


  free(spin);
  free(h_rng);

  return 0;
}
          
          
// Exact solution of 2d Ising model on the infinite lattice

void exact_2d(double T, double B, double *E, double *M)
{
  double x, y;
  double z, Tc, K, K1;
  const double pi = acos(-1.0);
    
  K = 2.0/T;
  if(B == 0.0) {
    Tc = -2.0/log(sqrt(2.0) - 1.0); // critical temperature;
    if(T > Tc) {
      *M = 0.0;
    }
    else if(T < Tc) {
      z = exp(-K);
      *M = pow(1.0 + z*z,0.25)*pow(1.0 - 6.0*z*z + pow(z,4),0.125)/sqrt(1.0 - z*z);
    }
    x = 0.5*pi;
    y = 2.0*sinh(K)/pow(cosh(K),2);
    K1 = ellf(x, y);
    *E = -1.0/tanh(K)*(1. + 2.0/pi*K1*(2.0*pow(tanh(K),2) - 1.0));
  }
  else
    printf("Exact solution is only known for B=0 !\n");
    
  return;
}


/*******
* ellf *      Elliptic integral of the 1st kind 
*******/

double ellf(double phi, double ak)
{
  double ellf;
  double s;

  s=sin(phi);
  ellf=s*rf(pow(cos(phi),2),(1.0-s*ak)*(1.0+s*ak),1.0);

  return ellf;
}

double rf(double x, double y, double z)
{
  double rf,ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4;
  ERRTOL=0.08; 
  TINY=1.5e-38; 
  BIG=3.0e37; 
  THIRD=1.0/3.0;
  C1=1.0/24.0; 
  C2=0.1; 
  C3=3.0/44.0; 
  C4=1.0/14.0;
  double alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;
    
  if(min(x,y,z) < 0 || min(x+y,x+z,y+z) < TINY || max(x,y,z) > BIG) {
    printf("invalid arguments in rf\n");
    exit(1);
  }

  xt=x;
  yt=y;
  zt=z;

  do {
    sqrtx=sqrt(xt);
    sqrty=sqrt(yt);
    sqrtz=sqrt(zt);
    alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
    xt=0.25*(xt+alamb);
    yt=0.25*(yt+alamb);
    zt=0.25*(zt+alamb);
    ave=THIRD*(xt+yt+zt);
    delx=(ave-xt)/ave;
    dely=(ave-yt)/ave;
    delz=(ave-zt)/ave;
  } 
  while (max(abs(delx),abs(dely),abs(delz)) > ERRTOL);

  e2=delx*dely-pow(delz,2);
  e3=delx*dely*delz;
  rf=(1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave);
    
  return rf;
}

double min(double x, double y, double z)
{
  double m;

  m = (x < y) ? x : y;
  m = (m < z) ? m : z;

  return m;
}

double max(double x, double y, double z)
{
  double m;

  m = (x > y) ? x : y;
  m = (m > z) ? m : z;

  return m;
}

void rng_MT(float* data, int n)   // RNG with uniform distribution in (0,1)
{
    for(int i = 0; i < n; i++)
      data[i] = rand()/(float)RAND_MAX; 
}

