/*
 * To compile, type the command.
 * g++ Deviation_Value.cc -o Deviation_Value
 */
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cctype>
#include<cstdlib>
#include<string>
#include<unistd.h>
using namespace std;
void getCorrelation(double *data, double *cor, int N, int tmax)
{
    double Ai_ave, Aj_ave;
    for(int t = 0; t < tmax; t++){
        cor[t] = 0.0;
        Ai_ave = 0.0;
        Aj_ave = 0.0;
        int count = 0;
        for(int i = 0; i <= N-t; i++){
            count += 1;
            Ai_ave += data[i];
            Aj_ave += data[i+t];
            cor[t] += data[i] * data[i+t];
        }
        Ai_ave /= count;
        Aj_ave /= count;
        cor[t] /= count;
        cor[t] = (cor[t]-Ai_ave*Aj_ave);
    }
}




int main(int argc,char *argv[])
{
    string InputFileName;
    int c;
    char *selectchar;
    while((c = getopt(argc,argv,"s:f:"))!= -1)
       switch(c){
           case 's':
               selectchar = optarg;
               break;
           case 'f':
               InputFileName.assign(optarg);
               break;
           case '?':
               fprintf(stderr,"Illegal option:-%c\n", isprint(optopt)?optopt:'#');
               break;                    
           default:
               abort();
       }
    printf("Selection: %s\n",selectchar);
    printf("Input File Name: %s\n",InputFileName.c_str());
    FILE *inputfile;
    inputfile = fopen(InputFileName.c_str(),"r");
    //inputfile = fopen("CPUT2.dat","r");
    int select = atoi(selectchar);
    int N = 1000; // # of measurement
    int tmax = 1000; // Calculate the autocorrelation time to the same size size as # of measurement

    double *A,tmp;
    double *Acor;
    double *tauA;

    //int select = 0;

    A = (double*)malloc(N*sizeof(double));
    Acor = (double*)malloc(tmax*sizeof(double));
    tauA = (double*)malloc(tmax*sizeof(double));
    for(int i = 0; i < N; i++)
    {
        if(select == 0){
            //printf("select character %s\n",selectchar);
            fscanf(inputfile,"%lf %lf",&A[i],&tmp);
            //printf("%lf\n",A[i]);
        }
        else if(select == 1){
            //printf("select character %s\n",selectchar);
            fscanf(inputfile,"%lf %lf",&tmp,&A[i]);
            //printf("%lf\n",A[i]);
        }
    }

    fclose(inputfile);

    //Calculate the result (<A^2>-<A>^2)/N-1
    double average = 0.0;
    double average2 = 0.0;
    for(int i = 0; i < N; i++)
    {
        average += A[i];
        average2 += A[i] * A[i];
    }

    average /= N;
    average2 /= N;

    double err = sqrt((average2-pow(average,2))/(N-1));
    getCorrelation(A,Acor,N,tmax);

    double Cj;
    int flag = 0,i,j;
    for(i = 1; i <= tmax; i++){
        tauA[i] = 0.5;
        for(j = 1; j <= i; j++){
            Cj = Acor[j]/Acor[0];
            if(Cj < 0){
                flag = 1;
                break;
            }
            tauA[i] += Cj;
        }
        if(flag == 1){
            break;
        }
    }

    if(flag == 0){
        printf("tau haven't saturate yet\n");
    }

    else if(flag == 1){
        double errMC = sqrt(2.0*tauA[i]);
        printf("t = %d, tau_int = %e, sqrt(2*tau_int) = %e\n",i,tauA[i],errMC);
        printf("<A> = %e +- %e\n",average,err*errMC);
    }
}
