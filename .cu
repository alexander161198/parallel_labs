#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctime> 
#include <iostream>
#include "inverse.h"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
using namespace boost::numeric::ublas;


//граничное условие 2 рода dT/dn=10
#define TT 10

matrix <double> all_calc (double a1,double a2,double L1,double L2,int N_i,int N_j, int k) {
	double x1 = (L1-a1)/2;
    double y1 = (L2-a2)/2; 
    double x2 = (L1+a1)/2; 
    double y2 = (L2+a2)/2; 

    double dx = L1/(N_i - 1);
    double dy = L2/(N_j - 1);

    //Инициализируем начальные массивы

    matrix<double> M (N_i*N_j,N_i*N_j);
    vector<double> R (N_i*N_j);

    for (int i = 0; i < N_i; i++) {
        for (int j = 0; j < N_j; j++) {
            double x=i*dx;
            double y=j*dy;
            if (i==0) { //самая левая граница dT/dn=10
                M(j+N_j*(i),j+N_j*(i+1)) = 1/dx; //i+1, j
                M(j+N_j*(i),j+N_j*(i)) = -1/dx;//i,j
                R(j + N_j * (i)) = TT;
                continue;
            }

            if (j==0){	//нижняя dT/dn=10
            	M(j+N_j*(i),j+N_j*(i)) = -1/dy;//i, j,
                M(j+N_j*(i),j+N_j*(i)+1) = 1/dy;//i,j+1
                R(j + N_j * (i)) = TT;
            	continue;
            }
            if (j==N_j-1) { //%самая верхняя граница dT/dn =10
            	M(j+N_j*(i),j+N_j*(i)) = -1/dy;//i, j,
                M(j+N_j*(i),j+N_j*(i)-1) = 1/dy;//i,j-1
                R(j + N_j * (i)) = TT;
                continue;
            }
            if (i==N_i-1) { //самая правая граница dT/dn=10
            	M(j+N_j*(i), j+N_j*(i-1)) = 1/dx; //i-1, j
                M(j+N_j*(i),j+N_j*(i)) = -1/dx;//i,j
                R(j + N_j * (i)) = TT;
                continue;
            }
            //внутренняя граница нижняя
            if ((y<=y1) &&((y+dy)>y1) && (x<=x2) &&(x>=x1)) { 
                M(j + N_j * (i),j + N_j * (i)) = 1;//i, j
                R(j + N_j * (i)) = 100;
                continue;
            }
            //внутренняя граница верхняя
            if ((y>y2) &&((y-dy)<y2) && (x<=x2) && (x>=x1)) { 
                M(j + N_j * (i),j + N_j * (i)) = 1;//i, j
                R(j + N_j * (i)) = 100;
                continue;
            }
            //внутренняя граница правая
            if (((x-dx)<x2) &&(x>=x2) && (y<=y2) &&(y>=y1)) {
            	M(j + N_j * (i),j + N_j * (i)) = 1;//i, j
                R(j + N_j * (i)) = 100;
                continue;
            }
            //внутренняя граница левая
            if (((x+dx)>x1) &&(x<=x1) && (y<=y2) &&(y>=y1)) { 
                M(j + N_j * (i),j + N_j * (i)) = 1;//i, j
                R(j + N_j * (i)) = 100;
                continue;
            }

            if ((x>=x1) && (x<=x2) && (y<=y2) && (y>=y1)) {
				M(j + N_j * (i),j + N_j * (i)) = 1;//i, j
                R(j + N_j * (i)) = 100;
                continue;
            }
            if ((x<=x1) || (x>=x2) || ((x>x1 && x<x2) && ((y<=y1) || (y>=y2)))) { //%Основная часть
                M(j+N_j*(i),j+N_j*(i+1)) =a1/(dx*dx); //i+1, j
                M(j+N_j*(i),j+N_j*(i)) = -2*a1/(dx*dx)-2*a2/(dy*dy);//i,j
                M(j+N_j*(i),j+N_j*(i-1)) = a1/(dx*dx);//i-1, j,
                M(j+N_j*(i),j+N_j*(i)+1) =a2/(dy*dy);//i,j+1
                M(j+N_j*(i),j+N_j*(i)-1) =a2/(dy*dy);//i,j-1
                continue;
            }
            M(j+N_j*(i),j+N_j*(i)) = 1;//i,j
            R(j+N_j*(i)) = 0;
        }
    }

    matrix<double> iM (N_i*N_j,N_i*N_j);
    bool flag = false;
    iM = gjinverse (M, flag);
    vector<double> T1 = prod(iM,R);

    matrix<double> z1(N_i,N_j);
    matrix<double> x_f(N_i,N_j);
    matrix<double> y_f(N_i,N_j);

    for (int i = 0; i < N_i; i++) {
        for (int j = 0; j < N_j; j++) {
            x_f(i,j)=i*dx;
            y_f(i,j)=j*dy;
            z1(i,j)=T1(i*N_j+j);
        }
    }

    return z1;
}


int main() {
    double a1 = 2;
    double a2 = 2;
    double L1 = 12;
    double L2 = 6;
   	int N_i = 14;
	int N_j = 14;

    /*clock_t start = clock();*/

    matrix<double> z1 = all_calc(a1,a2,L1,L2, N_i, N_j, 1);
    matrix<double> z2 = all_calc(a1,a2,L1,L2, N_i*2, N_j*2, 1);

   /* clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds); */
    return 1;
} 
