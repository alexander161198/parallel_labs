#include<stdio.h>
#include<iostream> 
#include<stdlib.h> 
#include<math.h> 
#include<cmath> 

using namespace std; 

#define X 32 
#define Y 32 
#define X_IN 10 
#define Y_IN 10

#define N ((X+1) * (Y+1)) 
#define TIME 10

double h_x = 0.2; 
double h_y = 0.2; 
double h_t = 1;

void gauss(double** matrix, double solve[], int n)
{ 
	int j1;
	double **matr = new double *[n]; 
	for (int i = 0; i < n; i++) 
		matr[i] = new double [n];
	double slv[N];

	for (int i = 0; i < N; i++)
	{ 
		for (int j = 0; j < N; j++)
		{
			matr[i][j] = matrix[i][j]; 
		}
			slv[i] = solve[i]; 
	}

	// прямой ход Гаусса 
	for(int j = 0; j < n; j++)
	{
		for (int i = j + 1; i < n; i++)
		{
			double koef = matr[i][j] / matr[j][j]; 
			for(int k = j; k < n; k++)
			{
				matr[i][k] -= koef * matr[j][k]; 
			}

			slv[i] -= koef * slv[j]; 
		}
	}

		// обратный ход Гаусса
	for(int i = n - 1; i >= 0; i--)
	{
		double sum = 0.0;
		for(int j = i + 1; j < n; j++)
		{ 
			sum += solve[j] * matr[i][j];
		}

		solve[i] = (slv[i] - sum) / matr[i][i]; 
	}

	for (int i = 0; i < n; i++) 
		delete matr[i];

	delete matr; 
}


int inside_tube(int i, int j, int flag)
{ 
	if(flag)
	{
		if((j >= (X - X_IN) / 2) && (j <= (X + X_IN) / 2) && (i >= (Y - Y_IN) / 2) && (i <= (Y + Y_IN) / 2) )
		{
			return 1; 
		}
		else
		{
			return 0; 
		}
	} 

	else
	{
		if((j > (X - X_IN) / 2) && (j < (X + X_IN) / 2) && (i > (Y - Y_IN) / 2) && (i < (Y + Y_IN) / 2) )
		{
			return 1; 
		}
		else
		{
			return 0; 
		}
	} 
}

double sss[2];

int main()
{
	//FILE* new_file = fopen("stat", "w");
	double **matr = new double*[N]; 

	for (int i = 0; i < N; i++) 
		matr[i] = new double [N]; 

	// заполняем матрицу нулями 
	for(int i = 0; i < N; ++i)
	{ 
		for (int j = 0; j < N; ++j)
		{
			matr[i][j] = 0.0; 
		}
	}

	double* solve = new double [N]; 
	double* x = new double [N]; 

	for(int i = 0; i <= Y; ++i)
	{ 
		for (int j = 0; j <= X; ++j)
		{
			if( inside_tube(i, j, 1) )
			{ 
				if(inside_tube(i, j, 0))
				{
				// температура будет решением для уравнения Гаусса во всех точках,
				// кроме пограничных, где заданы ГУ 3 рода 
					solve[i * (X + 1) + j] = 0;
				} 
				else
				{
					solve[i * (X + 1) + j] = 100.0; 
				}
			} 
			else
			{
				solve[i * (X + 1) + j] = 10.0; 
			}
		} 
	}

	// ГУ первого рода на внутренней стенке. Температура в узлах будет сохраняться
	for(int i = (Y - Y_IN) / 2; i <= (Y + Y_IN) / 2; ++i)
	{ 
		for (int j = (X - X_IN) / 2; j <= (X + X_IN) / 2; ++j)
		{
			int k = i * (X + 1) + j; 
			matr[k][k] = 1;
		} 
	}

	// ГУ 3го рода на внешних стенках 
	for (int j = 1; j < X; ++j)
	{
		int i = 0;
		int k_vnesh = j;
		int k_vnytr = j + (X + 1); 
		solve[k_vnesh] = 0; 

		matr[k_vnesh][k_vnesh] = 1 / h_y + 1; 
		matr[k_vnesh][k_vnytr] = -1 / h_y;
	} 

	for (int j = 1; j < X; ++j)
	{ 
		int i = Y;
		int k_vnesh = j + (X + 1) * i;
		int k_vnytr = j + (X + 1) * (i - 1); 

		solve[k_vnesh] = 0; 
		matr[k_vnesh][k_vnesh] = 1 / h_y + 1; 
		matr[k_vnesh][k_vnytr] = -1 / h_y;
	}

	for (int i = 0; i <= Y; ++i)
	{ 
		int j = 0;
		int k_vnesh = i * (X + 1); 
		int k_vnytr = i * (X + 1) + 1; 
		solve[k_vnesh] = 0;

		matr[k_vnesh][k_vnesh] = 1 / h_x + 1; 
		matr[k_vnesh][k_vnytr] = - 1 / h_x;
	}

	for (int i = 0; i <= Y; ++i)
	{ 
		int j = X;
		int k_vnesh = i * (X + 1) + j;
		int k_vnytr = i * (X + 1) + j - 1; 
		solve[k_vnesh] = 0; 

		matr[k_vnesh][k_vnesh] = 1 / h_x + 1; 
		matr[k_vnesh][k_vnytr] = - 1 / h_x;
	}

	// Внутренние узлы pаданные с помощью стандартной неявной схемы (по формуле 2)
	for (int i = 1; i < Y; ++i)
	{ 
		for (int j = 1; j < X; ++j)
		{
			if( inside_tube(i, j, 1) )
			{ 
				continue;
			}
			int ij = i * (X + 1) + j;
			int im1j = (i - 1) * (X + 1) + j; 
			int ip1j = (i + 1) * (X + 1) + j; 
			int ijm1 = i * (X + 1) + j - 1; 
			int ijp1 = i * (X + 1) + j + 1;

			matr[ij][ij] = 2 * ( h_t / (h_x * h_x) + h_t / (h_y * h_y) ) + 1;
			matr[ij][im1j] = - h_t / (h_y * h_y); 
			matr[ij][ip1j] = - h_t / (h_y * h_y); matr[ij][ijm1] = - h_t / (h_x * h_x); matr[ij][ijp1] = - h_t / (h_x * h_x);
			
			solve[ij] = x[ij]; 
		}
	}
			timespec start, end;
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	for (int k = 0; k <= TIME / h_t; k+= h_t) 
	{ 
		cout << k << endl;
		// в solve после решения системы уравнений методом Гаусса буде храниться температура трубы.
		// Чтобы избежать копирования темературы в другой массив, заметим, что в большинстве узлов
		// получення температура является решением для следующей итерации расчётов.
		// Температура не является решением только на границах трубы, где заданы ГУ 3-го рода.
		// В этих узлах в столбце-ответе должен стоять 0, установим его.
		for (int j = 1; j < X; ++j)
		{ 
			solve[j] = 0;
			solve[j + (X + 1) * Y] = 0; 
		}

		for (int i = 0; i <= Y; ++i)
		{ 
			solve[i * (X + 1)] = 0; 
			solve[i * (X + 1) + X] = 0;
		}

		for (int i = 0; i <= Y; ++i){ }

		gauss(matr, solve, N);

		// теперь в solve хранится температура. Записываем её в файл.
/*		for (int i = 0; i <= Y; i++) 
		{
			for (int j = 0; j <= X; j++) 
			{ 
				fprintf(new_file, "%3.2lf ", solve[i * (X + 1) + j]); 
			}

			fprintf(new_file, "\n"); 
		}

		fprintf(new_file, "\n"); */
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken: %lf sec.\n",end.tv_sec-start.tv_sec+ 0.000000001*(end.tv_nsec-start.tv_nsec));

	//fclose(new_file);

	for(int i = 0; i < N; i++)
	{ 
		delete matr[i];
	}
	delete matr; 
	delete solve; 
	delete x; 

	return 0;
} 