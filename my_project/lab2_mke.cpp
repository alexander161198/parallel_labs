#include <memory.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <omp.h>

using namespace std;

class FEM
{
	public:

	FEM(int n);
	~FEM();
	bool SolveSystem();
	bool GenerateMatrix();
	bool GenerateMatrix_linear();
	double getAnalyt(double);
	void printMatrix();
	void printFuncFEM();
	double ** A;					//матрица и вектор невязок
	double * X;						//вектор неизвестных
	double u_xN, u_x0;				//граничные условия
	double x0;						//начальное значение X
	double xN;
	int N;							//размер матрицы
	double L;
};

//конструктор
FEM::FEM(int n)
{
	N = n;	
	x0 = -1.0;
	xN = 10.0;
	u_x0 = 2;
	u_xN = 0;
	int i,j;
	A = new double * [N];

	for (i = 0; i < N; i++)	
	{
		A[i] = new double[N + 1];
	}

	for (i = 0; i < N; i++)
	{
		for(j = 0; j < N + 1; j++)
		{
			A[i][j] = 0.0;
		}
	}

	X = new double[N];
}

//диструктор
FEM::~FEM()
{
	for (int i = 0; i < N; i++)
	{
		delete [] A[i];
	}
	delete [] A;
	delete [] X;
}

bool FEM::SolveSystem()
{
	double elem = 0;
	int firstNotNull = 0;
	int i,j,cur;

	timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    #pragma omp parallel for num_threads(8) private(i, j, cur, elem, firstNotNull)
	for (cur = 0; cur < N; cur++)
	{
		if (A[cur][cur] == 0)					//если на диагонали 0
		{
			firstNotNull = -1;
			//#pragma omp for
			for (i = cur; i < N; i++)			//ищется 1 ненулевой элемент в столбце
			{
				if (A[i][cur] != 0)
				{
					firstNotNull = i;
				}
			}
			if (firstNotNull == -1)
			{}
			if (firstNotNull != cur)			//перестановка строк
			{
				double * tmp = new double[N + 1];
				memcpy(tmp, A[cur], (N + 1) * sizeof(double));
				memcpy(A[cur], A[firstNotNull], (N + 1) * sizeof(double));
				memcpy(A[firstNotNull], tmp, (N + 1) * sizeof(double));
				delete [] tmp;
			}
		}
		if ((elem = A[cur][cur]) != 1)			//делаем на диагонали 1
		{
			//#pragma omp for
			for (j = cur; j < N + 1; j++)
			{
				A[cur][j] /= elem;
			}
		}
		//#pragma omp for
		for (i = cur + 1; i < N; i++)			//приведение к верхнедиагональному виду
		{
			elem = A[i][cur];
			if (elem)
			{
				for (j = cur; j < N + 1; j++)
				{
					A[i][j] /= elem;
					A[i][j] -= A[cur][j];
				}
			}
		}
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken: %lf sec.\n",end.tv_sec-start.tv_sec+ 0.000000001*(end.tv_nsec-start.tv_nsec));

// printMatrix();
	double sum = 0.0;
	X[N - 1] = A[N - 1][N];

	#pragma omp parallel num_threads(8)  default(shared)
	{
	for (i = N - 2; i >= 0; i--)				//получение решения
	{
		#pragma omp for
		for (j = i + 1; j < N; j++)
		{
			sum += A[i][j] * X[j];
		}
		X[i] = A[i][N] - sum;
		sum = 0.0;
	}
}

	return 1;
}

//получение значения в конкретном узле
double FEM::getAnalyt(double x)
{
	double sq = sqrt((double)18 / 7);
	double C1 = (5 + (exp(11 * sq))) / (3 * (exp(-sq) - exp(21 * sq)));
	double C2 = (5 + (exp(-11 * sq))) / (3 * (exp(sq) - exp(-21 * sq)));
	double ans = C1 * (exp(sq * x)) + C2 * (exp(-sq * x)) + (double)1/3;
	return (ans);
}

//генерация матрицы с конкретными числовыми значениями
bool FEM::GenerateMatrix()
{
	int i;
	L = (xN - x0) / (double)(N - 1);			//длина

	double m11 = 48 * L / 5 + 112 / (3 * L);
	double m12 = 6 * L / 5 - 56 / (3 * L);
	double m13 = 6 * L / 5 - 56 / (3 * L);
	double m21 = 6 * L / 5 - 56 / (3 * L);
	double m22 = 12 * L / 5 + 49 / (3 * L);
	double m23 = 7 / (3 * L) - 3 * L / 5;
	double m31 = 6 * L / 5 - 56 / (3 * L);
	double m32 = 7 / (3 * L) - 3 * L / 5;
	double m33 = 12 * L / 5 + 49 / (3 * L);

	double t1 = 4 * L;
	double t2 = L;
	double t3 = L;

	m22 = m22 * m11 - m12 * m21;
	m23 = m23 * m11 - m13 * m21;
	m32 = m32 * m11 - m12 * m31;
	m33 = m33 * m11 - m13 * m31;

	t2 = t2 * m11 - t1 * m21;
	t3 = t3 * m11 - t1 * m31;

	//последовательно заполняем глобальную матрицу 
	//локальными, свдигаясь по диагонали
	for (i = 0; i < N; i++)
	{
		A[i][i] += m22;
		A[i][i+1] += m23;
		if (i != (N - 1))
		{
			A[i+1][i] += m32;
			A[i+1][i+1] += m33;
			A[i+1][N] += t3;
		}

		A[i][N] += t2;
	}

	/* for (i = 0; i < N; i++)
	{
		A[N-1][i] = 0.0;
	}*/

	A[N-1][N-2] = 0;
	A[N-1][N-1] = 1.0;
	A[N-1][N] = u_xN;

	/* for (i = 0; i < N; i++)
	{
		A[0][i] = 0.0;
	}*/

	A[0][1] = 0;
	A[0][0] = 1.0;
	A[0][N] = u_x0;

	//printMatrix();
	return 1;
}

bool FEM::GenerateMatrix_linear()
{
	int i;
	L = (xN - x0) / (double)(N - 1);			//длина элемента

	double m11 = 7 / L + (6 * L);
	double m12 = -7 / L + (3 * L);
	double m21 = -7 / L + (3 * L);
	double m22 = 7 / L + (6 * L);

	double t1 = 3 * L;
	double t2 = 3 * L;

	//последовательно заполняем глобальную матрицу 
	//локальными, свдигаясь по диагонали
	for (i = 0; i < N; i++)
	{
		A[i][i] += m11;
		A[i][i+1] += m12;
		if (i != (N - 1))
		{
			A[i+1][i] += m21;
			A[i+1][i+1] += m22;
			A[i+1][N] += t2;
		}

		A[i][N] += t1;
	}

	/* for (i = 0; i < N; i++)
	{
		A[N-1][i] = 0.0;
	}*/

	A[N-1][N-2] = 0;
	A[N-1][N-1] = 1.0;
	A[N-1][N] = u_xN;

	/* for (i = 0; i < N; i++)
	{
		A[0][i] = 0.0;
	}*/

	A[0][1] = 0;
	A[0][0] = 1.0;
	A[0][N] = u_x0;

	//printMatrix();
	return 1;
}

void FEM::printMatrix()
{
	FILE* fp;
	fp = fopen("matrix.txt","w");
	int i,j;
	fprintf(fp,"\n");

	for (i = 0; i < N ; i++)
	{
		for (j = 0; j < N + 1; j++)
		{
			if (A[i][j] == 0) 
				fprintf(fp,"----- ");
			else 
				fprintf(fp,"%+4.2f ", A[i][j]);
		}

		fprintf(fp,"\n");
	}

	fclose(fp);
}

void FEM::printFuncFEM()
{
	int i;
	double max_p = 0.0, value_node = 0;
	int step = 0;
	int w1 = 4;								//ширина столбцов
	int w2 = 18;
	L = (xN - x0) / (double)(N - 1);			//длина элемента
	//cout<<L<<endl;

	ofstream fout("results.txt");
	fout.width(w1);
	fout<<"Node";
	//fout<<" | ";
	fout.width(w2);
	fout<<"FEM";
	//fout<<" | ";
	fout.width(w2);
	fout<<"Analytic";
	fout.width(w2);
	fout<<"Error";
	fout.width(w2);
	fout<<"X"<<endl;
	fout.width(4 * w2 + w1 + 6);
	fout.fill('-');
	fout<<""<<endl;
	fout.fill(' ');
	fout.precision(8);
	//fout.setf(ios::showpoint);
	// fout.setf(ios::scientific);

	for (i = 0; i < N; i++)						//заполнение значений
	{
		value_node = getAnalyt(x0 + L*i);
		fout.width(w1);
		fout<<i;
		//fout<<" | ";
		fout.width(w2);
		fout<<X[i];
		//fout<<" | ";
		fout.width(w2);
		fout<<value_node;
		fout.width(w2);
		fout<<fabs(-X[i] + value_node);
		fout.width(w2);
		fout<<x0+L*i<<endl;

		if(max_p < fabs(-X[i] + value_node))		//максимальная погрешность
		{
			max_p = fabs(-X[i] + value_node);
			step = i;
		}
	}

	fout.width(2 * w2 + w1 + 6);
	fout.fill('-');
	fout<<""<<endl;
	fout.fill(' ');
	fout.precision(8);
	fout<<"Maximum error = "<<max_p<<" in "<<step<<" node"<<endl;
	fout.close();
}


int main()
{
	FEM equation(40000);
	//equation.printMatrix();
	equation.GenerateMatrix();
	//equation.GenerateMatrix_linear();
	//equation.printMatrix();
	equation.SolveSystem();
	equation.printFuncFEM();
	return 0;
} 