#include "system/Plate.cpp"
#include <stdio.h>
#include "system/Components/Temperature.cpp"
#include "system/Components/FirstConstrain.cpp"
#include "system/Components/SecondConstrain.cpp"
#include "parallels/paralel.h"
#include <pthread.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>


#define REENTRANT
#define C 1
#define WIDTH 10.
#define HEIGHT 10.
#define MaxToGnuPlot 101.

double* test = new double[100 * 100];

typedef struct{
	int id;
	Plate* localPlate;
	int width;
	double* localMatrix;
	double** cornerMatrix;

	void init(int p_num , Plate* p_localPlate , int p_width){
		id = p_num;
		localPlate = p_localPlate;
		width = p_width;
	}

} argThread;


Plate* plate;
double** voltage;
int countThreads;
int x;
int y;
double dx , dy;

void printMat(double* mat , int yi , int xi){
	for(int i = 0 ; i < yi ; i++){
		for(int j = 0 ; j < xi ; j++)
			fprintf(stderr, "%3.3f ", mat[i  * xi + j]);
		fprintf(stderr, "\n");
	}
}

Plate* createPlate(int x , int y , double** voltage){
	Plate* plate = new Plate(x * y + 1, voltage);
	for(int i = 1  , j; i < y - 1 ; i++){
		plate -> addComponent(i * x + 1 , i * x + 2 , new Temperature(C * C , dy));
		plate -> addComponent(i + 1 , i + 1 + x , new Temperature(C * C , dx));
		plate -> addComponent(0 , i * x + 1 , new FirstConstrain(100));
		plate -> addComponent(i * x + x - 1 , i * x + x , new SecondConstrain(dy ,-10));
		for(j = 1 ; j < x - 1 ; j++){
			plate -> addComponent(i * x + j + 1 , i * x + j + 2  , new Temperature(C * C , dy));
			plate -> addComponent(i * x + j + 1 , i * x + j + 1 + x , new Temperature(C * C , dx));
		}
	}
	plate -> addComponent(0 , 1 , new FirstConstrain(100));
	plate -> addComponent(0 , x , new FirstConstrain(100));
	plate -> addComponent(0 , x * y , new FirstConstrain(100));
	plate -> addComponent(0 , x * (y - 1) + 1 , new FirstConstrain(100));

	return plate;
}

void* action(void* v_arg){
	argThread* arg = (argThread*)v_arg;
	int countNodes = arg -> localPlate -> getCountNodes() - 1;
	arg -> localMatrix = new double[countNodes * (countNodes + 1)];
	arg -> cornerMatrix = new double*[countNodes - arg -> width];
	double** localVolt = &arg -> localPlate -> getVoltage()[1];
	int countCorner = plate -> getCountNodes() - 1 - arg -> width * countThreads;
	FILE* script = openFile("gnu.sh");

	int* lengthPlates = new int[countThreads];
	double* cornerVoltage = new double[countCorner + arg -> width];

	arg -> localPlate->createStiffMatrix(arg -> localMatrix);

	for(int i = arg -> width , j = 0 ; i < countNodes ; i++ , j++){
		memset(arg -> localMatrix + i * (countNodes + 1) + arg -> width, 0 , (countNodes + 1 - arg -> width) * sizeof(double));
		arg -> cornerMatrix[j] = arg -> localMatrix  + i * (countNodes + 1) + arg -> width;
	}

	for(int i = 0 ; i < arg -> width ; i++){
		for(int j = i + 1 ; j < countNodes; j++){
			double key = arg -> localMatrix[j * (countNodes + 1) + i] / arg -> localMatrix[i * (countNodes + 1) + i];
			for(int k = i; k < countNodes + 1 ; k++)
				arg -> localMatrix[j * (countNodes + 1) + k] -= arg -> localMatrix[i * (countNodes + 1) + k] * key;
		}
	}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(&countNodes , 1 , MPI_INT , lengthPlates , 1 , MPI_INT , 0 , MPI_COMM_WORLD);
		if(arg->id)
			for(int i = 0 ; i < countNodes - arg -> width ; i++)
				MPI_Send(arg -> cornerMatrix[i] , countNodes - arg -> width + 1 , MPI_DOUBLE , 0 , i , MPI_COMM_WORLD);

		if(arg -> id == 0 && countThreads > 1){
			int count = plate -> getCountNodes() - 1;
			double* matrix = new double[count * (count + 1)];
			double** corner = new double*[countCorner];
			MPI_Status status;
			memset(matrix , 0 , count * (count + 1) * sizeof(double));
			plate -> createStiffMatrix(matrix);

			for(int i = 0 , it = arg -> width * countThreads ; i < countCorner ; i++ , it++)
				corner[i] = matrix + it * (count + 1) + arg -> width * countThreads;

			for(int i = 0 , delta = 0 ; i < countThreads ; i++){
				int widthCorner = lengthPlates[i] - arg -> width;
				double* lineCorner;
				if(i > 0) lineCorner = new double[widthCorner + 1];

				for(int j = 0 ; j < widthCorner ; j++){
					if(i > 0)
						MPI_Recv(lineCorner , widthCorner + 1 , MPI_DOUBLE , i , j , MPI_COMM_WORLD , &status);
					else
						lineCorner = arg->cornerMatrix[j];
					for(int k = 0 ; k < widthCorner ; k++)
						corner[j+ delta][k + delta] += lineCorner[k];
					corner[j+ delta][countCorner] += lineCorner[widthCorner];
				}
				delta += widthCorner - (countNodes - arg -> width);
				if(i > 0) delete lineCorner;
			}

			for(int i = 0 ; i < countCorner ; i++){
				for(int j = i + 1 ; j < countCorner ; j++){
					double key = corner[j][i] / corner[i][i];
					for(int k = i; k <= countCorner; k++)
						corner[j][k] -= corner[i][k] * key;
				}
			}

			for(int i = countCorner - 1 ,  h = plate -> getCountNodes() - 1 ; i >= 0 ; i-- , h--){
				for(int j = countCorner - 1 ,  row =  plate -> getCountNodes() - 1  ; j > i ; j-- , row --)
					corner[i][countCorner] -= corner[i][j] * (*voltage[row]);
				cornerVoltage[arg -> width + i] = (*voltage[h]) = corner[i][countCorner] / corner[i][i];
			}

			delete corner;
			delete matrix;

		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(cornerVoltage + arg -> width , plate->getCountNodes() - 1 - arg -> width * countThreads , MPI_DOUBLE , 0 , MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		
		for(int i = arg -> width * countThreads , j = arg -> width ; i < plate->getCountNodes() - 1 ; i++ , j++){
			(*voltage[i + 1]) = cornerVoltage[j];
		}

		double* allVoltage;
		if(arg -> id == 0)
			allVoltage = new double[plate -> getCountNodes()];

		for(int i = arg -> width - 1 ; i >= 0 ; i--){
			for(int j = countNodes - 1 ; j > i; j--)
				arg->localMatrix[i * (countNodes + 1) + countNodes] -= arg->localMatrix[i * (countNodes  + 1) + j] * (*localVolt[j]);
			cornerVoltage[i] = (*localVolt[i]) = arg->localMatrix[i * (countNodes + 1) + countNodes] / arg -> localMatrix[i * (countNodes + 1) + i];
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(cornerVoltage , arg->width , MPI_DOUBLE , &allVoltage[1] , arg->width , MPI_DOUBLE , 0 , MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);


		if(arg -> id == 0){
			for(int i = 0 ; i < countCorner ; i++)
				allVoltage[arg -> width * countThreads + i + 1] = cornerVoltage[arg -> width + i];
			writeToFile(script , allVoltage , plate -> getOrder() , y , x);
		}

	MPI_Barrier(MPI_COMM_WORLD);
	closeFile(script);
	delete lengthPlates;
	if(arg -> id == 0)
		delete allVoltage;
	delete arg -> cornerMatrix;
	delete arg -> localMatrix;
}

int main(int argc , char** argv){
	x = atoi(argv[1]);
	y = atoi(argv[2]);
	dx = WIDTH / (x - 1);
	dy = HEIGHT / (y - 1);
	int countNodes = x * y , currentProc;

	voltage = new double*[countNodes + 1];
	for(int i = 0 ; i < countNodes + 1 ; i++)
		voltage[i] = new double[1];
	plate = createPlate(x , y , voltage);
	argThread* args = new argThread();

	struct timeval begin;
	struct timeval end;
	struct timezone zone;

	gettimeofday(&begin, &zone);

	MPI_Init (&argc, &argv);
  	MPI_Comm_size (MPI_COMM_WORLD, &countThreads);
	MPI_Comm_rank (MPI_COMM_WORLD, &currentProc);

	MPI_Barrier(MPI_COMM_WORLD);

	int** intervals = createBlockOrder(&plate -> getOrder()[1] , y ,  x , countThreads);

	if(countThreads > 1){
		Plate* plates;
		if(currentProc == 0)
			plates = new Plate(&plate -> getNodes()[intervals[0][0]] ,  intervals[1][1] - intervals[0][0]);
		else if(currentProc == countThreads - 1)
			plates = new Plate(&plate -> getNodes()[intervals[2 * currentProc - 1][0]] ,  intervals[2 * currentProc][1] - intervals[2 * currentProc - 1][0]);
		else
			plates = new Plate(&plate -> getNodes()[intervals[2 * currentProc - 1][0]] , intervals[2 * currentProc + 1][1] - intervals[2 * currentProc - 1][0]);

		args -> init(currentProc , plates , intervals[2 * currentProc][1] - intervals[2 * currentProc][0]);
	}
	else
		args -> init(0 , plate , intervals[0][1] - intervals[0][0]);


	MPI_Barrier(MPI_COMM_WORLD);


	action((void*)(args));

	gettimeofday(&end, &zone);
	fprintf(stderr , "Time executing :: %lu on number of process %d\n" , end.tv_sec * 1000000 + end.tv_usec - begin.tv_usec - begin.tv_sec * 1000000 , countThreads);
		

	createScript(x , y , MaxToGnuPlot);

	MPI_Finalize();
	delete args;
	delete plate;
	delete voltage;
	delete test;
    return 0;
}