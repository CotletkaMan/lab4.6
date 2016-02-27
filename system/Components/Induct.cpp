#include "Component.h"
#include <iostream>

class Induct: public Component{
	private:
		double L;
		double I;
	public:
		Induct(double l) : Component(){
			L = l;
			I = 0;
		}

		double* include(double value){
			double dt = ModelingTime::getStep();
			I += dt/L * (value);
			matrix[0][0] = dt/L; matrix[0][1] = - dt/L; matrix[0][2] = -I;
			matrix[1][0] = - dt/L; matrix[1][1] = dt/L; matrix[1][2] = I;

			return &matrix[0][0];
		}
};