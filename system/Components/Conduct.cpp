#include "Component.h"

class Conduct: public Component{
	private:
		double C;
	public:
		Conduct(double c) : Component(){
			C = c;
		}

		double* include(double value){
			double dt = ModelingTime::getStep();
			matrix[0][0] = C/dt; matrix[0][1] = - C/dt; matrix[0][2] = C/dt * value;
			matrix[1][0] = - C/dt; matrix[1][1] = C/dt; matrix[1][2] = -C/dt * value;
			return &matrix[0][0];
		}
};