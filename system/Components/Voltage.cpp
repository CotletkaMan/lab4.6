#include "Component.h"

class Voltage : public Component{
	private:
		double (*Func)(double);
		double inResist;
	public:
		Voltage(double (*func)(double)) : Component(){
			Func = func;
			inResist = 0.001;
		}

		double* include(double value){
			matrix[0][0] = -1./inResist; matrix[0][1] = 1./inResist; matrix[0][2] = -Func(ModelingTime::getTime() + ModelingTime::getStep()) / inResist;
			matrix[1][0] = 1./inResist; matrix[1][1] = -1./inResist; matrix[1][2] = Func(ModelingTime::getTime() + ModelingTime::getStep()) / inResist;
			return &matrix[0][0];
	}
};