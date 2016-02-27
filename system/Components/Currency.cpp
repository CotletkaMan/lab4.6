#include "Component.h"

class Currency: public Component{
	private:
		double (*Func)(double);
	public:
		Currency(double (*func)(double)) : Component(){
			Func = func;
		}

		double* include(double value){
			matrix[0][0] = 0; matrix[0][1] = 0; matrix[0][2] = -Func(ModelingTime::getTime() + ModelingTime::getStep());
			matrix[1][0] = 0; matrix[1][1] = 0; matrix[1][2] = +Func(ModelingTime::getTime() + ModelingTime::getStep());
			return &matrix[0][0];
		}
};