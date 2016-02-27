#include "Component.h"

class FirstConstrain : public Component{
	private:
		double t;
		double inC;
	public:
		FirstConstrain(double t){
			this-> t = t;
			inC = 0.0001;
		}
		double* include(double par){
			matrix[0][0] = -1./inC ; matrix[0][1] = 1./inC; matrix[0][2] = t/inC;
			matrix[1][0] = 1./inC; matrix[1][1] = -1./inC; matrix[1][2] = -t/inC;
			return &matrix[0][0];
		}
};