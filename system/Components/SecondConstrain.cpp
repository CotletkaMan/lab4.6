#include "Component.h"

class SecondConstrain : public Component{
	private:
		double dt;
		double dn;
	public:
		SecondConstrain(double dn , double dt){
			this -> dn = dn;
			this -> dt = dt;
		}

		double* include(double param){
			matrix[0][0] = 0. ; matrix[0][1] = 0. ; matrix[0][2] = dt / dn;
			matrix[0][0] = 0. ; matrix[0][1] = 0. ; matrix[0][2] = -dt / dn;
			return &matrix[0][0];
		}
};