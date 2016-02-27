#include "Component.h"

class Temperature : public Component{
	private:
		double c;
		double dn;
	public:
		Temperature(double c , double dn){
			this->c = c;
			this->dn = dn * dn;
		}

		double* include(double param){
			matrix[0][0] = c / dn ; matrix[0][1] = -c / dn; matrix[0][2] = 0.;
			matrix[1][0] = -c / dn ; matrix[1][1] = c / dn; matrix[0][2] = 0.;
			return &matrix[0][0];
		}

};