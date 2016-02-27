#ifndef COMPONENT_H_
#define COMPONENT_H_
#include "../ModelingTime.cpp"

class Component{
	private:
		static int countComponents;
	protected:
		double matrix[2][3];
		int idComponent;
	public:
		Component(){
			idComponent = countComponents;
			countComponents++;
		}
		virtual double* include(double value) = 0;

		int getId(){
			return idComponent;
		}

		static bool comparate(Component* comp1 , Component* comp2){
			return comp1 -> getId() < comp2 -> getId();
		}
};

int Component :: countComponents = 0; 

#endif