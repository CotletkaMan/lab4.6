#include <vector>
#include <iostream>
#include <algorithm>
#include "Components/Component.h"

using namespace std;

class Node;

class AddableComponent : public Component{
	private:
		Component* component;
		Node* inNode;
		Node* outNode;
	public:
		AddableComponent(Component* component , Node* first , Node* second){
			this -> component = component;
			this -> inNode = first;
			this -> outNode = second;
			idComponent = component -> getId();
		}

		~AddableComponent(){
			delete component;
		}

		double* include(double value){
			return component -> include(value);
		}

		Node* getFirstNode(){
			return inNode;
		}

		Node* getSecondNode(){
			return outNode;
		}

		Component* getComponent(){
			return component;
		}
};

class Node{
	private:
		int* numNode;
		double** states;
		vector<AddableComponent*> components;
	public:
		Node(int* numNode , double** states){
			this -> numNode = numNode;
			this -> states = states;
		}

		int getNumNode(){
			return *numNode;
		}

		double* getState(){
			return states[*numNode];
		}

		void addComponent(AddableComponent* component){
			std:sort(components.begin() , components.end() , Component::comparate); 
			if(!std::binary_search(components.begin() , components.end() , component , Component::comparate))
				components.push_back(component);
		}

		vector<AddableComponent*> getComponents(){
			return components;
		}

		void createStiffMatrix(double *array , int length){
			for(vector<AddableComponent*>::iterator iterator = components.begin() ; iterator != components.end() ; iterator++){
				AddableComponent* component = *iterator;
				int out = component -> getFirstNode() -> getNumNode();
				int in = component -> getSecondNode() -> getNumNode();
				double* matrix = component -> include(*states[out] - *states[in]);

				if(out == *numNode){
					array[out - 1] += matrix[0];
					if(in) array[in - 1] += matrix[1];
					array[length - 1] += matrix[2];
				}
				else{
					if(out) array[out - 1] += matrix[3];
					array[in - 1] += matrix[4];
					array[length - 1] += matrix[5];
				}
			}
		}

		static bool comparate(Node* node1 , Node* node2){
			return node1 -> getNumNode() < node2 -> getNumNode(); 
		}
};

