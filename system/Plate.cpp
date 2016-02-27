#include "Node.cpp"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;
class Plate{
	private:
		int countNodes;
		double** voltage;
		Node** nodes;
		int* inOrder;
		vector<Component*> components;

	public:
		Plate(int countNodes , double** voltage){
			this -> voltage = voltage;
			this -> countNodes = countNodes;
			nodes = new Node*[countNodes];
			inOrder = new int[countNodes];
			for(int i = 0 ; i < countNodes ; i++){
				inOrder[i] = i;
				nodes[i] = new Node(&inOrder[i] , voltage);
			}
		}

		Plate(Node** p_nodes , int p_length){
			countNodes = p_length + 1;
			voltage = new double*[countNodes];
			nodes = new Node*[countNodes];
			inOrder = new int[countNodes];
			vector<Node*> sorted;

			voltage[0] = new double[1];
			for(int i = 0 ; i < p_length ; i++){
				sorted.push_back(p_nodes[i]);
				inOrder[i] = i;
				nodes[i] = new Node(&inOrder[i] , voltage);
			}
			sorted.push_back(nodes[0]);
			nodes[p_length] = new Node(&inOrder[p_length] , voltage);
			inOrder[p_length] = p_length;


			sort(sorted.begin() , sorted.end() , Node::comparate);
			int i = 1;

			for(vector<Node*>::iterator it = sorted.begin() + 1 ; it != sorted.end() ; it++ , i++){	
				voltage[i] = (*it) -> getState();
				vector<AddableComponent*> addComponents = (*it) -> getComponents();
				for(vector<AddableComponent*>::iterator ad = addComponents.begin() ; ad != addComponents.end() ; ad++){
					int out = ((*ad) -> getFirstNode() -> getNumNode() == 0) ? 0 : find(sorted.begin() , sorted.end() , (*ad) -> getFirstNode()) - sorted.begin();
					int in = ((*ad) -> getSecondNode() -> getNumNode() == 0) ? 0 : find(sorted.begin() , sorted.end() , (*ad) -> getSecondNode()) - sorted.begin();
					if(in <= p_length && out <= p_length){
						AddableComponent* component = new AddableComponent((*ad) -> getComponent() , nodes[out] , nodes[in]);
						nodes[in] -> addComponent(component);
						nodes[out] -> addComponent(component);
					}
				}
			}
		}

		~Plate(){
			for(int i = 0 ; i < countNodes ; i++)
				delete nodes[i];
			delete nodes;
			delete inOrder;
			for(int i = 0 ; i < components.size() ; i++)
				delete components[i];
		}

		bool validateNode(int Node){
			return Node >= 0 && Node < countNodes;
		}

		void addComponent(int in , int out , Component* component){
			if(validateNode(in) && validateNode(out)){
				AddableComponent* addComponent = new AddableComponent(component , nodes[in] , nodes[out]);
				nodes[in] -> addComponent(addComponent);
				nodes[out] -> addComponent(addComponent);
				components.push_back(addComponent);
			}
		}

		double* createStiffMatrix(double* matrix){
			for(int i = countNodes - 1 ; i > 0 ; i--){
				nodes[i] -> createStiffMatrix(&matrix[(inOrder[i] - 1) * (countNodes)] , countNodes);
			}
			return &matrix[1];
		}

		int getCountNodes(){
			return countNodes;
		}

		Node** getNodes(){
			return nodes;
		}

		int* getOrder(){
			return inOrder;
		}

		void setOrder(int* order){
			inOrder = order;
		}

		double** getVoltage(){
			return voltage;
		}
};
