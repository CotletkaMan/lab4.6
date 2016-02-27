#include <iostream>

class Test{
public:
	int k;
	Test(int& z){
		k = z;
	}
};

int main(){
	int x = 2;
	Test test(x);
	std::cout << test.k;
	x = 4;
	std::cout << test.k;
}