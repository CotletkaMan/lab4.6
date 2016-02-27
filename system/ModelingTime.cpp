#include "ModelingTime/Time.h"



class ModelingTime{
    private:
        static Time* timer;
    public:
        static void start(Time* time){
            timer = time;
        }

        static double getTime(){
            return timer -> getCurrentTime();
        }

        static double getStep(){
            return timer -> getCurrentStep();
        }

        static void nextStep(){
            timer -> nextStep();
        }
};