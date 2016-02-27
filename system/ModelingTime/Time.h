#ifndef MODELINGTIME_H
#define MODELINGTIME_H


class Time {
    private:
        bool use;
    protected:
        double time;
        double timeStep;
    public:
        Time(){
            timeStep = time = 0;
            use = false;
        }      

        double getCurrentTime(){
            return time;
        }

        double getCurrentStep(){
            if(!use){
                return changeTime();
                use = true;
            }
            return timeStep;
        }
        void nextStep(){
            time += timeStep;
            timeStep = changeTime();
        }
    protected:
        virtual double changeTime() = 0;
};

#endif // MODELINGTIME_H