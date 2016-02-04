/*
 * Worker base class of Parameter server architecture
 * Hins Pan
 * 2016.2.2
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace ParameterServer
{
    using namespace std;
    class Worker
    {
        public:
            Worker(){};
            virtual ~Worker(){};
            virtual void Run() = 0;
    };
}
