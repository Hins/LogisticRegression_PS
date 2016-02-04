/*
 * Server base class of parameter server architecture
 * Hins Pan
 * 2016.2.2
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace ParameterServer
{
    using namespace std;

    class Server
    {
        public:
            Server(){};
            virtual ~Server(){};
            virtual void Run() = 0;
    };
}
