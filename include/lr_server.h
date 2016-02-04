/*
 * Logistic regression server base class
 * Hins Pan
 * 2016.2.2
 */
#include "server.h"
#include "mpi.h"

namespace ParameterServer
{
    class lr_server : public Server
    {
         public:
             lr_server(int rank, int feature_size, int max_iteration, double epsilon) : 
                 m_mpi_rank(rank), 
                 m_feature_size(feature_size), 
                 m_max_iteration(max_iteration),
                 m_epsilon(epsilon){};
             virtual ~lr_server(){};
             void Run();
         private:
             int m_mpi_rank;
             int m_feature_size;
             int m_max_iteration;
             double m_epsilon;
    };
} 
