/*
 * Logistic regression worker base class
 * Hins Pan
 * 2016.2.2
 */
#include <string>

#include "worker.h"
#include "mpi.h"
#include "data_loader.h"

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ParameterServer
{
    class lr_worker : public Worker
    {
         public:
             lr_worker(string file_path, int rank, int procs, int server_size, int record_num, int feature_size, double epsilon, double learning_rate, int max_iteration, bool sgd) : 
                 m_file_path(file_path),
                 m_mpi_rank(rank),
                 m_mpi_procs(procs),
                 m_server_size(server_size),
                 m_record_num(record_num),
                 m_feature_size(feature_size),
                 m_epsilon(epsilon),
                 m_learning_rate(learning_rate),
                 m_max_iteration(max_iteration),
                 m_sgd(sgd){};
             virtual ~lr_worker(){};
             void Run();
         private:
             string m_file_path;
             int m_mpi_rank;
             int m_mpi_procs;
             int m_server_size;
             int m_record_num;
             int m_feature_size;
             double m_epsilon;
             double m_learning_rate;
             int m_max_iteration;
             bool m_sgd;
    };
}
