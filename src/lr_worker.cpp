#include "lr_worker.h"
#include "mpi.h"

namespace ParameterServer
{
    void lr_worker::Run()
    {
        MPI_Comm comm;
        MPI_Comm_split(MPI_COMM_WORLD, 0, m_mpi_rank, &comm);
        // Calcualte data partition by start and end offsets;
        int startOffset = (m_mpi_rank - m_server_size) * (m_record_num / (m_mpi_procs - m_server_size)) + 1;
        int endOffset = (m_mpi_rank == m_mpi_procs - 1) ? m_record_num : (m_mpi_rank - m_server_size + 1) * (m_record_num / (m_mpi_procs - m_server_size));
        // y is label;
        boost::numeric::ublas::vector<double> y(endOffset - startOffset + 1);
        // x is training set;
        boost::numeric::ublas::matrix<double> x(endOffset - startOffset + 1, m_feature_size + 1);
        for (size_t i = 0; i < x.size1(); i++)
        {
            for (size_t j = 0; j < x.size2(); j++)
            {
                x(i, j) = 0;
            }
        }
        SimpleDataLoader loader(endOffset - startOffset + 1, m_feature_size + 1);
        // Load current partition data;
        loader.load_file(m_file_path.c_str(), y, x, startOffset, endOffset - startOffset + 1);

        // Initialize weight;
        boost::numeric::ublas::vector<double> weight(x.size2());
        for (size_t i = 0; i < weight.size(); ++i)
        {
            weight(i) = 0;
        }

        double *loss = new double[1];
        double preLoss = 0.0;
        int iter = 0;
        while (iter < m_max_iteration)
        {
            loss[0] = 0;
            // Logistic regression learning;
            lr_without_regularization(m_sgd, m_learning_rate, m_epsilon, 1, x, y, weight, loss[0]);
            // Aggregate loss into Server node;
            MPI_Reduce(loss, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
            // Broadcast loss to all workers;
            MPI_Bcast(loss, 1, MPI_DOUBLE, 1, comm);
            cout<<"Worker previous loss is "<<preLoss<<"; current loss is "<<loss[0]<<"; abs is "<<abs(preLoss - loss[0])<<endl;
            // Termination condition by epsilon or loss difference;
            if (loss[0] < m_epsilon ||
                abs(preLoss - loss[0]) < m_epsilon)
            {
                break;
            }
            preLoss = loss[0];
            double *tWeight = new double[weight.size()];
            for (size_t i = 0; i < weight.size(); i++)
            {
                tWeight[i] = weight(i);
            }
            // Aggregate weight into Server node;
            MPI_Reduce(tWeight, NULL, m_feature_size + 1, MPI_DOUBLE, MPI_SUM, 0, comm);
            // Broadcast weight to all workers;
            MPI_Bcast(tWeight, m_feature_size + 1, MPI_DOUBLE, 0, comm);
            for (size_t i = 0; i < weight.size(); i++)
            {
                weight(i) = tWeight[i];
            }
            delete tWeight;
            iter++;
        }
        delete loss;
        MPI_Comm_free(&comm);
    }
}
