//
// Created by xetql on 18.12.17.
//

#ifndef NBMPI_LOADBALANCER_HPP
#define NBMPI_LOADBALANCER_HPP

#include <map>
#include <mpi.h>

#include "partitioner.hpp"
#include "spatial_elements.hpp"
#include "spatial_bisection.hpp"

namespace load_balancing {
    using ProcessingElementID=int;

    template<typename PartitionType, typename ContainedDataType, typename DomainContainer>
    class LoadBalancer {
    protected:
        const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer>* partitioner;
        const std::map<ProcessingElementID, std::vector<ContainedDataType>> storage_table;
        int world_size;
        int caller_rank;
        partitioning::CommunicationDatatype communication_datatypes;

    public:

        const MPI_Datatype get_element_datatype() const {
            return communication_datatypes.elements_datatype;
        }

        const MPI_Datatype get_range_datatype() const {
            return communication_datatypes.range_datatype;
        }

        //Assume that the data are all located on PE 0.
        LoadBalancer(const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner2,
                     MPI_Comm comm) : partitioner(&partitioner2), communication_datatypes(partitioner2.register_datatype()){
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
        };

        //Provide a table that specifies where is the data located (on which PE)
        LoadBalancer(const std::map<ProcessingElementID, std::vector<ContainedDataType>> &storage_table,
                     const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner2,
                     MPI_Comm comm) : partitioner(&partitioner2), storage_table(storage_table), communication_datatypes(partitioner2.register_datatype()) {
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
        };

        void stop(){
            communication_datatypes.free_datatypes();
        }

        //TODO: It should somehow retrieves the data from the processing elements, at least makes it possible.
        //TODO: It should map the partitions as function of the PEs' pace
        virtual void load_balance(std::vector<ContainedDataType>& data, DomainContainer& domain_boundary){
            std::vector<int> counts(world_size);
            std::vector<int> displs(world_size);
            std::fill(counts.begin(), counts.end(), 0);

            //TODO: Have to detect if the algorithm is sequential or parallel
            if (caller_rank == 0){
                //partition the dataset in world_size partitions
                std::unique_ptr<PartitionType> partitions = partitioner->partition_data(data, domain_boundary, world_size);
                auto partitioned_data = partitions->parts;
                auto subdomains = partitions->domains;

                std::sort(partitioned_data.begin(), partitioned_data.end(), [](const auto & a, const auto & b) -> bool{
                    return a.first < b.first;
                });
                //counts the number to send to each processor
                displs[0] = 0;
                for(int i=0; i < world_size; ++i)
                    for(size_t cpt=displs[i]; cpt < partitioned_data.size(); ++cpt)
                        if(partitioned_data.at(cpt).first == i)
                            counts[i] += 1; //increment the number of particles attributed to the PE
                        else {
                            displs[i+1] = cpt; //set the shift in the buffer
                            break;
                        }
                //send elements
                for(int i = 0; i < world_size; ++i) {
                    MPI_Send(&data[displs[i]], counts[i], this->get_element_datatype(), i, 666, MPI_COMM_WORLD);
                    MPI_Send(&subdomains.front(), ContainedDataType::size() / 2, this->get_range_datatype(), i, 666, MPI_COMM_WORLD);
                }
            }

            // Check the number of elements sent to us
            MPI_Status status;
            MPI_Probe(0, 666, MPI_COMM_WORLD, &status);
            int my_data_size;
            // get the number of elements according to the status.
            MPI_Get_count(&status, this->get_element_datatype(), &my_data_size);
            // resize the data to fit the number of elements
            data.resize(my_data_size);
            // receive the elements
            MPI_Recv(&data.front(), my_data_size, this->get_element_datatype(), 0 , 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&domain_boundary.front(), ContainedDataType::size() / 2, this->get_range_datatype(), 0, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    };
    namespace geometric {
        template<int N>
        class GeometricLoadBalancer : public LoadBalancer<partitioning::geometric::PartitionsInfo<N>, elements::Element<N>, std::array<std::pair<double, double>, N>> {
        public:
            /**
             * Create a load balancer based on geometric partitioning algorithms
             * @param partitioner Partitioning algorithm
             * @param comm Communication group
             */
            GeometricLoadBalancer(
                    const partitioning::Partitioner<
                            partitioning::geometric::PartitionsInfo<N>,
                            std::vector<elements::Element<N>>,
                            std::array<std::pair<double, double>, N>> &partitioner, MPI_Comm comm) : LoadBalancer<partitioning::geometric::PartitionsInfo<N>, elements::Element<N>, std::array<std::pair<double, double>, N>>(partitioner, comm) {}
        };
    }

}

#endif //NBMPI_LOADBALANCER_HPP
