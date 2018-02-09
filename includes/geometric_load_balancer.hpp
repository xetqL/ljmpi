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
        MPI_Comm LB_COMM;
    public:
        const MPI_Datatype get_element_datatype() const {
            return communication_datatypes.elements_datatype;
        }
        const MPI_Datatype get_range_datatype() const {
            return communication_datatypes.range_datatype;
        }
        const MPI_Datatype get_domain_datatype() const {
            return communication_datatypes.domain_datatype;
        }
        const MPI_Comm get_communicator() const {
            return LB_COMM;
        }

        //Assume that the data are all located on PE 0.
        LoadBalancer(const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner2,
                     MPI_Comm comm) : partitioner(&partitioner2), communication_datatypes(partitioner2.register_datatype()), LB_COMM(comm){
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
        };

        //Provide a table that specifies where is the data located (on which PE)
        LoadBalancer(const std::map<ProcessingElementID, std::vector<ContainedDataType>> &storage_table,
                     const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner2,
                     MPI_Comm comm) : partitioner(&partitioner2), storage_table(storage_table), communication_datatypes(partitioner2.register_datatype()), LB_COMM(comm) {
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
        };

        void stop(){
            communication_datatypes.free_datatypes();
        }

        //TODO: It should somehow retrieves the data from the processing elements, at least makes it possible.
        //TODO: It should map the partitions as function of the PEs' pace
        virtual void load_balance(std::vector<ContainedDataType>& data, std::vector<DomainContainer>& domain_boundary){
            std::vector<int> counts(this->world_size);
            std::vector<int> displs(this->world_size);
            std::fill(counts.begin(), counts.end(), 0);
            std::vector<partitioning::geometric::Domain<ContainedDataType::number_of_dimensions>> subdomains;
            //TODO: Have to detect if the algorithm is sequential or parallel
            if (this->caller_rank == 0){
                //partition the dataset in world_size partitions
                std::unique_ptr<PartitionType> partitions = partitioner->partition_data(data, domain_boundary.at(0), world_size);
                auto partitioned_data = partitions->parts;
                subdomains = partitions->domains; //all the sub domains (geometric)

                //std::for_each(subdomains.begin(), subdomains.end(), [](auto const& el){std::cout << to_string(el) << std::endl;});
                std::sort(partitioned_data.begin(), partitioned_data.end(), [](const auto & a, const auto & b) -> bool{
                    return a.first < b.first; //sort by partition id
                });

                //counts the number to send to each processor... normally the same number of particles has to be sent
                displs[0] = 0;

                for(int i=0; i < this->world_size; ++i)
                    for(size_t cpt=displs[i]; cpt < partitioned_data.size(); ++cpt)
                        if(partitioned_data.at(cpt).first == i)
                            counts[i] += 1; //increment the number of particles attributed to the PE
                        else {
                            displs[i+1] = cpt; //set the shift in the buffer
                            break;
                        }
                //send elements
                for(size_t i = 0; i < partitioned_data.size(); ++i) data[i] = partitioned_data.at(i).second;
                for(int i = 0; i < this->world_size; ++i) {
                    //send the particle attributed to the PE
                    MPI_Send(&data[displs[i]], counts[i], this->get_element_datatype(), i, 666, LB_COMM);
                }
                //broadcast the geometric partition of the initial domain
                MPI_Bcast(&subdomains.front(), subdomains.size(), this->get_domain_datatype(), 0, LB_COMM);
            }
            MPI_Status status;
            //Probe the data sent
            MPI_Probe(0, 666, LB_COMM, &status);
            int my_data_size;
            // Get the number of elements according to the status.
            MPI_Get_count(&status, this->get_element_datatype(), &my_data_size);
            // resize the data to fit the number of elements
            data.resize(my_data_size);
            // receive the elements
            MPI_Recv(&data.front(), my_data_size, this->get_element_datatype(), 0 , 666, LB_COMM, MPI_STATUS_IGNORE);
            //broadcast the geometric partition of the initial domain
            domain_boundary.clear();
            if(this->caller_rank!=0){
                domain_boundary.resize(this->world_size);
                MPI_Bcast(&domain_boundary.front(), this->world_size, this->get_domain_datatype(), 0, LB_COMM);
            }else{
                std::move(subdomains.begin(), subdomains.end(), std::back_inserter(domain_boundary));
            }
        }

        /**
         * Acquire data from neighboring PEs and store the data in the buffer.
         * @param data
         * @param domains
         */
        virtual const std::vector<ContainedDataType> exchange_data(const std::vector<ContainedDataType>& data, const std::vector<DomainContainer>& domains)= 0;
        virtual void migrate_particles(std::vector<ContainedDataType>& data, const std::vector<DomainContainer>& domains)  = 0;
    };
    namespace geometric {
        template<int N>
        class GeometricLoadBalancer : public LoadBalancer<partitioning::geometric::PartitionsInfo<N>, elements::Element<N>, partitioning::geometric::Domain<N>> {
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
                            std::array<std::pair<double, double>, N>> &partitioner,
                            MPI_Comm comm) :
                    LoadBalancer<
                            partitioning::geometric::PartitionsInfo<N>,
                            elements::Element<N>,
                            std::array<std::pair<double, double>, N>>(partitioner, comm) {}

            void migrate_particles(std::vector<elements::Element<N>> &data,
                                   const std::vector<partitioning::geometric::Domain<N>> &domains) override {
                std::vector<elements::Element<N>> buffer;
                std::vector<elements::Element<N>> data_to_migrate;
                size_t wsize = (size_t) this->world_size;
                for(size_t PE = 0; PE < wsize; ++PE){
                    if (PE == (size_t) this->caller_rank) continue; //do not check with myself
                    //check within the remaining elements which belong to the current PE
                    size_t data_id = 0;
                    while(data_id < data.size()){
                        if(elements::is_inside<N>(data.at(data_id), domains.at(PE))) {
                            //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                            //swap iterator values in constant time
                            std::iter_swap(data.begin() + data_id, data.end()-1);
                            //get the value and push it in the "to migrate" vector
                            data_to_migrate.push_back(*(data.end()-1));
                            //pop the head of the list in constant time
                            data.pop_back();
                        } else data_id++; //if the element must stay with me then check the next one
                    }
                    int send_size = data_to_migrate.size(), recv_size;
                    //exchange the size between two processes
                    MPI_Sendrecv(&send_size, 1, MPI_INT, PE, 666, &recv_size, 1, MPI_INT, PE, 666, this->LB_COMM, MPI_STATUS_IGNORE);
                    if (send_size != 0 || recv_size != 0){ //something has to be exchanged
                        buffer.resize(recv_size);
                        MPI_Sendrecv(&data_to_migrate.front(), send_size, this->get_element_datatype(), PE, 666,
                                     &buffer.front(), recv_size, this->get_element_datatype(), PE, 666, this->LB_COMM, MPI_STATUS_IGNORE );
                        std::move(buffer.begin(), buffer.end(), std::back_inserter(data)); //not optimal because data are checked twice
                        data_to_migrate.clear();
                    }
                }
                MPI_Barrier(this->LB_COMM);
            }

            const std::vector<elements::Element<N>> exchange_data(const std::vector<elements::Element<N>> &data,
                                                                  const std::vector<partitioning::geometric::Domain<N>> &domains) override
            {
                // Get the neighbors
                auto neighbors = partitioning::utils::unzip(partitioning::geometric::get_neighboring_domains(this->caller_rank, domains, 0.0007)).first;
                std::sort(neighbors.begin(), neighbors.end());
                //std::cout << to_string(domains.at(caller_rank)) << " " << neighbors.size() << std::endl;
                //std::cout << to_string(domains.at(caller_rank)) << " "<< neighbors.size() << std::endl;
                std::vector<elements::Element<N>> buffer;
                std::vector<elements::Element<N>> remote_data_gathered;
                for(const size_t neighbor_idx : neighbors){
                    if(neighbor_idx == (size_t) this->caller_rank) continue;
                    int send_size = data.size(), recv_size;
                    //exchange the size between two processes
                    MPI_Sendrecv(&send_size, 1, MPI_INT, neighbor_idx, 666, &recv_size, 1, MPI_INT, neighbor_idx, 666, this->LB_COMM, MPI_STATUS_IGNORE);
                    buffer.resize(recv_size);
                    MPI_Sendrecv(&data.front(), send_size, this->get_element_datatype(), neighbor_idx, 666,
                                 &buffer.front(), recv_size, this->get_element_datatype(), neighbor_idx, 666, this->LB_COMM, MPI_STATUS_IGNORE );
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
                }
                MPI_Barrier(this->LB_COMM);
                return remote_data_gathered;
            }


        };
    }

}

#endif //NBMPI_LOADBALANCER_HPP
