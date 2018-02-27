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

    template<int N>
    void gather_elements_on( const int nb_elements,
                                    const std::vector<elements::Element<N>> &local_el,
                                    const int dest_rank,
                                    std::vector<elements::Element<N>> &dest_el,
                                    const MPI_Datatype& sendtype,
                                    const MPI_Comm& comm){
        int nlocal = local_el.size();
        int world_size, my_rank;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &my_rank);

        std::vector<int> counts(world_size,0), displs(world_size, 0);
        MPI_Gather(&nlocal, 1, MPI_INT, &counts.front(), 1, MPI_INT, dest_rank, comm);
        for(int cpt = 0; cpt < world_size; ++cpt) displs[cpt] = cpt == 0 ? 0: displs[cpt-1]+counts[cpt-1];
        if(my_rank == dest_rank) if(dest_el.size() < nb_elements) dest_el.resize(nb_elements);
        MPI_Gatherv(&local_el.front(), nlocal, sendtype,
                    &dest_el.front(), &counts.front(), &displs.front(), sendtype, dest_rank, comm);
    }

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
        const MPI_Datatype get_element_datatype()  {
            return communication_datatypes.elements_datatype;
        }
        const MPI_Datatype get_range_datatype()  {
            return communication_datatypes.range_datatype;
        }
        const MPI_Datatype get_domain_datatype()  {
            return communication_datatypes.domain_datatype;
        }
        const MPI_Comm get_communicator()  {
            return LB_COMM;
        }

        //Assume that the data are all located on PE 0.
        LoadBalancer(const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner,
                     MPI_Comm comm) : partitioner(&partitioner), communication_datatypes(partitioner.register_datatype()), LB_COMM(comm){
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
            MPI_Barrier(this->LB_COMM);
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
                const size_t wsize = (size_t) this->world_size;
                std::vector<elements::Element<N>> buffer;
                std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
                auto neighbors = partitioning::utils::unzip(partitioning::geometric::get_neighboring_domains(this->caller_rank, domains, 0.0007)).first;

                for(const size_t &PE : neighbors){//size_t PE = 0; PE < wsize; ++PE) {
                    if (PE == (size_t) this->caller_rank) continue; //do not check with myself
                    //check within the remaining elements which belong to the current PE
                    size_t data_id = 0;
                    while (data_id < data.size()) {
                        if (elements::is_inside<N>(data.at(data_id), domains.at(PE))) {
                            //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                            //swap iterator values in constant time
                            std::iter_swap(data.begin() + data_id, data.end() - 1);
                            //get the value and push it in the "to migrate" vector
                            data_to_migrate.at(PE).push_back(*(data.end() - 1));
                            //pop the head of the list in constant time
                            data.pop_back();
                        } else data_id++; //if the element must stay with me then check the next one
                    }
                }
                std::vector<MPI_Request> reqs(neighbors.size());
                std::vector<MPI_Status> statuses(neighbors.size());
                int cpt = 0, nb_neighbors = neighbors.size();
                for(const size_t &PE : neighbors){
                    int send_size = data_to_migrate.at(PE).size();
                    MPI_Isend(&data_to_migrate.at(PE).front(), send_size, this->get_element_datatype(), PE, 300, this->LB_COMM, &reqs[cpt]);
                    cpt++;
                }
                cpt=0;
                while(cpt < nb_neighbors) {// receive the data in any order
                    int source_rank, size;
                    MPI_Probe(MPI_ANY_SOURCE, 300, this->LB_COMM, &statuses[cpt]);
                    source_rank = statuses[cpt].MPI_SOURCE;
                    MPI_Get_count(&statuses[cpt], this->get_element_datatype(), &size);
                    buffer.resize(size);
                    MPI_Recv(&buffer.front(), size, this->get_element_datatype(), source_rank, 300, this->LB_COMM, &statuses[cpt]);
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(data));
                    cpt++;
                }
                MPI_Waitall(cpt, &reqs.front(), &statuses.front());
            }

            const std::vector<elements::Element<N>> exchange_data(const std::vector<elements::Element<N>> &data,
                                                                  const std::vector<partitioning::geometric::Domain<N>> &domains) override
            {
                std::vector<elements::Element<N>> buffer;
                std::vector<elements::Element<N>> remote_data_gathered;
                // Get the neighbors
                const size_t wsize = (size_t) this->world_size;
                std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
                auto neighbors = partitioning::utils::unzip(partitioning::geometric::get_neighboring_domains(this->caller_rank, domains, 0.0007)).first;

                for(const size_t &PE : neighbors){//size_t PE = 0; PE < wsize; ++PE) {
                    if (PE == (size_t) this->caller_rank) continue; //do not check with myself
                    //check within the remaining elements which belong to the current PE
                    size_t data_id = 0;
                    while (data_id < data.size()) {
                        if (elements::distance2<N>(domains.at(PE), data.at(data_id)) < 0.0007) {
                            //get the value and push it in the "to migrate" vector
                            data_to_migrate.at(PE).push_back(data.at(data_id));
                        }
                        data_id++; //if the element must stay with me then check the next one
                    }
                }
                std::vector<MPI_Request> reqs(neighbors.size());
                std::vector<MPI_Status> statuses(neighbors.size());
                int cpt = 0, nb_neighbors = neighbors.size();
                for(const size_t &neighbor_idx : neighbors){   //give all my data to neighbors
                    int send_size = data_to_migrate.at(neighbor_idx).size();
                    MPI_Isend(&data_to_migrate.at(neighbor_idx).front(), send_size, this->get_element_datatype(), neighbor_idx, 200, this->LB_COMM, &reqs[cpt]);
                    cpt++;
                }
                cpt=0;
                while(cpt < nb_neighbors) {// receive the data in any order
                    int source_rank, size;
                    MPI_Probe(MPI_ANY_SOURCE, 200, this->LB_COMM, &statuses[cpt]);
                    source_rank = statuses[cpt].MPI_SOURCE;
                    MPI_Get_count(&statuses[cpt], this->get_element_datatype(), &size);
                    buffer.resize(size);
                    MPI_Recv(&buffer.front(), size, this->get_element_datatype(), source_rank, 200, this->LB_COMM, &statuses[cpt]);
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
                    cpt++;
                }
                MPI_Waitall(reqs.size(), &reqs.front(), &statuses.front()); //less strict than mpi_barrier

                return remote_data_gathered;
            }
        };
    }

}

#endif //NBMPI_LOADBALANCER_HPP
