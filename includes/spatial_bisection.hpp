//
// Created by xetql on 12.12.17.
//

#ifndef NBMPI_SPATIALBISECTION_HPP
#define NBMPI_SPATIALBISECTION_HPP


#include <algorithm>
#include <memory>
#include <mpi.h>
#include <ostream>
#include <tuple>
#include <vector>


#include "partitioner.hpp"
#include "spatial_elements.hpp"
#include "utils.hpp"


namespace partitioning { namespace geometric {
    using PartitionID=int;

    template<int N>
    using Domain = std::array<std::pair<double, double>, N>;
        template<size_t N>
        bool are_domain_neighbors(const Domain<N> &A, const Domain<N> &B){
            bool share_dim = false;
            size_t shared_dim = -1;
            for(size_t dim = 0; dim < N; ++dim){
                if(A.at(dim).first == B.at(dim).first || A.at(dim).first == B.at(dim).second ||
                   A.at(dim).second == B.at(dim).first || A.at(dim).second == B.at(dim).second) {
                    shared_dim = dim;
                    share_dim  = true;
                    break;
                }
            }
            if(!share_dim) return false;
            int dim_cpt = 1;
            for(size_t dim = 0; dim < N; ++dim){
                if (dim == shared_dim) continue;
                if(B.at(dim).first <= A.at(dim).first && A.at(dim).first <= B.at(dim).second) {
                    dim_cpt++;
                    continue;
                }
                if(B.at(dim).first <= A.at(dim).second && A.at(dim).second <= B.at(dim).second) {
                    dim_cpt++;
                    continue;
                }
                if(A.at(dim).first <= B.at(dim).first && B.at(dim).first <= A.at(dim).second) {
                    dim_cpt++;
                    continue;
                }
                if(A.at(dim).first <= B.at(dim).second && B.at(dim).second <= A.at(dim).second) {
                    dim_cpt++;
                    continue;
                }
            }
            return dim_cpt == N;
        }
    template<int N>
    struct PartitionInfo{
        using Domain=std::array<std::pair<double, double>, N>; //One range per dimension
        const PartitionID part_id;
        const std::vector<elements::Element<N>> elements;
        const Domain domain;
    };

    template<int N>
    struct PartitionsInfo2 {
        const std::vector<PartitionInfo<N>> parts;
        PartitionsInfo2(const std::vector<PartitionInfo<N>> &partitions) : parts(partitions){}
    };

    template<int N>
    struct PartitionsInfo {
        using Domain=std::array<std::pair<double, double>, N>; //One range per dimension

        const std::vector<std::pair<PartitionID, elements::Element<N>>> parts;
        const std::vector<Domain> domains;

        PartitionsInfo(const std::vector<std::pair<PartitionID, elements::Element<N>>> &partitions, const std::vector<Domain> &domain) : parts(partitions), domains(domain){}
        PartitionsInfo(const std::vector<std::pair<PartitionID, elements::Element<N>>> &partitions) : parts(partitions){}
    };

    template<int N>
    struct BisectionInfo{
        using Domain=std::array<std::pair<double, double>, N>; //One range per dimension
        const std::vector<std::pair<elements::Element<N>, int>> left, right;
        const Domain left_domain, right_domain;
        BisectionInfo(const std::vector<std::pair<elements::Element<N>, int>> &left, const std::vector<std::pair<elements::Element<N>, int>> &right, const Domain &left_domain, const Domain &right_domain)
                : left(left), right(right), left_domain(left_domain), right_domain(right_domain) {}
    };

    /**
     * Bisect a dataset in a given dimension at the median point in the dataset
     * Time: Upper bound is O(n.lg(n)) to retrieve the median point
     * @param points
     * @param IDs
     * @param dim
     * @return The two new partitions and their points
     */
    template<int N>
    std::unique_ptr<BisectionInfo<N>> bisect(std::vector<elements::Element<N>> points, const std::vector<int>& IDs, int dim){
        std::vector<std::pair<elements::Element<N>, int>> left, right;

        auto zipped_points = partitioning::utils::zip(points, IDs); //O(n)
        //get the median by sorting in O(n.lg(n))
        //TODO: Median can be retrieved in O(n) c.f Get median of median etc.
        std::sort(zipped_points.begin(), zipped_points.end(),
                  [dim](const std::pair<elements::Element<N>, int> & a, const std::pair<elements::Element<N>, int> & b) -> bool{
                      return a.first.position.at(dim) < b.first.position.at(dim);
                  });
        unsigned int median_idx = points.size() / 2;
        //bind to partition in O(n)
        for(unsigned int i = 0; i < zipped_points.size(); ++i){
            if(i < median_idx) left.push_back(zipped_points.at(i));
            else right.push_back(zipped_points.at(i));
        }
        return std::make_unique<BisectionInfo<N>>(left, right);
    }

    /**
     * Bisect a dataset in a given dimension at the median point in the dataset
     * Time: Upper bound is O(n.lg(n)) to retrieve the median point
     * @param points
     * @param IDs
     * @param dim
     * @return The two new partitions and their points
     */
    template<int N>
    std::unique_ptr<BisectionInfo<N>> bisect(std::tuple<std::vector<elements::Element<N>>, const std::array<std::pair<double, double>, N>, const std::vector<int>>& region_info, const int dim){
        auto points = std::get<0>(region_info);
        auto domain = std::get<1>(region_info);
        auto IDs    = std::get<2>(region_info);

        std::vector<std::pair<elements::Element<N>, int>> left, right;

        auto zipped_points = partitioning::utils::zip(points, IDs); //O(n)
        //get the median by sorting in O(n.lg(n))
        //TODO: Median can be retrieved in O(n) c.f Get median of median etc.
        std::sort(zipped_points.begin(), zipped_points.end(),
                  [dim](const std::pair<elements::Element<N>, int> & a, const std::pair<elements::Element<N>, int> & b) -> bool{
                      return a.first.position.at(dim) < b.first.position.at(dim);
                  });
        unsigned int median_idx = points.size() / 2;

        //construct new domain boundaries
        double boundary_dim = zipped_points.at(median_idx).first.position.at(dim);

        auto left_domain  = domain;
        left_domain.at(dim).second = boundary_dim;
        auto right_domain = domain;
        right_domain.at(dim).first = boundary_dim;

        //bind to partition in O(n)
        for(unsigned int i = 0; i < zipped_points.size(); ++i){
            if(i < median_idx) left.push_back(zipped_points.at(i));
            else right.push_back(zipped_points.at(i));
        }
        return std::make_unique<BisectionInfo<N>>(left, right, left_domain, right_domain);
    }

    template<int N>
    class SeqSpatialBisection : public partitioning::Partitioner<PartitionsInfo<N>, std::vector<elements::Element<N>>, std::array<std::pair<double, double>, N>> {
    public:

        virtual std::unique_ptr<PartitionsInfo<N>> partition_data(
                std::vector<elements::Element<N>> spatial_data,
                std::array<std::pair<double, double>, N> &domain_boundary,
                int number_of_partitions) const {

            std::vector<std::tuple<std::vector<elements::Element<N>>, const std::array<std::pair<double, double>, N>, const std::vector<int>>> domain;
            std::vector<int> range(spatial_data.size());

            std::iota(range.begin(), range.end(), 0); //generate ids of particles from 0 to N-1
            domain.push_back(std::make_tuple(spatial_data, domain_boundary, range)); //the particles and their id.

            unsigned int logPE = std::log(number_of_partitions) / std::log(2);

            //O(lg(p)*2^lg(p-1)*n.lg(n)), P is the number of partitions
            for(unsigned int depth = 0; depth < logPE; ++depth) {
                unsigned int dim = depth % N;
                std::vector<std::unique_ptr<BisectionInfo<N>>> bisections;
                for(std::tuple<std::vector<elements::Element<N>>, const std::array<std::pair<double, double>, N>, const std::vector<int>>& section : domain) {
                    bisections.push_back(std::move(bisect<N>(section, dim)));
                }
                domain.clear();
                for(std::unique_ptr<BisectionInfo<N>> const& subsection: bisections){
                    auto lsubsection = partitioning::utils::unzip(subsection->left);
                    domain.push_back( std::make_tuple(lsubsection.first, subsection->left_domain, lsubsection.second) );
                    auto rsubsection = partitioning::utils::unzip(subsection->right);
                    domain.push_back( std::make_tuple(rsubsection.first, subsection->right_domain, rsubsection.second) );
                }
            }

            std::vector<std::pair<PartitionID , elements::Element<N>>> partitions(spatial_data.size());
            std::vector<std::array<std::pair<double, double>, N>> domains(domain.size());

            PartitionID pID = 0;
            //O(p*n) => O(n) p <<< n
            for(std::tuple<std::vector<elements::Element<N>>, const std::array<std::pair<double, double>, N>, const std::vector<int>>& partition: domain){
                auto zipped = partitioning::utils::zip(std::get<0>(partition), std::get<2>(partition));
                for(auto const& element : zipped){
                    partitions[element.second] = std::make_pair(pID, element.first);
                }
                domains[pID] = std::get<1>(partition);
                pID++;
            }

            return std::move(std::make_unique<PartitionsInfo<N>>(partitions, domains));
        }

        /**
         * Register a MPI data type for a 2D or 3D particle structure.
         */
        //TODO: The structure will integrate an id => add a new field in the data type.
        virtual partitioning::CommunicationDatatype register_datatype() const {

            MPI_Datatype element_datatype,
                         vec_datatype,
                         range_datatype,
                         oldtype_range[1],
                         oldtype_element[2];

            MPI_Aint offset[2], intex;

            int blockcount_element[2], blockcount_range[1];

            // register particle element type
            int array_size = N;
            MPI_Type_contiguous(array_size, MPI_DOUBLE, &vec_datatype);
            MPI_Type_commit(&vec_datatype);

            blockcount_element[0] = 1;
            blockcount_element[1] = 3; //position, velocity, acceleration

            oldtype_element[0] = MPI_INT;
            oldtype_element[1] = vec_datatype;

            MPI_Type_extent(MPI_INT, &intex);
            offset[0] = static_cast<MPI_Aint>(0);
            offset[1] = intex;

            MPI_Type_struct(2, blockcount_element, offset, oldtype_element, &element_datatype);
            MPI_Type_commit(&element_datatype);

            blockcount_range[0] = 2;
            oldtype_range[0] = MPI_DOUBLE;
            MPI_Type_struct(1, blockcount_range, offset, oldtype_range, &range_datatype);
            MPI_Type_commit(&range_datatype);

            return partitioning::CommunicationDatatype(vec_datatype, element_datatype, range_datatype);
        }
    };

}}

#endif //NBMPI_SPATIALBISECTION_HPP
