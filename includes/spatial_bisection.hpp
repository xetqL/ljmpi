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

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include "partitioner.hpp"
#include "spatial_elements.hpp"
#include "utils.hpp"

std::string to_string(const std::array<std::pair<double, double>, 2> &element) {
    std::string x = "("+std::to_string(element.at(0).first)+", "+std::to_string(element.at(0).second) + ")";
    std::string y = "("+std::to_string(element.at(1).first)+", "+std::to_string(element.at(1).second) + ")";
    std::string retval = "X: "+x+" Y:"+y;
    return retval;
}
namespace partitioning {
namespace geometric {

using PartitionID=int;

template<int N>
using Domain = std::array<std::pair<double, double>, N>;

template <int N>
inline Domain<N> borders_to_domain(const double xmin, const double ymin, const double zmin, const double xmax, const double ymax, const double zmax, const double simsize){
    Domain<N> res;
/*  p-Min
    -----> o------o
          /      /|
         o------o |
         |      | o
         |      |/   p-Max
         o------o <------- */
    res.at(0).first = xmin > 0 ? xmin : 0; res.at(0).second = xmax < simsize ? xmax : simsize;
    res.at(1).first = ymin > 0 ? ymin : 0; res.at(1).second = ymax < simsize ? ymax : simsize;
    if(N == 3) {
        res.at(2).first = zmin > 0 ? zmin : 0;
        res.at(2).second = zmax< simsize ? zmax : simsize;
    }
    return res;
}

inline double dist2(const std::pair<double, double> p1, const std::pair<double,double> p2){
    return std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2);
}

template<size_t N>
inline boost::geometry::model::box<boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>>
domain_to_box (const Domain<N> &domain){
    using Point3D = boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;
    using Box3D = boost::geometry::model::box<Point3D>;
    Point3D minA(domain.at(0).first,  domain.at(1).first,N > 2 ? domain.at(2).first : 0.0);
    Point3D maxA(domain.at(0).second, domain.at(1).second, N > 2 ? domain.at(2).second : 0.0);
    return Box3D(minA, maxA);
};

template<size_t N>
inline const bool are_domain_neighbors(const Domain<N> &A, const Domain<N> &B, const double min_d){
    auto boxA = domain_to_box(A);
    auto boxB = domain_to_box(B);
    return boost::geometry::distance(boxA, boxB) <= min_d;
}

template<size_t N>
const std::vector<std::pair<size_t, Domain<N> > > get_neighboring_domains(const size_t my_domain_idx, const std::vector<Domain<N>> &domain_list, const double min_d = 0){
    std::vector<std::pair<size_t, Domain<N>>> retval;
    const size_t nb_domain = domain_list.size();
    for(size_t domain_idx = 0; domain_idx < nb_domain; ++domain_idx){
        if(domain_idx != my_domain_idx) {
            if(are_domain_neighbors<N>(domain_list.at(my_domain_idx), domain_list.at(domain_idx), min_d))
                retval.push_back(std::make_pair(domain_idx, domain_list.at(domain_idx)));
        }
    }
    return retval;
}

template<int N>
struct PartitionInfo {
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
    virtual partitioning::CommunicationDatatype register_datatype() const {

        MPI_Datatype element_datatype,
                     vec_datatype,
                     range_datatype,
                     domain_datatype,
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

        blockcount_range[0] = N;
        oldtype_range[0] = MPI_DOUBLE;
        MPI_Type_struct(1, blockcount_range, offset, oldtype_range, &range_datatype);
        MPI_Type_commit(&range_datatype);

        MPI_Type_contiguous(N, range_datatype, &domain_datatype);
        MPI_Type_commit(&domain_datatype);

        return partitioning::CommunicationDatatype(vec_datatype, element_datatype, range_datatype, domain_datatype);
    }
};
}
}

#endif //NBMPI_SPATIALBISECTION_HPP
