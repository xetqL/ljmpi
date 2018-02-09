//
// Created by xetql on 13.12.17.
//
#include <cppcheck/cppcheck.hpp>
#include <random>
#include <map>
#include <unordered_map>

#include "../includes/spatial_bisection.hpp"
using namespace partitioning::geometric;
int main(int argc, char** argv){
    TestsRunner runner("Space Partitioning: Tests");

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::array<int , 4> result_should_be;
    unsigned int p = 256;
    for(int i = 0; i < 4; ++i) result_should_be.at(i) = p / 4;

    auto test_partition_balanced = std::make_shared<UnitTest<std::array<int, 4>>>
            ("Partitions contains the same number of elements", [&dist, &gen, &elements=p] {
                constexpr int DIM=2;
                std::vector<elements::Element<DIM>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    points.push_back(elements::Element<2>::create_random(dist, gen, i));
                }

                //apply bisection
                SeqSpatialBisection<DIM> partitioner;
                std::array<std::pair<double ,double>, DIM> domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0)
                };
                auto partitions = partitioner.partition_data(points, domain_boundary, 4);

                std::array<int, 4> result;
                std::fill(result.begin(), result.end(), 0);
                for(auto const& element : partitions->parts)
                    result[element.first] += 1;
                return result;
            }, result_should_be);

    auto  total_area_is_area_of_initial_domain = std::make_shared<UnitTest<std::vector<partitioning::geometric::Domain<2> > > >
            ("Total sub area equals initial total area", [&dist, &gen, &elements=p] {
                constexpr int DIM=2;
                std::vector<elements::Element<DIM>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    points.push_back(elements::Element<2>::create_random(dist, gen, i));
                }
                //apply bisection
                SeqSpatialBisection<DIM> partitioner;
                std::array<std::pair<double ,double>, DIM> domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0)
                };
                auto partitions = partitioner.partition_data(points, domain_boundary, 16);
                std::vector<partitioning::geometric::Domain<2> > d = partitions->domains;
                return d;
            }, [](auto const& subdomains) {
                double total_area = 1.0; //sqr domain
                double area_sum = 0.0;
                for(auto const& subdomain : subdomains){
                    const double width = subdomain.at(0).second - subdomain.at(0).first;
                    const double height= subdomain.at(1).second - subdomain.at(1).first;
                    area_sum += (width * height);
                }
                return std::fabs(total_area - area_sum) <= std::numeric_limits<double>::epsilon();
            });

    auto can_convert_raw_buffer_to_elements = std::make_shared<UnitTest<std::vector<elements::Element<2>> > >
            ("Raw data can be converted to convenient type", [] {
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                std::vector<elements::Element<2>> points(500);
                //populate points
                std::array<double, 1000> pos, vel;
                std::generate(pos.begin(), pos.end(), [=] () mutable {return (dist(gen)+1)*100.0;});
                std::generate(vel.begin(), vel.end(), [=] () mutable {return (dist(gen)+1)*100.0;});

                points = elements::transform<2>(500, &pos.front(), &vel.front());
                return points;

            }, [] (std::vector<elements::Element<2>> elements){
                return std::all_of(elements.begin(), elements.end(), [](auto el) {
                    return std::all_of(el.position.begin(), el.position.end(), [](auto p){
                        return p >= 1.0;}) && std::all_of(el.velocity.begin(), el.velocity.end(), [](auto v){
                        return v >= 1.0;
                    });
                });
            });

    auto can_convert_raw_buffer_to_elements_2 = std::make_shared<UnitTest<std::vector<elements::Element<2>> > >
            ("Raw data can be converted to convenient type in place", [] {
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(0.0, 1.0);

                std::vector<elements::Element<2>> points(500);

                //populate points
                std::array<double, 1000> pos, vel;

                std::generate(pos.begin(), pos.end(), [=] () mutable {return (dist(gen)+1)*100.0;});
                std::generate(vel.begin(), vel.end(), [=] () mutable {return (dist(gen)+1)*100.0;});
                elements::transform<2>(points, &pos.front(), &vel.front());

                return points;

            }, [] (std::vector<elements::Element<2>> elements){
                return std::all_of(elements.begin(), elements.end(), [](auto el) {
                    return std::all_of(el.position.begin(), el.position.end(), [](auto p){
                        return p >= 1.0;}) && std::all_of(el.velocity.begin(), el.velocity.end(), [](auto v){
                        return v >= 1.0;
                    });
                });
            });

    auto create_random_elements = std::make_shared<UnitTest<std::vector<elements::Element<2>> > >
            ("Element are created randomly following probability distribution", [] {
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(1.1, 2.1);

                std::vector<elements::Element<2>> points(500);

                //std::generate(points.begin(), points.end(), [=] () mutable{return elements::Element<2>::create_random(dist, gen);});
                elements::Element<2>::create_random_n(points, dist, gen);
                return points;

            }, [] (std::vector<elements::Element<2>> elements){
                return std::all_of(elements.begin(), elements.end(), [](auto el) {
                    return std::all_of(el.position.begin(), el.position.end(), [](auto p){
                        return p >= 1.0;}) && std::all_of(el.velocity.begin(), el.velocity.end(), [](auto v){
                        return v >= 1.0;
                    });
                });
            });

    auto create_random_elements_generic_vec = std::make_shared<UnitTest<std::vector<elements::Element<2>> > >
            ("N Elements (vector) are created randomly following probability distribution", [] {
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(1.1, 2.1);

                std::vector<elements::Element<2>> points(500);

                elements::Element<2>::create_random_n(points, dist, gen);

                return points;

            }, [] (std::vector<elements::Element<2>> elements){
                return std::all_of(elements.begin(), elements.end(), [](auto el) {
                    return std::all_of(el.position.begin(), el.position.end(), [](auto p){
                        return p >= 1.0;}) && std::all_of(el.velocity.begin(), el.velocity.end(), [](auto v){
                        return v >= 1.0;
                    });
                });
            });

    auto create_random_elements_generic_arr = std::make_shared<UnitTest<std::array<elements::Element<2>, 500> > >
            ("N Elements (array) are created randomly following probability distribution", [] {
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(1.1, 2.1);
                std::array<elements::Element<2>, 500> points;
                elements::Element<2>::create_random_n(points, dist, gen);
                return points;
            }, [] (std::array<elements::Element<2>, 500> elements){
                return std::all_of(elements.begin(), elements.end(), [](auto el) {
                    return std::all_of(el.position.begin(), el.position.end(), [](auto p){ return p >= 1.0;}) &&
                           std::all_of(el.velocity.begin(), el.velocity.end(), [](auto v){ return v >= 1.0;
                           });
                });
            });

    auto create_random_elements_generic_arr_with_predicate = std::make_shared<UnitTest<std::array<elements::Element<2>, 100> > >
            ("N Elements (array) are created randomly following probability distribution given a predicate", [] {
                std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                std::array<elements::Element<2>, 100> points;
                elements::Element<2>::create_random_n(points, dist, gen, [](auto point, auto other){
                    return elements::distance2<2>(point, other) >= 0.001;
                });
                return points;

            }, [] (std::array<elements::Element<2>, 100> elements){
                for(size_t i = 0; i < elements.size(); ++i){
                    for (size_t j = i+1; j < elements.size(); ++j) {
                        if(elements::distance2<2>(elements[i].position, elements[j].position) < 0.001) {
                            return false;
                        }
                    }
                }
                return true;
            });

    auto no_overlapping_region = std::make_shared<UnitTest<std::vector<partitioning::geometric::Domain<2> > > >
            ("Regions does not overlap", [&dist, &gen, &elements=p] {
                constexpr int DIM=2;
                std::vector<elements::Element<DIM>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    points.push_back(elements::Element<2>::create_random(dist, gen, i));
                }
                //apply bisection
                SeqSpatialBisection<DIM> partitioner;
                std::array<std::pair<double ,double>, DIM> domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0)
                };
                auto partitions = partitioner.partition_data(points, domain_boundary, 128);

                std::vector<partitioning::geometric::Domain<2> > d = partitions->domains;
                return d;
            }, [](auto const& subdomains) {
                for(size_t i = 0; i < subdomains.size(); ++i){
                    auto const subdomain_a = subdomains.at(i);
                    for(size_t j = i; j < subdomains.size(); ++j){
                        auto const subdomain_b = subdomains.at(j);
                        //Left or right boundary is within boundaries of another region
                        bool regionA_overlaps_regionB =
                                ((subdomain_a.at(0).first < subdomain_b.at(0).first && subdomain_b.at(0).first < subdomain_a.at(0).second) ||
                                 (subdomain_a.at(0).first < subdomain_b.at(0).second && subdomain_b.at(0).second < subdomain_a.at(0).second)) &&
                                ((subdomain_a.at(1).first < subdomain_b.at(1).first && subdomain_b.at(1).first < subdomain_a.at(1).second) ||
                                 (subdomain_a.at(1).first < subdomain_b.at(1).second && subdomain_b.at(1).second < subdomain_a.at(1).second));
                        bool regionB_overlaps_regionA =
                                ((subdomain_b.at(0).first < subdomain_a.at(0).first && subdomain_a.at(0).first < subdomain_b.at(0).second) ||
                                 (subdomain_b.at(0).first < subdomain_a.at(0).second && subdomain_a.at(0).second < subdomain_b.at(0).second)) &&
                                ((subdomain_b.at(1).first < subdomain_a.at(1).first && subdomain_a.at(1).first < subdomain_b.at(1).second) ||
                                 (subdomain_b.at(1).first < subdomain_a.at(1).second && subdomain_a.at(1).second < subdomain_b.at(1).second));
                        if (regionA_overlaps_regionB || regionB_overlaps_regionA){ return false; }
                    }
                }
                return true;
            });

    auto data_belongs_to_regions = std::make_shared<UnitTest<std::shared_ptr<PartitionsInfo<2>> > >
            ("Partition I is associated to domain at index I", [&dist, &gen, &elements=p] {
                constexpr int DIM=2;
                std::vector<elements::Element<DIM>> points;
                //populate points
                for (unsigned int i = 0; i < elements; ++i) {
                    points.push_back(elements::Element<2>::create_random(dist, gen, i));
                }
                //apply bisection
                SeqSpatialBisection<DIM> partitioner;
                std::array<std::pair<double ,double>, DIM> domain_boundary = {
                        std::make_pair(0.0, 1.0),
                        std::make_pair(0.0, 1.0)
                };
                auto partitions = partitioner.partition_data(points, domain_boundary, 16);
                std::shared_ptr<PartitionsInfo<2>> shared_partitions(partitions.release());
                return shared_partitions;
            }, [](auto const& partitions) {
                auto p = partitions->parts;
                auto d = partitions->domains;
                for(auto const& el : p){
                    if(!elements::is_inside<2>(el.second, d.at(el.first))) return false;
                }
                return true;
            });

    auto are_domain_neighbors = std::make_shared<UnitTest< bool > >
            ("Domain neighboring detection", [] {
                SeqSpatialBisection<2> partitioner;

                std::array<std::pair<double ,double>, 2> domain_boundary = {
                        std::make_pair(0.0, std::sqrt(8)),
                        std::make_pair(0.0, std::sqrt(8))
                };

                //std::random_device rd; //Will be used to obtain a seed for the random number engine
                std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dist(0.0, std::sqrt(8));
                std::vector<elements::Element<2>> points;
                for (unsigned int i = 0; i < 5000; ++i) {
                    points.push_back(elements::Element<2>::create_random(dist, gen, i));
                }
                auto part = partitioner.partition_data(points, domain_boundary, 64);
                std::vector<Domain<2>> domains = part->domains;
                //std::unordered_map<Domain<2>, std::shared_ptr<std::unordered_map<Domain<2>, double>>> distance_matrix;

                for (auto &d1 : domains){

                    for(auto &d2 : domains){
                        std::cout << to_string(d1) <<" & "<<to_string(d2)<<" -> " << std::sqrt(partitioning::geometric::domain_distance(d1,d2)) << std::endl;
                    }

                }
                return true;

            }, [](auto const& neighbors) {
                return true;
            });
/*
    auto distance2_line_to_el = std::make_shared<UnitTest< double > >
            ("Dist line to element", [] {
                std::vector<Domain<2>> domains = {
                        {std::make_pair(0.2, 0.5), std::make_pair(0.2, 0.5)}, // 0
                };
                std::array<double, 2> p = {0.3, 0.8}, v= {0.0, 0.0};
                elements::Element<2> e = elements::Element<2>::create(p, v, 0);
                return elements::distance2<2>(domains.at(0), e);
            }, [](auto const& d2) {
                std::cout << d2 << std::endl;
                return d2 == std::pow(0.8 - 0.5, 2);
            });
*/
    runner.add_test(test_partition_balanced);
    runner.add_test(total_area_is_area_of_initial_domain);
    runner.add_test(no_overlapping_region);
    runner.add_test(can_convert_raw_buffer_to_elements);
    runner.add_test(can_convert_raw_buffer_to_elements_2);
    runner.add_test(create_random_elements);
    runner.add_test(create_random_elements_generic_vec);
    runner.add_test(create_random_elements_generic_arr);
    runner.add_test(create_random_elements_generic_arr_with_predicate);
    runner.add_test(data_belongs_to_regions);
    runner.add_test(are_domain_neighbors);
    //runner.add_test(distance2_line_to_el);

    runner.run();
    runner.summarize();

    return runner.are_tests_passed();
}