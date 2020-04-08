//
// Created by xetql on 3/30/20.
//

#ifndef NBMPI_SHORTEST_PATH_HPP
#define NBMPI_SHORTEST_PATH_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <zoltan.h>
#include <cstdlib>

#include "../decision_makers/strategy.hpp"

#include "../ljpotential.hpp"
#include "../report.hpp"
#include "../physics.hpp"
#include "../nbody_io.hpp"
#include "../utils.hpp"

#include "../params.hpp"
#include "../spatial_elements.hpp"
#include "../zoltan_fn.hpp"
#include "../astar.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

using LBSolutionPath = std::vector<std::shared_ptr<Node> >;
using LBLiHist       = std::vector<Time>;
using LBDecHist      = std::vector<int>;
using NodeQueue      = std::multiset<std::shared_ptr<Node>, Compare>;

template<int N>
std::tuple<LBSolutionPath, LBLiHist, LBDecHist, TimeHistory> simulate_using_shortest_path(MESH_DATA<N> *mesh_data,
              Zoltan_Struct *load_balancer,
              sim_param_t *params,
              MPI_Comm comm = MPI_COMM_WORLD) {
    constexpr bool automatic_migration = true;
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    auto nb_solution_wanted = 1;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    const int nb_iterations = nframes*npframe;

    auto datatype = elements::register_datatype<N>();

    std::vector<elements::Element<N>> recv_buf(params->npart);

    std::vector<Time> times(nproc), my_frame_times(nframes);
    std::vector<Index> lscl(mesh_data->els.size()), head;
    std::vector<Complexity> my_frame_cmplx(nframes);

    const int nb_data = mesh_data->els.size();
    for(int i = 0; i < nb_data; ++i) mesh_data->els[i].lid = i;

    using TNode = Node;
    std::vector<std::shared_ptr<TNode>> container;
    container.reserve((unsigned long) std::pow(2, 20));
    using PriorityQueue = std::multiset<std::shared_ptr<TNode>, Compare>;
    PriorityQueue pQueue;
    pQueue.insert(std::make_shared<Node>(load_balancer, -npframe, npframe, DoLB));

    std::vector<std::shared_ptr<Node>> solutions;
    std::vector<bool> foundYes(nframes+1, false);
    std::vector<MESH_DATA<N>> rollback_data(nframes+1);
    std::for_each(rollback_data.begin(), rollback_data.end(), [mesh_data](auto& vec){vec.els.reserve(mesh_data->els.size());});

    rollback_data[0] = *mesh_data;

    do {
        std::shared_ptr<Node> currentNode = *pQueue.begin();
        pQueue.erase(pQueue.begin());
        if(!rank ) std::cout << currentNode << std::endl;
        //Ok, I found a Yes Node for a given depth of the binary tree, no other Yes node at this depth can be better
        if(currentNode->decision == DoLB && currentNode->start_it > 0) {
            prune_similar_nodes(currentNode, pQueue);
            foundYes.at(currentNode->start_it / npframe) = true;
        }

        if(currentNode->end_it >= nb_iterations) {
            solutions.push_back(currentNode);
            break;
        } else {
            auto children = currentNode->get_children();
            for(std::shared_ptr<Node> node : children) {
                const auto frame      = currentNode->end_it / npframe;
                const auto next_frame = frame + 1;
                if(node && ((node->decision == DontLB) || (node->decision == DoLB && !foundYes.at(frame)))) {
                    /* compute node cost */
                    Time comp_time = 0.0;

                    Time starting_time = currentNode->cost();

                    auto mesh_data = rollback_data.at(frame);
                    auto load_balancer = node->lb;

                    auto& cum_li_hist = node->li_slowdown_hist;
                    auto& time_hist   = node->time_hist;
                    auto& dec_hist    = node->dec_hist;
                    auto& it_stats    = node->stats;

                    // Move data according to my parent's state
                    Zoltan_Migrate_Particles<N>(mesh_data.els, load_balancer, datatype, comm);
                    // Compute my bounding box as function of my local data
                    auto bbox      = get_bounding_box<N>(params->rc, mesh_data.els);
                    // Compute which cells are on my borders
                    auto borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc);
                    // Get the ghost data from neighboring processors
                    auto remote_el = get_ghost_data<N>(load_balancer, mesh_data.els, &head, &lscl, bbox, borders, params->rc, datatype, comm);



                    for (int i = 0; i < node->batch_size; ++i) {
                        START_TIMER(it_compute_time);
                        lj::compute_one_step<N>(mesh_data.els, remote_el, &head, &lscl, bbox, borders, params);
                        END_TIMER(it_compute_time);

                        // Measure load imbalance
                        MPI_Allreduce(&it_compute_time, it_stats.max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
                        MPI_Allreduce(&it_compute_time, it_stats.sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);
                        it_stats.update_cumulative_load_imbalance_slowdown();
                        it_compute_time = *it_stats.max_it_time();

                        cum_li_hist[i] = it_stats.get_cumulative_load_imbalance_slowdown();
                        dec_hist[i]    = node->decision == DoLB && i == 0;


                        if (node->decision == DoLB && i == 0) {
                            PAR_START_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            Zoltan_Do_LB<N>(&mesh_data, load_balancer);
                            PAR_END_TIMER(lb_time_spent, MPI_COMM_WORLD);
                            MPI_Allreduce(MPI_IN_PLACE, &lb_time_spent,  1, MPI_TIME, MPI_MAX, comm);
                            *it_stats.get_lb_time_ptr() = lb_time_spent;
                            it_stats.reset_load_imbalance_slowdown();
                            it_compute_time += lb_time_spent;
                        } else {
                            Zoltan_Migrate_Particles<N>(mesh_data.els, load_balancer, datatype, comm);
                        }
                        time_hist[i]   = i == 0 ? starting_time + it_compute_time : time_hist[i-1] + it_compute_time;

                        bbox      = get_bounding_box<N>(params->rc, mesh_data.els);
                        borders   = get_border_cells_index<N>(load_balancer, bbox, params->rc);
                        remote_el = get_ghost_data<N>(load_balancer, mesh_data.els, &head, &lscl, bbox, borders, params->rc, datatype, comm);
                        comp_time += it_compute_time;
                    }
                    node->set_cost(comp_time);
                    pQueue.insert(node);
                    if(node->end_it < nb_iterations)
                        rollback_data.at(next_frame) = mesh_data;
                }
                MPI_Barrier(comm);
            }
        }
    } while(solutions.size() < 1);

    LBSolutionPath solution_path;

    auto solution = solutions[0];
    Time total_time = solution->cost();
    LBLiHist cumulative_load_imbalance;
    LBDecHist decisions;
    TimeHistory time_hist;
    auto it_li = cumulative_load_imbalance.begin();
    auto it_dec= decisions.begin();
    auto it_time= time_hist.begin();
    while (solution->start_it >= 0) { //reconstruct path
        solution_path.push_back(solution);
        it_li = cumulative_load_imbalance.insert(it_li, solution->li_slowdown_hist.begin(), solution->li_slowdown_hist.end());
        it_dec= decisions.insert(it_dec, solution->dec_hist.begin(), solution->dec_hist.end());
        it_time= time_hist.insert(it_time, solution->time_hist.begin(), solution->time_hist.end());
        solution = solution->parent;
    }

    std::reverse(solution_path.begin(), solution_path.end());

    spdlog::drop("particle_logger");
    spdlog::drop("lb_times_logger");
    spdlog::drop("lb_cmplx_logger");
    spdlog::drop("frame_time_logger");
    spdlog::drop("frame_cmplx_logger");
    return {solution_path, cumulative_load_imbalance, decisions, time_hist};
}
#endif //NBMPI_SHORTEST_PATH_HPP