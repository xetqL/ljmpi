//
// Created by xetql on 27.06.18.
//

#ifndef NBMPI_SIMULATE_HPP
#define NBMPI_SIMULATE_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <cstdlib>

#include "../decision_makers/strategy.hpp"

#include "../ljpotential.hpp"
#include "../nbody_io.hpp"
#include "../utils.hpp"
#include "../parallel_utils.hpp"

#include "../params.hpp"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

using ApplicationTime = Time;
using CumulativeLoadImbalanceHistory = std::vector<Time>;
using TimeHistory = std::vector<Time>;
using Decisions = std::vector<int>;
template<int N> using Position  = std::array<Real, N>;
template<int N> using Velocity  = std::array<Real, N>;


template<int N, class T, class D, class LoadBalancer, class Wrapper>
std::tuple<ApplicationTime, CumulativeLoadImbalanceHistory, Decisions, TimeHistory>
        simulate(
              LoadBalancer* LB,
              MESH_DATA<T> *mesh_data,
              decision_making::LBPolicy<D>&& lb_policy,
              Wrapper fWrapper,
              sim_param_t *params,
              Probe* probe,
              MPI_Datatype datatype,
              const MPI_Comm comm = MPI_COMM_WORLD,
              const std::string output_names_prefix = "") {

    auto boxIntersectFunc   = fWrapper.getBoxIntersectionFunc();
    auto doLoadBalancingFunc= fWrapper.getLoadBalancingFunc();
    auto pointAssignFunc    = fWrapper.getPointAssignationFunc();
    auto getPosPtrFunc      = fWrapper.getPosPtrFunc();
    auto getVelPtrFunc      = fWrapper.getVelPtrFunc();
    auto getForceFunc       = fWrapper.getForceFunc();

    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    const int nframes = params->nframes;
    const int npframe = params->npframe;

    SimpleCSVFormatter frame_formater(',');

    auto time_logger = spdlog::basic_logger_mt("frame_time_logger", "logs/"+output_names_prefix+std::to_string(params->seed)+"/time/frame-p"+std::to_string(rank)+".txt");
    auto cmplx_logger = spdlog::basic_logger_mt("frame_cmplx_logger", "logs/"+output_names_prefix+std::to_string(params->seed)+"/complexity/frame-p"+std::to_string(rank)+".txt");

    time_logger->set_pattern("%v");
    cmplx_logger->set_pattern("%v");

    std::vector<T> recv_buf;
    if (params->record) {
        recv_buf.reserve(params->npart);
        gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);
        if (!rank) {
            auto particle_logger = spdlog::basic_logger_mt("particle_logger", "logs/"+output_names_prefix+std::to_string(params->seed)+"/frames/particles.csv.0");
            particle_logger->set_pattern("%v");
            std::stringstream str;
            frame_formater.write_header(str, params->npframe, params->simsize);
            write_frame_data<N>(str, recv_buf, [](auto& e){return e.position;}, frame_formater);
            particle_logger->info(str.str());
        }
    }

    std::vector<Time> times(nproc), my_frame_times(nframes);
    std::vector<Index> lscl(mesh_data->els.size()), head;
    std::vector<Complexity> my_frame_cmplx(nframes);

    // Compute my bounding box as function of my local data
    auto bbox      = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data->els);
    // Compute which cells are on my borders
    auto borders   = get_border_cells_index<N>(LB, bbox, params->rc, boxIntersectFunc, comm);
    // Get the ghost data from neighboring processors
    auto remote_el = get_ghost_data<N>(mesh_data->els, getPosPtrFunc, &head, &lscl, bbox, borders, params->rc, datatype, comm);

    const int nb_data = mesh_data->els.size();
    for(int i = 0; i < nb_data; ++i) mesh_data->els[i].lid = i;
    ApplicationTime app_time = 0.0;
    CumulativeLoadImbalanceHistory cum_li_hist; cum_li_hist.reserve(nframes*npframe);
    TimeHistory time_hist; time_hist.reserve(nframes*npframe);
    Decisions dec; dec.reserve(nframes*npframe);
    Time total_time = 0.0;
    for (int frame = 0; frame < nframes; ++frame) {
        Time comp_time = 0.0;
        Complexity complexity = 0;
        for (int i = 0; i < npframe; ++i) {
            START_TIMER(it_compute_time);
            complexity += lj::compute_one_step<N>(mesh_data->els, remote_el, getPosPtrFunc, getVelPtrFunc, &head, &lscl, bbox,  getForceFunc, borders, params);
            END_TIMER(it_compute_time);

            // Measure load imbalance
            MPI_Allreduce(&it_compute_time, probe->max_it_time(), 1, MPI_TIME, MPI_MAX, comm);
            MPI_Allreduce(&it_compute_time, probe->min_it_time(), 1, MPI_TIME, MPI_MIN, comm);
            MPI_Allreduce(&it_compute_time, probe->sum_it_time(), 1, MPI_TIME, MPI_SUM, comm);
            probe->update_cumulative_imbalance_time();
            it_compute_time = *probe->max_it_time();

            if(probe->is_balanced()) {
                probe->update_lb_parallel_efficiencies();
            }

            bool lb_decision = lb_policy.should_load_balance();

            cum_li_hist.push_back(probe->get_cumulative_imbalance_time());
            dec.push_back(lb_decision);

            if (lb_decision) {
                PAR_START_TIMER(lb_time_spent, MPI_COMM_WORLD);
                doLoadBalancingFunc(LB, mesh_data);
                PAR_END_TIMER(lb_time_spent, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &lb_time_spent,  1, MPI_TIME, MPI_MAX, MPI_COMM_WORLD);
                probe->push_load_balancing_time(lb_time_spent);
                probe->reset_cumulative_imbalance_time();
                it_compute_time += lb_time_spent;
                if(!rank) {
                    std::cout << "Average C = " << probe->compute_avg_lb_time() << std::endl;
                }
            } else {
                migrate_data(LB, mesh_data->els, pointAssignFunc, datatype, comm);
            }

            probe->set_balanced(lb_decision);

            total_time += it_compute_time;
            time_hist.push_back(total_time);

            bbox      = get_bounding_box<N>(params->rc, getPosPtrFunc, mesh_data->els);
            borders   = get_border_cells_index<N>(LB, bbox, params->rc, boxIntersectFunc, comm);
            remote_el = get_ghost_data<N>(mesh_data->els, getPosPtrFunc, &head, &lscl, bbox, borders, params->rc, datatype, comm);

            comp_time += it_compute_time;
            probe->next_iteration();
        }

        probe->batch_time = comp_time;
        if(!rank) std::cout << probe->batch_time << std::endl;

        app_time += probe->batch_time;

        // Write metrics to report file
        if (params->record) {
            time_logger->info("{:0.6f}", probe->batch_time);
            cmplx_logger->info("{}", complexity);

            if(frame % 5 == 0) { time_logger->flush(); cmplx_logger->flush(); }

            gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype, comm);

            if (rank == 0) {
                spdlog::drop("particle_logger");
                auto particle_logger = spdlog::basic_logger_mt("particle_logger", "logs/"+output_names_prefix+std::to_string(params->seed)+"/frames/particles.csv."+ std::to_string(frame + 1));
                particle_logger->set_pattern("%v");
                std::stringstream str;
                frame_formater.write_header(str, params->npframe, params->simsize);
                write_frame_data<N>(str, recv_buf, [](auto& e){return e.position;}, frame_formater);
                particle_logger->info(str.str());
            }
        }

        my_frame_times[frame] = probe->batch_time;
        my_frame_cmplx[frame] = complexity;
    }

    MPI_Barrier(comm);
    std::vector<Time> max_times(nframes), min_times(nframes), avg_times(nframes);
    Time sum_times;
    std::vector<Complexity > max_cmplx(nframes), min_cmplx(nframes), avg_cmplx(nframes);
    Complexity sum_cmplx;

    std::shared_ptr<spdlog::logger> lb_time_logger;
    std::shared_ptr<spdlog::logger> lb_cmplx_logger;

    if(!rank){
        lb_time_logger = spdlog::basic_logger_mt("lb_times_logger", "logs/"+output_names_prefix+std::to_string(params->seed)+"/time/frame_statistics.txt");
        lb_time_logger->set_pattern("%v");
        lb_cmplx_logger = spdlog::basic_logger_mt("lb_cmplx_logger", "logs/"+output_names_prefix+std::to_string(params->seed)+"/complexity/frame_statistics.txt");
        lb_cmplx_logger->set_pattern("%v");
    }

    for (int frame = 0; frame < nframes; ++frame) {
        MPI_Reduce(&my_frame_times[frame], &max_times[frame], 1, MPI_TIME, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_times[frame], &min_times[frame], 1, MPI_TIME, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_times[frame], &sum_times,        1, MPI_TIME, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Reduce(&my_frame_cmplx[frame], &max_cmplx[frame], 1, MPI_COMPLEXITY, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_cmplx[frame], &min_cmplx[frame], 1, MPI_COMPLEXITY, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_frame_cmplx[frame], &sum_cmplx,        1, MPI_COMPLEXITY, MPI_SUM, 0, MPI_COMM_WORLD);

        if(!rank) {
            avg_times[frame] = sum_times / nproc;
            avg_cmplx[frame] = sum_cmplx / nproc;
            lb_time_logger->info("{}\t{}\t{}\t{}", max_times[frame], min_times[frame], avg_times[frame], (max_times[frame]/avg_times[frame]-1.0));
            lb_cmplx_logger->info("{}\t{}\t{}\t{}", max_cmplx[frame], min_cmplx[frame], avg_cmplx[frame], (max_cmplx[frame]/avg_cmplx[frame]-1.0));
        }
    }

    spdlog::drop("particle_logger");
    spdlog::drop("lb_times_logger");
    spdlog::drop("lb_cmplx_logger");
    spdlog::drop("frame_time_logger");
    spdlog::drop("frame_cmplx_logger");

    return { app_time, cum_li_hist, dec, time_hist};
}

#endif //NBMPI_SIMULATE_HPP
