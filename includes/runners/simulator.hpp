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
#include <zoltan.h>
#include <cstdlib>
#include <gsl/gsl_statistics.h>

#include "../ljpotential.hpp"
#include "../report.hpp"
#include "../physics.hpp"
#include "../nbody_io.hpp"
#include "../utils.hpp"

#include "../geometric_load_balancer.hpp"
#include "../params.hpp"
#include "../spatial_elements.hpp"
#include "../graph.hpp"
#include "../metrics.hpp"
#include "../zoltan_fn.hpp"
#include "../decision_makers/strategy.hpp"

template<int N>
double simulate(FILE *fp,          // Output file (at 0)
            MESH_DATA<N> *mesh_data,
            Zoltan_Struct *load_balancer,
            std::shared_ptr<decision_making::Policy> lb_policy,
            sim_param_t *params,
            const MPI_Comm comm = MPI_COMM_WORLD) {
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream lb_file, metric_file, frame_file;

    const int nframes = params->nframes;
    const int npframe = params->npframe;

    SimpleCSVFormatter frame_formater(',');

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries = retrieve_domain_boundaries<N>(load_balancer, nproc, params);
    std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    //double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig

    std::vector<elements::Element<N>> recv_buf(params->npart);
    auto date = get_date_as_string();

    if (params->record)
        load_balancing::gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf,
                                           datatype.elements_datatype, comm);
    if (params->record && !rank) {
        std::string mkdir_cmd = "mkdir -p data/time-series/"+std::to_string(params->seed);
        system(mkdir_cmd.c_str());
        frame_file.open("data/time-series/"+std::to_string(params->seed)+"/run_cpp.csv.0", std::ofstream::out | std::ofstream::trunc);
        frame_formater.write_header(frame_file, params->npframe, params->simsize);
        write_frame_data<N>(frame_file, recv_buf, frame_formater, params);
        frame_file.close();
    }

    int nb_lb = 0;
    //std::vector<elements::Element<N>> remote_el;
    metric::LBMetrics<double>* a = new metric::LBMetrics<double>({0.0});
    std::vector<double> times(nproc);
    MESH_DATA<N> tmp_data;

    double total_time = 0.0;
    int complexity, received, sent;
    for (int frame = 0; frame < nframes; ++frame) {
        double frame_time = 0.0;
        for (int i = 0; i < npframe; ++i) {
            double it_time;
            MPI_Barrier(comm); //wait for all to do communications
            double begin = MPI_Wtime(); //start of step

            if (lb_policy->should_load_balance(i + frame * npframe, a /* should be replaced by the metrics */)){
                zoltan_load_balance<N>(mesh_data, domain_boundaries, load_balancer, nproc, params, datatype, comm);
                nb_lb ++;
            } else {
                load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);
            }

            MPI_Barrier(comm); //everybody've finished communications
            //everybody computes a step
            auto computation_info = lennard_jones::compute_one_step<N>(mesh_data, plklist, domain_boundaries, datatype, params, comm, frame);

            double end = MPI_Wtime();// End of step
            //compute my own time
            it_time = (end - begin);
            //everybody share their time
            MPI_Allgather(&it_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, comm);

            double true_iteration_time = *std::max_element(times.begin(), times.end());

            frame_time += true_iteration_time;

            std::tie(complexity, received, sent) = computation_info;

            std::vector<double> complexities(nproc);
            double cmplx = complexity;
            MPI_Allgather(&cmplx, 1, MPI_DOUBLE, &complexities.front(), 1, MPI_DOUBLE, comm);
            double gini_complexities = metric::load_balancing::compute_gini_index(complexities);
            delete a;
            a = new metric::LBMetrics<double>({gini_complexities});
        }

        if(!rank) {
            printf("Frame [%d] completed in %f seconds\n", frame, frame_time);
        }
        total_time += frame_time;

        // Write metrics to report file
        if (params->record)
            load_balancing::gather_elements_on(nproc, rank, params->npart,
                                               mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
        if (rank == 0) {
	        if (params->record) {
                frame_file.open("data/time-series/"+std::to_string(params->seed)+"/run_cpp.csv."+std::to_string(frame+1), std::ofstream::out | std::ofstream::trunc);
                frame_formater.write_header(frame_file, params->npframe, params->simsize);
                write_frame_data(frame_file, recv_buf, frame_formater, params);
                frame_file.close();
            }
        }
    }

    MPI_Barrier(comm);
    if (!rank) std::cout << "nb lb = " << nb_lb << std::endl;
    if (rank == 0 && frame_file.is_open()) frame_file.close();
    delete a;
    return total_time;
}

/*
template<int N>
void simulate_box_custom(FILE *fp, // Output file (at 0)
             const int npframe, // Steps per frame
             const int nframes, // Frames
             const double dt, // Time step
             std::vector<elements::Element<2>> local_elements,
             std::vector<partitioning::geometric::Domain<N>> domain_boundaries,
             load_balancing::geometric::GeometricLoadBalancer<N> load_balancer,
             const sim_param_t *params,
             MPI_Comm comm = MPI_COMM_WORLD) {
    std::ofstream lb_file;
    partitioning::CommunicationDatatype datatype = elements::register_datatype<2>();
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    double start_sim = MPI_Wtime();

    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    double rm = 3.2 * std::sqrt(params->sig_lj); // r_m = 3.2 * sig
    int M = (int) (params->simsize / rm); //number of cell in a row
    float lsub = params->simsize / ((float) M); //cell size
    std::vector<elements::Element<2>> recv_buf(params->npart);

    if (params->record)
        load_balancing::gather_elements_on(nproc, rank, params->npart, local_elements, 0, recv_buf,
                                           load_balancer.get_element_datatype(), comm);
    if (rank == 0) {
        auto date = get_date_as_string();
        lb_file.open("load_imbalance_report-" + date + ".data", std::ofstream::out | std::ofstream::trunc);
        write_report_header(lb_file, params, rank);
        if (params->record) {
            write_header(fp, params->npart, params->simsize);
            write_frame_data(fp, params->npart, &recv_buf[0]);
        }
    }

    auto local_el = local_elements;
    double begin = MPI_Wtime();
    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            // Rebalance if asked
            if (params->one_shot_lb_call == (i + (frame - 1) * npframe) ||
                params->lb_interval > 0 && ((i + (frame - 1) * npframe) % params->lb_interval) == 0) {
                load_balancing::gather_elements_on(nproc, rank, params->npart, local_el, 0, local_el,
                                                   load_balancer.get_element_datatype(), comm);
                partitioning::geometric::Domain<N> _domain_boundary = {
                        std::make_pair(0.0, params->simsize), std::make_pair(0.0, params->simsize)};
                domain_boundaries = {_domain_boundary};
                load_balancer.load_balance(local_el, domain_boundaries);
            }
            int r, s;
            // get particles that can potentially interact with mine
            std::vector<elements::Element<2>> remote_el = load_balancing::geometric::exchange_data<2>(local_el,
                                                                                                      domain_boundaries,
                                                                                                      datatype, comm, r,
                                                                                                      s);
            //select computation method
            switch (params->computation_method) {
                case 1:
                    lennard_jones::compute_forces(local_el, remote_el, params);
                    break;
                case 2:
                case 3:
                    lennard_jones::create_cell_linkedlist(M, lsub, local_el, remote_el, plklist);
                    lennard_jones::compute_forces(M, lsub, local_el, remote_el, plklist, params);
                    break;
            }
            leapfrog2(dt, local_el);
            leapfrog1(dt, local_el);
            apply_reflect(local_el, params->simsize);
            //finish this time step by sending particles that does not belong to me anymore...
            load_balancing::geometric::migrate_particles<2>(local_el, domain_boundaries, datatype, comm);

            double diff = (MPI_Wtime() - start) / 1e-3; //divide time by tick resolution
            std::vector<double> times(nproc);
            MPI_Gather(&diff, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            write_report_data(lb_file, i + (frame - 1) * npframe, times, rank);
        }

        if (params->record)
            load_balancing::gather_elements_on(nproc, rank, params->npart,
                                               local_el, 0, recv_buf, load_balancer.get_element_datatype(), comm);
        if (rank == 0) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            if (params->record) write_frame_data(fp, params->npart, &recv_buf[0]);
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }

    load_balancer.stop();
    if (rank == 0) {
        double diff = (MPI_Wtime() - start_sim);
        lb_file << diff << std::endl;
        lb_file.close();
    }
}
*/


#endif //NBMPI_SIMULATE_HPP
