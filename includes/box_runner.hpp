//
// Created by xetql on 05.03.18.
//

#ifndef NBMPI_BOXRUNNER_HPP
#define NBMPI_BOXRUNNER_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <zoltan.h>

#include "../includes/ljpotential.hpp"
#include "../includes/report.hpp"
#include "../includes/physics.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/utils.hpp"
#include "../includes/geometric_load_balancer.hpp"
#include "../includes/params.hpp"
#include "../includes/spatial_elements.hpp"
#include "../includes/graph.hpp"
#include "../includes/metrics.hpp"

#include "zoltan_fn.hpp"
#include <gsl/gsl_statistics.h>
template<int N>
void run_box(FILE* fp, // Output file (at 0)
             const int npframe, // Steps per frame
             const int nframes, // Frames
             const double dt, // Time step
             std::vector<elements::Element<2>> local_elements,
             std::vector<partitioning::geometric::Domain<N>> domain_boundaries,
             load_balancing::geometric::GeometricLoadBalancer<N> load_balancer,
             const sim_param_t* params,
             MPI_Comm comm = MPI_COMM_WORLD) // Simulation params
{
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

    if(params->record) load_balancing::gather_elements_on(nproc, rank, params->npart, local_elements, 0, recv_buf, load_balancer.get_element_datatype(), comm);
    if (rank == 0) {
        auto date = get_date_as_string();
        lb_file.open("load_imbalance_report-"+date+".data", std::ofstream::out | std::ofstream::trunc );
        write_report_header(lb_file, params, rank);
        if(params->record) {
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
            if (params->one_shot_lb_call == (i+(frame-1)*npframe) || params->lb_interval > 0 && ((i+(frame-1)*npframe) % params->lb_interval) == 0) {
                load_balancing::gather_elements_on(nproc, rank, params->npart, local_el, 0, local_el, load_balancer.get_element_datatype(), comm);
                partitioning::geometric::Domain<N> _domain_boundary = {
                        std::make_pair(0.0, params->simsize), std::make_pair(0.0, params->simsize)};
                domain_boundaries = { _domain_boundary };
                load_balancer.load_balance(local_el, domain_boundaries);
            }

            // get particles that can potentially interact with mine
            std::vector<elements::Element<2>> remote_el = load_balancing::geometric::exchange_data<2>(local_el, domain_boundaries,datatype , comm);
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
            load_balancing::geometric::migrate_particles<2>(local_el, domain_boundaries,datatype , comm);

            double diff = (MPI_Wtime() - start) / 1e-3; //divide time by tick resolution
            std::vector<double> times(nproc);
            MPI_Gather(&diff, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            write_report_data(lb_file, i+(frame-1)*npframe, times, rank);
        }

        if(params->record) load_balancing::gather_elements_on(nproc, rank, params->npart,
                                           local_el, 0, recv_buf, load_balancer.get_element_datatype(), comm);
        if (rank == 0) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            if(params->record) write_frame_data(fp, params->npart, &recv_buf[0]);
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }

    load_balancer.stop();
    if(rank == 0){
        double diff =(MPI_Wtime() - start_sim);
        lb_file << diff << std::endl;
        lb_file.close();
    }
}

template<int N>
void zoltan_run_box_dataset(FILE* fp,          // Output file (at 0)
                    MESH_DATA<N>* mesh_data,
                    Zoltan_Struct* load_balancer,
                    const sim_param_t* params,
                    const MPI_Comm comm = MPI_COMM_WORLD)
{
    int nproc,rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream dataset;
    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    const int WINDOW_SIZE = 50;
    const double tick_freq = 1e-3;
    int dim;
    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig
    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size
    auto date = get_date_as_string();

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for(int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax, params->simsize);
        domain_boundaries[part] = domain;
    }
    std::unique_ptr<SlidingWindow<double>> window_load_imbalance, window_complexity, window_loads;
    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    std::vector<elements::Element<N>> remote_el;

    if (rank == 0) { // Write report and outputs ...
        window_load_imbalance = std::make_unique<SlidingWindow<double>>(WINDOW_SIZE); //sliding window of size 50
        window_loads          = std::make_unique<SlidingWindow<double>>(WINDOW_SIZE); //sliding window of size 50
        window_complexity     = std::make_unique<SlidingWindow<double>>(WINDOW_SIZE); //sliding window of size 50
    }

    std::vector<float> dataset_entry(13);

    double total_metric_computation_time = 0.0;
    double compute_time_after_lb = 0.0;
    for (int frame = 0; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            if((params->one_shot_lb_call + DELTA_LB_CALL) == (i+frame*npframe) ) {
                if(rank == 0) {
                    std::cout << " Time within "<< ((i+frame*npframe)-DELTA_LB_CALL) << " and "
                              <<  (i+frame*npframe)<<": "<< compute_time_after_lb << " ms. "
                              << ", metrics: "<< total_metric_computation_time << std::endl;
                    dataset_entry[dataset_entry.size() - 1] = compute_time_after_lb;
                    dataset.open("dataset-rcb-"+std::to_string(params->seed)+
                                 "-"+std::to_string(params->world_size)+
                                 "-"+std::to_string(params->npart)+
                                 "-"+std::to_string((params->T0))+
                                 "-"+std::to_string((params->G))+
                                 "-"+std::to_string((params->eps_lj))+
                                 "-"+std::to_string((params->sig_lj)),
                                 std::ofstream::out | std::ofstream::app | std::ofstream::binary);
                    write_report_data_bin<float>(dataset, params->one_shot_lb_call, dataset_entry, rank);
                    dataset.close();
                    std::cout << " Go to the next experiment. " << std::endl;
                }
                return;
            }
            MPI_Barrier(comm);
            float start = MPI_Wtime();
            if ((params->one_shot_lb_call == (i+frame*npframe) ) && (i+frame*npframe) > 0) {
                compute_time_after_lb = 0.0;
                zoltan_fn_init(load_balancer, mesh_data);
                Zoltan_LB_Partition(load_balancer,           /* input (all remaining fields are output) */
                                         &changes,           /* 1 if partitioning was changed, 0 otherwise */
                                         &numGidEntries,     /* Number of integers used for a global ID */
                                         &numLidEntries,     /* Number of integers used for a local ID */
                                         &numImport,         /* Number of vertices to be sent to me */
                                         &importGlobalGids,  /* Global IDs of vertices to be sent to me */
                                         &importLocalGids,   /* Local IDs of vertices to be sent to me */
                                         &importProcs,       /* Process rank for source of each incoming vertex */
                                         &importToPart,      /* New partition for each incoming vertex */
                                         &numExport,         /* Number of vertices I must send to other processes*/
                                         &exportGlobalGids,  /* Global IDs of the vertices I must send */
                                         &exportLocalGids,   /* Local IDs of the vertices I must send */
                                         &exportProcs,       /* Process to which I send each of the vertices */
                                         &exportToPart);     /* Partition to which each vertex will belong */
                if(changes) for(int part = 0; part < nproc; ++part) { // algorithm specific ...
                    Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
                    auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin,
                                                                                xmax, ymax, zmax, params->simsize);
                    domain_boundaries[part] = domain;
                }

                load_balancing::geometric::migrate_zoltan<N>(mesh_data->els, numImport, numExport, exportProcs,
                                                             exportGlobalGids, datatype, MPI_COMM_WORLD);
                MPI_Barrier(comm);
                Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
                Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
            }

            remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm, lsub);

            // update local ids
            for(size_t i = 0; i < mesh_data->els.size(); ++i) mesh_data->els[i].lid = i;

            lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
            float complexity = (float) lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);

            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);

            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);

            MPI_Barrier(comm);
            float iteration_time = (MPI_Wtime() - start) / tick_freq;
            compute_time_after_lb += iteration_time;
            if((i+frame*npframe) > params->one_shot_lb_call - (WINDOW_SIZE) && (i+frame*npframe) < params->one_shot_lb_call) {
                double start_metric = MPI_Wtime();

                // Retrieve local data to Master PE
                std::vector<float> times(nproc);
                MPI_Gather(&iteration_time, 1, MPI_FLOAT, &times.front(), 1, MPI_FLOAT, 0, comm);

                std::vector<float> complexities(nproc);
                MPI_Gather(&complexity, 1, MPI_FLOAT, &complexities.front(), 1, MPI_FLOAT, 0, comm);

                if (rank == 0) {
                    float gini_times =  metric::load_balancing::compute_gini_index(times);
                    //float gini_loads = 0.0;//metric::load_balancing::compute_gini_index(loads);
                    float gini_complexities = 0.0;//metric::load_balancing::compute_gini_index(complexities);
                    float skewness_times = gsl_stats_float_skew(&times.front(), 1, times.size());
                    //float skewness_loads = gsl_stats_float_skew(&loads.front(), 1, loads.size());
                    float skewness_complexities = gsl_stats_float_skew(&complexities.front(), 1, complexities.size());

                    //window_loads->add(gini_loads);
                    window_complexity->add(gini_complexities);
                    window_load_imbalance->add(gini_times);

                    // Generate y from 0 to 1 and store in a vector
                    std::vector<float> it(window_load_imbalance->data_container.size()); std::iota(it.begin(), it.end(), 0);

                    float slope_load_imbalance = statistic::linear_regression(it, window_load_imbalance->data_container).first;
                    float macd_load_imbalance = metric::load_dynamic::compute_macd_ema(window_load_imbalance->data_container, 12, 26,  2.0/(window_load_imbalance->data_container.size()+1));

                    float slope_complexity = statistic::linear_regression(it, window_complexity->data_container).first;
                    float macd_complexity = metric::load_dynamic::compute_macd_ema(window_complexity->data_container, 12, 26,  1.0/(window_complexity->data_container.size()+1));

                    dataset_entry = {
                        gini_times, gini_complexities,
                        skewness_times, skewness_complexities,
                        slope_load_imbalance, slope_complexity,
                        macd_load_imbalance, macd_complexity, 0.0
                    };
                }
                total_metric_computation_time += (MPI_Wtime() - start_metric) / tick_freq;
            } // end of metric computation
        } // end of time-steps
    } // end of frames
    if(rank==0) dataset.close();

    if(rank == 0) {
        std::cout << " Time within "<< params->one_shot_lb_call << " and "
                  <<  params->npframe*params->nframes <<": "<< compute_time_after_lb << " ms. "
                  << ", metrics: "<< total_metric_computation_time << std::endl;
        dataset_entry[dataset_entry.size() - 1] = compute_time_after_lb;
        dataset.open("dataset-rcb-"+std::to_string(params->seed)+
                     "-"+std::to_string(params->world_size)+
                     "-"+std::to_string(params->npart)+
                     "-"+std::to_string((params->nframes*params->npframe))+
                     "-"+std::to_string((params->T0))+
                     "-"+std::to_string((params->G))+
                     "-"+std::to_string((params->eps_lj))+
                     "-"+std::to_string((params->sig_lj)),
                     std::ofstream::out | std::ofstream::app | std::ofstream::binary);
        write_report_data_bin<float>(dataset, params->one_shot_lb_call, dataset_entry, rank);
        dataset.close();
        std::cout << " Go to the next experiment. " << std::endl;
    }
}


template<int N>
void compute_dataset_base_gain(FILE* fp,          // Output file (at 0)
                            MESH_DATA<N>* mesh_data,
                            Zoltan_Struct* load_balancer,
                            const sim_param_t* params,
                            const MPI_Comm comm = MPI_COMM_WORLD)
{
    int nproc,rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream dataset;
    if(rank==0)
        dataset.open("dataset-base-times-rcb-"+std::to_string(params->seed)+
                     "-"+std::to_string(params->world_size)+
                     "-"+std::to_string(params->npart)+
                     "-"+std::to_string((params->nframes*params->npframe))+
                     "-"+std::to_string((params->T0))+
                     "-"+std::to_string((params->G))+
                     "-"+std::to_string((params->eps_lj))+
                     "-"+std::to_string((params->sig_lj)),
                     std::ofstream::out | std::ofstream::app | std::ofstream::binary);
    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    const double tick_freq = 1e-3;

    // ZOLTAN VARIABLES
    int dim;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for(int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax, params->simsize);
        domain_boundaries[part] = domain;
    }

    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;
    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig
    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size

    std::vector<elements::Element<N>> remote_el;
    double compute_time_after_lb = 0.0;
    for (int frame = 0; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            if((i+frame*npframe) % DELTA_LB_CALL == 0 && (i+frame*npframe) > 0 && rank == 0) {
                std::cout << " Time within "<< ((i+frame*npframe)-DELTA_LB_CALL) << " and " <<  (i+frame*npframe)<< " is " << compute_time_after_lb << " ms" << std::endl;
                write_metric_data_bin(dataset, (i+frame*npframe) - DELTA_LB_CALL, {compute_time_after_lb}, rank);
                compute_time_after_lb = 0.0;
            }
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm, lsub);
            lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
            lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);
            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);
            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);
            MPI_Barrier(comm);
            double iteration_time = (MPI_Wtime() - start) / tick_freq;
            compute_time_after_lb += iteration_time;

        } // end of time-steps
    } // end of frames
    if(rank == 0) dataset.close();
}

template<int N>
void zoltan_run_box(FILE* fp,          // Output file (at 0)
                    MESH_DATA<N>* mesh_data,
                    Zoltan_Struct* load_balancer,
                    const sim_param_t* params,
                    const MPI_Comm comm = MPI_COMM_WORLD)
{
    int nproc,rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream lb_file, metric_file, frame_file;
    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    int dim;

    SimpleXYZFormatter frame_formater;

    // ZOLTAN VARIABLES
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    double xmin, ymin, zmin, xmax, ymax, zmax;
    // END OF ZOLTAN VARIABLES

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for(int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax, params->simsize);
        domain_boundaries[part] = domain;
    }

    double start_sim = MPI_Wtime();
    std::unordered_map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    double rm = 3.2 * params->sig_lj; // r_m = 3.2 * sig

    int M = std::ceil(params->simsize / rm); // number of cell in a row
    float lsub = rm; //cell size

    std::vector<elements::Element<N>> recv_buf(params->npart);
    auto date = get_date_as_string();
    if(params->record) load_balancing::gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
    if (rank == 0) { // Write report and outputs ...
        lb_file.open("LIr_"+std::to_string(params->world_size)+
                     "-"+std::to_string(params->npart)+
                     "-"+std::to_string((params->nframes*params->npframe))+
                     "-"+std::to_string((int)(params->T0))+
                     "-"+std::to_string((params->G))+
                     "-"+std::to_string((params->eps_lj))+
                     "-"+std::to_string((params->sig_lj))+
                     "-"+std::to_string((params->one_shot_lb_call))+
                     "_"+date+".data", std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        write_report_header_bin(lb_file, params, rank);     // write header

        metric_file.open(params->uuid+"-"+date+".metrics",  std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        write_report_header_bin(metric_file, params, rank, rank); // write the same header

        if(params->record) {
            //write_header(fp, params->npart, params->simsize);
            //write_frame_data(fp, params->npart, &recv_buf[0]);
            frame_file.open("run_cpp.out", std::ofstream::out | std::ofstream::trunc);
            frame_formater.write_header(frame_file, params->npframe, params->simsize);
            write_frame_data(frame_file, recv_buf, frame_formater, params);
        }

    }
    std::vector<elements::Element<N>> remote_el;
    double begin = MPI_Wtime();
    for (int frame = 0; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            MPI_Barrier(comm);
            // double start = MPI_Wtime();
            // Load balance criteria...
            if (params->one_shot_lb_call == (i+frame*npframe) || (params->lb_interval > 0 && ((i+frame*npframe) % params->lb_interval) == 0)) {
                zoltan_fn_init(load_balancer, mesh_data);
                Zoltan_LB_Partition(load_balancer,      /* input (all remaining fields are output) */
                                    &changes,           /* 1 if partitioning was changed, 0 otherwise */
                                    &numGidEntries,     /* Number of integers used for a global ID */
                                    &numLidEntries,     /* Number of integers used for a local ID */
                                    &numImport,         /* Number of vertices to be sent to me */
                                    &importGlobalGids,  /* Global IDs of vertices to be sent to me */
                                    &importLocalGids,   /* Local IDs of vertices to be sent to me */
                                    &importProcs,       /* Process rank for source of each incoming vertex */
                                    &importToPart,      /* New partition for each incoming vertex */
                                    &numExport,         /* Number of vertices I must send to other processes*/
                                    &exportGlobalGids,  /* Global IDs of the vertices I must send */
                                    &exportLocalGids,   /* Local IDs of the vertices I must send */
                                    &exportProcs,       /* Process to which I send each of the vertices */
                                    &exportToPart);     /* Partition to which each vertex will belong */
                if(changes) for(int part = 0; part < nproc; ++part) {
                    Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
                    auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax, params->simsize);
                    domain_boundaries[part] = domain;
                }
                load_balancing::geometric::migrate_zoltan<N>(mesh_data->els, numImport, numExport, exportProcs,
                                                             exportGlobalGids, datatype, MPI_COMM_WORLD);
                Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
                Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
            }
            remote_el = load_balancing::geometric::exchange_data<N>(mesh_data->els, domain_boundaries, datatype, comm, lsub);

            // update local ids
            for(size_t i = 0; i < mesh_data->els.size(); ++i) mesh_data->els[i].lid = i;

            lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);

            lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);

            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);

            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);

            // END OF COMPUTATION
            ////////////////////////////////////////////////////////////////////////////////////////
            // Retrieve timings
            //float diff = (MPI_Wtime() - start) / 1e-3; // divide diff time by tick resolution
            //std::vector<float> times(nproc);
            //MPI_Gather(&diff, 1, MPI_FLOAT, &times.front(), 1, MPI_FLOAT, 0, comm);
            //write_report_data_bin(lb_file, i+frame*npframe, times, rank, 0);
            ////////////////////////////////////////////////////////////////////////////////////////
        }
        // Send metrics

        // Write metrics to report file
        if(params->record)
            load_balancing::gather_elements_on(nproc, rank, params->npart,
                                               mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
        MPI_Barrier(comm);
        if (rank == 0) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            if(params->record) {
                write_frame_data(frame_file, recv_buf, frame_formater, params);
                //write_frame_data(fp, params->npart, &recv_buf[0]);
            }
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }

    MPI_Barrier(comm);
    if(rank == 0){
        double diff = (MPI_Wtime() - start_sim) / 1e-3;
        write_report_total_time_bin<float>(lb_file, diff, rank);
        lb_file.close();
        frame_file.close();
        metric_file.close();
    }
}


#endif //NBMPI_BOXRUNNER_HPP
