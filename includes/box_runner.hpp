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
#include <zoltan.h>

#include "../includes/ljpotential.hpp"
#include "../includes/report.hpp"
#include "../includes/physics.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/utils.hpp"
#include "../includes/geometric_load_balancer.hpp"
#include "../includes/params.hpp"
#include "../includes/spatial_elements.hpp"
#include "zoltan_fn.hpp"

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

    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    double start_sim = MPI_Wtime();

    std::map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    double rm = 3.2 * std::sqrt(params->sig_lj); // r_m = 3.2 * sig
    int M = (int) (params->simsize / rm); //number of cell in a row
    float lsub = params->simsize / ((float) M); //cell size
    std::vector<elements::Element<2>> recv_buf(params->npart);

    load_balancing::gather_elements_on(nproc, rank, params->npart, local_elements, 0, recv_buf, load_balancer.get_element_datatype(), comm);
    if (fp) {
        auto date = get_date_as_string();
        lb_file.open("load_imbalance_report-"+date+".data", std::ofstream::out | std::ofstream::trunc );
        write_report_header(lb_file, params, rank);
        write_header(fp, params->npart, params->simsize);
        write_frame_data(fp, params->npart, &recv_buf[0]);
    }

    auto local_el = local_elements;
    double begin = MPI_Wtime();
    std::vector<elements::Element<2>> remote_el = load_balancer.exchange_data(local_el, domain_boundaries);
    switch (params->computation_method) {
        case 1: //brute force
            lennard_jones::compute_forces(local_el, remote_el, params);
            break;
        case 2: // cell linked list method
        case 3: // TODO: FME method...
            lennard_jones::create_cell_linkedlist(M, lsub, local_el, remote_el, plklist);
            lennard_jones::compute_forces(M, lsub, local_el, remote_el, plklist, params);
            break;
    }
    leapfrog1(dt, local_el);
    apply_reflect(local_el, params->simsize);

    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            if (params->lb_interval > 0 && ((i+(frame-1)*npframe) % params->lb_interval) == 0) {
                load_balancing::gather_elements_on(nproc, rank, params->npart, local_el, 0, local_el, load_balancer.get_element_datatype(), comm);
                partitioning::geometric::Domain<N> _domain_boundary = {
                        std::make_pair(0.0, params->simsize), std::make_pair(0.0, params->simsize)};
                domain_boundaries = { _domain_boundary };
                load_balancer.load_balance(local_el, domain_boundaries);
            } else load_balancer.migrate_particles(local_el, domain_boundaries);

            remote_el = load_balancer.exchange_data(local_el, domain_boundaries);
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
            double diff = (MPI_Wtime() - start) / 1e-3; //divide time by tick resolution

            std::vector<double> times(nproc);
            MPI_Gather(&diff, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            write_report_data(lb_file, i+(frame-1)*npframe, times, rank);

        }
        load_balancing::gather_elements_on(nproc, rank, params->npart,
                                           local_el, 0, recv_buf, load_balancer.get_element_datatype(), comm);
        if (fp) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            write_frame_data(fp, params->npart, &recv_buf[0]);
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }
    load_balancer.stop();
    if(rank==0){
        double diff =(MPI_Wtime() - start_sim);
        lb_file << diff << std::endl;
        lb_file.close();
    }
}
template<int N=2>
void zoltan_run_box(FILE* fp,          // Output file (at 0)
                    MESH_DATA* mesh_data,
                    Zoltan_Struct* load_balancer,
                    const sim_param_t* params,
                    const MPI_Comm comm = MPI_COMM_WORLD)
{
    int nproc,rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    std::ofstream lb_file;
    const double dt = params->dt;
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    int dim, rc;
    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;
    double xmin, ymin, zmin, xmax, ymax, zmax;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<2>();
    std::vector<partitioning::geometric::Domain<N>> domain_boundaries(nproc);

    // get boundaries of all domains
    for(int part = 0; part < nproc; ++part){
        Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<2>(xmin, ymin, zmin, xmax, ymax, zmax, params->simsize);
        domain_boundaries[part] = domain;
    }
    
    double start_sim = MPI_Wtime();
    std::map<int, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    double rm = 3.2 * std::sqrt(params->sig_lj); // r_m = 3.2 * sig
    int M = (int) (params->simsize / rm); //number of cell in a row
    float lsub = params->simsize / ((float) M); //cell size
    std::vector<elements::Element<2>> recv_buf(params->npart);

    load_balancing::gather_elements_on(nproc, rank, params->npart, mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
    if (fp) {
        auto date = get_date_as_string();
        lb_file.open("load_imbalance_report-"+date+".data", std::ofstream::out | std::ofstream::trunc );
        write_report_header(lb_file, params, rank);
        write_header(fp, params->npart, params->simsize);
        write_frame_data(fp, params->npart, &recv_buf[0]);
    }

    double begin = MPI_Wtime();
    std::vector<elements::Element<2>> remote_el;
    for (int frame = 1; frame < nframes; ++frame) {
        for (int i = 0; i < npframe; ++i) {
            MPI_Barrier(comm);
            double start = MPI_Wtime();
            if (params->lb_interval > 0 && ((i+(frame-1)*npframe) % params->lb_interval) == 0) {
                zoltan_fn_init(load_balancer, mesh_data);
                rc = Zoltan_LB_Partition(load_balancer,      /* input (all remaining fields are output) */
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

                if(changes) for(int part = 0; part < nproc; ++part){
                    Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
                    auto domain = partitioning::geometric::borders_to_domain<2>(xmin, ymin, zmin, xmax, ymax, zmax, params->simsize);
                    domain_boundaries[part] = domain;
                }

                Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids,
                                    &importProcs, &importToPart);
                Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids,
                                    &exportProcs, &exportToPart);
            }
            load_balancing::geometric::migrate_particles<2>(mesh_data->els, domain_boundaries, datatype, comm);

            remote_el = load_balancing::geometric::exchange_data<2>(mesh_data->els, domain_boundaries, datatype, comm);
            switch (params->computation_method) {
                case 1:
                    lennard_jones::compute_forces(mesh_data->els, remote_el, params);
                    break;
                case 2:
                case 3:
                    lennard_jones::create_cell_linkedlist(M, lsub, mesh_data->els, remote_el, plklist);
                    lennard_jones::compute_forces(M, lsub, mesh_data->els, remote_el, plklist, params);
                    break;
            }

            leapfrog2(dt, mesh_data->els);
            leapfrog1(dt, mesh_data->els);
            apply_reflect(mesh_data->els, params->simsize);
            double diff = (MPI_Wtime() - start) / 1e-3; //divide time by tick resolution

            std::vector<double> times(nproc);
            MPI_Gather(&diff, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, 0, comm);
            write_report_data(lb_file, i+(frame-1)*npframe, times, rank);

            mesh_data->els = mesh_data->els;
        }
        load_balancing::gather_elements_on(nproc, rank, params->npart,
                                           mesh_data->els, 0, recv_buf, datatype.elements_datatype, comm);
        if (fp) {
            double end = MPI_Wtime();
            double time_spent = (end - begin);
            write_frame_data(fp, params->npart, &recv_buf[0]);
            printf("Frame [%d] completed in %f seconds\n", frame, time_spent);
            begin = MPI_Wtime();
        }
    }
    if(rank==0){
        double diff =(MPI_Wtime() - start_sim);
        lb_file << diff << std::endl;
        lb_file.close();
    }
}

#endif //NBMPI_BOXRUNNER_HPP