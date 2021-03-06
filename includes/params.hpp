//
// Created by xetql on 2/5/18.
//

#ifndef PARAMS_H
#define PARAMS_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#include "zupply.hpp"

/*@T
 * \section{System parameters}
 *
 * The [[sim_param_t]] structure holds the parameters that
 * describe the simulation.  These parameters are filled in
 * by the [[get_params]] function (described later).
 *@c*/
struct sim_param_t {
    std::string fname;   /* File name (run.out)        */
    int   npart;   /* Number of particles (500)  */
    int   nframes; /* Number of frames (200)     */
    int   npframe; /* Steps per frame (100)      */
    int   world_size; /* wsize     */
    float dt;      /* Time step (1e-4)           */
    float eps_lj;  /* Strength for L-J (1)       */
    float sig_lj;  /* Radius for L-J   (1e-2)    */
    float G;       /* Gravitational strength (1) */
    float T0;      /* Initial temperature (1)    */
    float simsize; /* Borders of the simulation  */
    float rc;
    bool  record;  /* record the simulation in a binary file */
    int   seed;    /* seed used in the RNG */
    int   particle_init_conf = 1;
    int   id = 0;
    int nb_best_path;
    std::string uuid;
    int verbosity;
};

void print_params(std::ostream& stream, const sim_param_t& params){
    stream << "==============================================" << std::endl;
    stream << "= Parameters: " << std::endl;
    stream << "= Particles: " << params.npart << std::endl;
    stream << "= Seed: " << params.seed << std::endl;
    stream << "= id: " << params.id << std::endl;
    stream << "= PEs: " << params.world_size << std::endl;
    stream << "= Simulation size: " << params.simsize << std::endl;
    stream << "= Number of time-steps: " << params.nframes << "x" << params.npframe << std::endl;
    stream << "= Initial conditions: " << std::endl;
    stream << "= SIG:" << params.sig_lj << std::endl;
    stream << "= EPS:  " << params.eps_lj << std::endl;
    stream << "= Borders: collisions " << std::endl;
    stream << "= Gravity:  " << params.G << std::endl;
    stream << "= Temperature: " << params.T0 << std::endl;
    stream << "==============================================" << std::endl;
}
void print_params(const sim_param_t& params) {
    print_params(std::cout, params);
}
std::optional<sim_param_t> get_params(int argc, char** argv){
    sim_param_t params;

    zz::cfg::ArgParser parser;
    parser.add_opt_version('V', "version", "MiniLB v1.0:\nMiniLB is a fast parallel (MPI) n-body mini code for load balancing brenchmarking.");
    parser.add_opt_help('h', "help"); // use -h or --help

    parser.add_opt_value('B', "best", params.nb_best_path, 1, "Number of Best path to retrieve (A*)", "INT");
    parser.add_opt_value('d', "distribution", params.particle_init_conf, 1, "Initial particle distribution 1: Uniform, 2:Half, 3:Wall, 4: Cluster", "INT");
    parser.add_opt_value('e', "epslj", params.eps_lj, 1.0f, "Epsilon (lennard-jones)", "FLOAT");
    parser.add_opt_value('f', "npframe", params.npframe, 100, "steps per frame", "INT").require();
    parser.add_opt_value('F', "nframes", params.nframes, 100, "number of frames", "INT").require();
    parser.add_opt_value('g', "gravitation", params.G, 1.0f, "Gravitational strength", "FLOAT");
    parser.add_opt_value('i', "id", params.id, 0, "Simulation id", "INT").require();
    parser.add_opt_value('l', "lattice", params.rc, 3.5f*1e-2f, "Lattice size", "FLOAT");
    parser.add_opt_value('n', "nparticles", params.npart, 500, "Number of particles", "INT").require();
     parser.add_opt_flag('r', "record", "Record the simulation", &params.record);
    parser.add_opt_value('s', "siglj", params.sig_lj, 1e-2f, "Sigma (lennard-jones)", "FLOAT");
    parser.add_opt_value('S', "seed", params.seed, rand(), "Random seed", "INT").require();
    parser.add_opt_value('t', "dt", params.dt, 1e-4f, "Time step", "float");
    parser.add_opt_value('T', "temperature", params.T0, 1.0f, "Initial temperatore", "float");
    parser.add_opt_value('w', "width", params.simsize, 1.0f, "Simulation box width", "FLOAT");

    bool output;
    auto &verbose = parser.add_opt_flag('v', "verbose", "Set verbosity", &output);
    parser.parse(argc, argv);

    if (parser.count_error() > 0) {
        std::cout << parser.get_error() << std::endl;
        std::cout << parser.get_help() << std::endl;
        return std::nullopt;
    }

    params.verbosity = verbose.get_count();
    return params;
}

/*@q*/
#endif /* PARAMS_H */

