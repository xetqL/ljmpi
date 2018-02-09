//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_LJPOTENTIAL_HPP
#define NBMPI_LJPOTENTIAL_HPP

#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>

#include "physics.hpp"
#include "params.hpp"
namespace lennard_jones {
template<typename RealType>
void create_cell_linkedlist(
        const int nsub, /* number of subdomain per row*/
        const double lsub, /* width of subdomain */
        const int nparticle, /* number of particles */
        const RealType* x, /* particle location */
        int* particleslklist, /* particle linked list */
        int* head) throw() /* head of particle linked list */{
    int cell_of_particle;
    for (int icell = 0; icell < nsub * nsub; icell++) head[icell] = -1;
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        cell_of_particle = (int) (std::floor(x[2 * iparticle] / lsub)) + nsub * (std::floor(x[2 * iparticle + 1] / lsub));
        if (cell_of_particle >= (nsub * nsub)) throw std::runtime_error("Particle "+std::to_string(cell_of_particle) + " is out of domain");
        particleslklist[iparticle] = head[cell_of_particle];
        head[cell_of_particle] = iparticle;
    }
}
/*
template<int N>
void create_cell_linkedlist(
        const int nsub, // number of subdomain per row
        const double lsub, // width of subdomain
        const std::vector<elements::Element<N>> &particles, // particle location
        std::unordered_map<int, int> &particleslklist, // particle linked list
        std::vector<int> &head) throw() // head of particle linked list  {
    int cell_of_particle;
    for (int icell = 0; icell < nsub * nsub; icell++) head[icell] = -1;
    for (auto const &particle : particles) {
        cell_of_particle = (int) (std::floor(particle.position.at(0) / lsub)) + nsub * (std::floor(particle.position.at(1) / lsub));
        if (cell_of_particle >= (nsub * nsub) || cell_of_particle < 0)
            throw std::runtime_error("Particle "+std::to_string(cell_of_particle) + " is out of domain");
        particleslklist[particle.identifier] = head[cell_of_particle];

        head[cell_of_particle] = particle.identifier;
    }
}
*/
template <int N>
void create_cell_linkedlist(
        const int nsub, /* number of subdomain per row*/
        const double lsub, /* width of subdomain */
        const std::vector<elements::Element<N>> &local_elements, /* particle location */
        const std::vector<elements::Element<N>> &remote_elements, /* particle location */
        std::map<int, std::unique_ptr<std::vector<elements::Element<N>>>> &plist) throw() {
    int cell_of_particle;
    //for (int icell = 0; icell < nsub * nsub; icell++) { plist[icell]->clear(); }
    plist.clear();
    size_t local_size = local_elements.size(),
            remote_size = remote_elements.size();
    for (size_t cpt = 0; cpt < local_size + remote_size; ++cpt) {
        auto const& particle = cpt >= local_size ? remote_elements[cpt-local_size] : local_elements[cpt];
        cell_of_particle = (int) (std::floor(particle.position.at(0) / lsub)) + nsub * (std::floor(particle.position.at(1) / lsub));

        if (cell_of_particle >= (nsub * nsub) || cell_of_particle < 0)
            throw std::runtime_error("Particle "+std::to_string(cell_of_particle) + " is out of domain");
        if(plist.find(cell_of_particle) != plist.end())
            plist[cell_of_particle]->push_back(particle);
        else{
            plist[cell_of_particle]= std::make_unique<std::vector<elements::Element<N>>>();
            plist[cell_of_particle]->push_back(particle);
        }
    }
}

template<int N>
void compute_forces(
        const int M, /* Number of subcell in a row  */
        const float lsub, /* length of a cell */
        std::vector<elements::Element<N>> &local_elements,
        const std::vector<elements::Element<N>> &remote_elements,
        const std::map<int, std::unique_ptr<std::vector<elements::Element<N>>>> &plist,
        const sim_param_t* params) /* Simulation parameters */ {

    double g    = (double) params->G;
    double eps  = (double) params->eps_lj;
    double sig  = (double) params->sig_lj;
    double sig2 = sig*sig;

    // each particle MUST checks the local particles and the particles from neighboring PE
    std::unordered_map<int, elements::Element<N>> element_map;

    for(auto &el : local_elements){
        std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
        el.acceleration.at(1) = -g;
    }

    int linearcellidx;
    int nlinearcellidx;

    // process only the particle we are interested in
    for (auto &force_recepter : local_elements) { //O(n)

        // find cell from particle position
        linearcellidx = (int) (std::floor(force_recepter.position.at(0) / lsub)) + M * (std::floor(force_recepter.position.at(1) / lsub));
        // convert linear index to grid position
        int xcellidx = linearcellidx % M;
        int ycellidx = linearcellidx / M;
        /* Explore neighboring cells */
        for (int neighborx = -1; neighborx < 2; neighborx++) {
            for (int neighbory = -1; neighbory < 2; neighbory++) {

                /* Check boundary conditions */
                if (xcellidx + neighborx < 0 || xcellidx + neighborx >= M) continue;
                if (ycellidx + neighbory < 0 || ycellidx + neighbory >= M) continue;

                nlinearcellidx = (xcellidx + neighborx) + M * (ycellidx + neighbory);
                if(plist.find(nlinearcellidx) != plist.end()){
                    auto el_list = plist.at(nlinearcellidx).get();
                    size_t cell_el_size = el_list->size();
                    for(size_t el_idx = 0; el_idx < cell_el_size; ++el_idx){
                        auto const& force_source = el_list->at(el_idx);
                        if (force_recepter.identifier != force_source.identifier) {

                            double dx = force_source.position.at(0) - force_recepter.position.at(0);
                            double dy = force_source.position.at(1) - force_recepter.position.at(1);

                            double C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);

                            force_recepter.acceleration[0] += (C_LJ * dx);
                            force_recepter.acceleration[1] += (C_LJ * dy);
                        }
                    }
                }
            }
        }
    }
}

/*
template<int N>
void create_cell_linkedlist(
        const int nsub, // number of subdomain per row
        const double lsub, // width of subdomain
        const std::vector<elements::Element<N>> &local_elements, // particle location
        const std::vector<elements::Element<N>> &remote_elements, // particle location
        std::unordered_map<int, int> &particleslklist, // particle linked list
        std::vector<int> &head) throw() {// head of particle linked list {
    int cell_of_particle;
    for (int icell = 0; icell < nsub * nsub; icell++) head[icell] = -1;
    size_t local_size = local_elements.size(),
           remote_size = remote_elements.size();
    for (size_t cpt = 0; cpt < local_size + remote_size; ++cpt) {
        auto const& particle = cpt >= local_size ? remote_elements[cpt-local_size] : local_elements[cpt];

        cell_of_particle = (int) (std::floor(particle.position.at(0) / lsub)) + nsub * (std::floor(particle.position.at(1) / lsub));

        if (cell_of_particle >= (nsub * nsub) || cell_of_particle < 0) throw std::runtime_error("Particle "+std::to_string(cell_of_particle) + " is out of domain");

        particleslklist[particle.identifier] = head[cell_of_particle];
        head[cell_of_particle] = particle.identifier;
    }
}
*/

template<typename RealType>
void compute_forces(
        const int n, /* Number of particle */
        const int M, /* Number of subcell in a row  */
        const float lsub, /* length of a cell */
        const std::vector<RealType> x, /* Particles position */
        const int istart,
        const int iend,
        const std::vector<RealType> &xlocal,
        std::vector<RealType>& Flocal,
        const std::vector<int> &head, /* Starting neighboring particles */
        const std::vector<int> &plklist, /* Neighboring subcell particles organized as a linkedlist */
        const sim_param_t* params) /* Simulation parameters */ {

    RealType g    = (RealType) params->G;
    RealType eps  = (RealType) params->eps_lj;
    RealType sig  = (RealType) params->sig_lj;
    RealType sig2 = (RealType) sig*sig;

    /* Global force downward (e.g. gravity) */
    for (int i = 0; i < (iend - istart); ++i) {
        Flocal[2 * i + 0] = 0;
        Flocal[2 * i + 1] = -g;
    }

    int linearcellidx;
    int nlinearcellidx;
    int nparticleidx;

    // process only the particle we are interested in
    for (int particleidx = istart; particleidx < iend; ++particleidx) {

        int ii = particleidx - istart;

        // find cell from particle position
        linearcellidx = (int) (std::floor(x[2 * particleidx] / lsub)) + M * (std::floor(x[2 * particleidx + 1] / lsub));

        // convert linear index to grid position
        int xcellidx = linearcellidx % M;
        int ycellidx = linearcellidx / M;

        /* Explore neighboring cells */
        for (int neighborx = -1; neighborx < 2; neighborx++) {
            for (int neighbory = -1; neighbory < 2; neighbory++) {

                /* Check boundary conditions */
                if (xcellidx + neighborx < 0 || xcellidx + neighborx >= M) continue;
                if (ycellidx + neighbory < 0 || ycellidx + neighbory >= M) continue;

                nlinearcellidx = (xcellidx + neighborx) + M * (ycellidx + neighbory);

                nparticleidx = head[nlinearcellidx];
                while (nparticleidx != -1) {
                    if (particleidx != nparticleidx) {

                        RealType dx = x[2 * nparticleidx + 0] - x[2 * particleidx + 0];
                        RealType dy = x[2 * nparticleidx + 1] - x[2 * particleidx + 1];

                        RealType C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);

                        Flocal[2 * ii + 0] += (C_LJ * dx);
                        Flocal[2 * ii + 1] += (C_LJ * dy);
                    }
                    nparticleidx = plklist[nparticleidx];
                }
            }
        }
    }
}

template<typename RealType>
void compute_forces(
        const int n,
        const std::vector<RealType> &x,
        const int istart,
        const int iend,
        const std::vector<RealType> &xlocal,
        std::vector<RealType>& Flocal,
        const sim_param_t* params) {

    int nlocal = iend - istart;
    RealType g    = (RealType) params->G;
    RealType eps  = (RealType) params->eps_lj;
    RealType sig  = (RealType) params->sig_lj;
    RealType sig2 = sig*sig;

    /* Global force downward (e.g. gravity) */
    for (int i = 0; i < nlocal; ++i) {
        Flocal[2 * i + 0] = 0;
        Flocal[2 * i + 1] = -g;
    }

    /* Particle-particle interactions (Lennard-Jones) */
    for (int i = istart; i < iend; ++i) {
        int ii = i - istart;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                RealType dx = x[2 * j + 0] - xlocal[2 * ii + 0];
                RealType dy = x[2 * j + 1] - xlocal[2 * ii + 1];
                RealType C_LJ = compute_LJ_scalar(dx*dx + dy*dy, eps, sig2);
                Flocal[2 * ii + 0] += (C_LJ * dx);
                Flocal[2 * ii + 1] += (C_LJ * dy);
            }
        }
    }
}

template<int N>
void compute_forces(
        std::vector<elements::Element<N>> &local_elements,
        const std::vector<elements::Element<N>> &remote_elements,
        const sim_param_t* params) {

    double g    = (double) params->G;
    double eps  = (double) params->eps_lj;
    double sig  = (double) params->sig_lj;
    double sig2 = sig*sig;

    for(auto &el : local_elements){
        std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
        el.acceleration.at(1) = -g;//set GRAVITY MOFO YO
    }

    /* Particle-particle interactions (Lennard-Jones) */
    size_t local_size = local_elements.size(), remote_size = remote_elements.size();
    for(auto &force_recepter : local_elements){
        for(size_t cpt = 0; cpt < local_size + remote_size; ++cpt){
            auto force_source = cpt >= local_size ? remote_elements[cpt-local_size] : local_elements[cpt];
            if(force_source.identifier != force_recepter.identifier){
                double dx = force_source.position.at(0) - force_recepter.position.at(0);
                double dy = force_source.position.at(1) - force_recepter.position.at(1);
                double C_LJ = compute_LJ_scalar(dx*dx + dy*dy, eps, sig2);
                force_recepter.acceleration[0] += (C_LJ * dx);
                force_recepter.acceleration[1] += (C_LJ * dy);
            }
        }
    }
}

/*template<int N>
void compute_forces(
        const int M, // Number of subcell in a row
        const float lsub, // length of a cell
        std::vector<elements::Element<N>> &local_elements,
        const std::vector<elements::Element<N>> &remote_elements,
        const std::vector<int> &head, // Starting neighboring particles
        const std::unordered_map<int, int> &plklist, // Neighboring subcell particles organized as a linkedlist
        const sim_param_t* params) // Simulation parameters  {

    double g    = (double) params->G;
    double eps  = (double) params->eps_lj;
    double sig  = (double) params->sig_lj;
    double sig2 = sig*sig;

    // each particle MUST checks the local particles and the particles from neighboring PE
    std::unordered_map<int, elements::Element<N>> element_map;

    for(auto &el : remote_elements){
        element_map[el.identifier]= el;
    }
    for(auto &el : local_elements){
        element_map[el.identifier]= el;
    }
    for(auto &el : local_elements){
        std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
        el.acceleration.at(1) = -g;
    }

    int linearcellidx;
    int nlinearcellidx;
    int nparticleidx;

    // process only the particle we are interested in
    for (auto &force_recepter : local_elements) {

        // find cell from particle position
        linearcellidx = (int) (std::floor(force_recepter.position.at(0) / lsub)) + M * (std::floor(force_recepter.position.at(1) / lsub));

        // convert linear index to grid position
        int xcellidx = linearcellidx % M;
        int ycellidx = linearcellidx / M;

        // Explore neighboring cells
        for (int neighborx = -1; neighborx < 2; neighborx++) {
            for (int neighbory = -1; neighbory < 2; neighbory++) {

                // Check boundary conditions
                if (xcellidx + neighborx < 0 || xcellidx + neighborx >= M) continue;
                if (ycellidx + neighbory < 0 || ycellidx + neighbory >= M) continue;

                nlinearcellidx = (xcellidx + neighborx) + M * (ycellidx + neighbory);

                nparticleidx = head[nlinearcellidx];
                while (nparticleidx != -1) {

                    if (force_recepter.identifier != nparticleidx) {
                        double dx = force_recepter.position.at(0) - element_map[nparticleidx].position.at(0);
                        double dy = force_recepter.position.at(1) - element_map[nparticleidx].position.at(1);

                        double C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);

                        force_recepter.acceleration[0] += (C_LJ * dx);
                        force_recepter.acceleration[1] += (C_LJ * dy);
                    }
                    nparticleidx = plklist.at(nparticleidx);
                }
            }
        }
    }
}
*/
}
#endif //NBMPI_LJPOTENTIAL_HPP
