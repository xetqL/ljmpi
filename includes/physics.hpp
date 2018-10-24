//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_PHYSICS_HPP
#define NBMPI_PHYSICS_HPP

#include <limits>
#include "spatial_elements.hpp"

template <int N>
void init_particles_random_v(std::vector<elements::Element<N>> &elements, sim_param_t* params, int seed = 0) noexcept {
    float T0 = params->T0;
    int n = elements.size();
    //std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        double R = T0 * std::sqrt(-2 * std::log(udist(gen)));
        double T = 2 * M_PI * udist(gen);
        elements[i].velocity[0] = (double) (R * std::cos(T));
        elements[i].velocity[1] = (double) (R * std::sin(T));
    }
}

template<typename RealType>
RealType compute_LJ_scalar(RealType r2, RealType eps, RealType sig2) {
    if (r2 < 6.25 * sig2) { /* r_cutoff = 2.5 *sigma */
        RealType z = sig2 / r2;
        RealType u = z * z*z;
        return 24 * eps / r2 * u * (1 - 2 * u);
    }
    return 0;
}

template<typename RealType>
void __leapfrog1(int n, RealType dt, RealType* x, RealType* v, RealType* a) {
    for (int i = 0; i < n; ++i, x += 2, v += 2, a += 2) {
        v[0] += a[0] * dt / 2;
        v[1] += a[1] * dt / 2;
        x[0] += v[0] * dt;
        x[1] += v[1] * dt;
    }
}

template<int N>
void leapfrog1(const double dt, std::vector<elements::Element<N>> &elements, double cut_off, double ff = 1.0) {
    for(auto &el : elements){
        for(size_t dim = 0; dim < N; ++dim) {
            el.velocity[dim] += el.acceleration.at(dim) * dt / 2;
            /**
             * This is a mega-giga hotfix.
             * Let say that particles are so close that they produce so much force on them such that the timestep
             * is too big to prevent them to cross the min radius. If a particle cross the min radius of another one
             * it creates an almost infinity repulsive force that breaks everything. In real life, this should not
             * happen because life is supposed to be continuous (planck distance)*/
            if(std::abs(el.velocity.at(dim) * dt) >= cut_off ) {
                el.velocity[dim] = (0.9 * cut_off / dt) * ff; //max speed is 90% of cutoff per timestep
            }
            el.position[dim] += el.velocity.at(dim) * dt;

        }
    }
}

template<typename RealType>
void leapfrog2(int n, RealType dt, RealType* v, RealType* a) {
    for (int i = 0; i < n; ++i, v += 2, a += 2) {
        v[0] += a[0] * dt / 2;
        v[1] += a[1] * dt / 2;
    }
}

template<int N>
void leapfrog2(const double dt, std::vector<elements::Element<N>> &elements, double ff = 1.0) {
    for(auto &el : elements){
        for(size_t dim = 0; dim < N; ++dim){
            el.velocity[dim] += ff * el.acceleration.at(dim) * dt / 2;
        }
    }
}

/**
 * Reflection at the boundary
 * @param wall
 * @param x
 * @param v
 * @param a
 */
template<typename RealType>
static void reflect(RealType wall, RealType* x, RealType* v, RealType* a) {
    *x = 2 * wall - (*x);
    *v = -(*v);
    *a = -(*a);
}

/**
 * Apply the reflection on all the particles at the border of the simulation
 * @param n Number of particles
 * @param x Position of particles
 * @param v Velocity of particles
 * @param a Acceleration of particles
 * @param borders Border of the simulation (XMIN, XMAX, YMIN, YMAX)
 */
template<typename RealType>
void apply_reflect(unsigned int n, RealType* x, RealType* v, RealType* a, RealType simsize) {
    unsigned int i = 0;

    while(i < n) {

        if (x[0] < 0.0) reflect((RealType) 0.0, x + 0, v + 0, a + 0);

        if (x[0] >= simsize) reflect(simsize-std::numeric_limits<RealType>::epsilon(), x + 0, v + 0, a + 0);

        if (x[1] < 0.0) reflect((RealType) 0.0, x + 1, v + 1, a + 1);

        if (x[1] >= simsize) reflect(simsize-std::numeric_limits<RealType>::epsilon(), x + 1, v + 1, a + 1);

        i++; x+=2; v+=2; a+=2;
    }
}

template<int N>
void apply_reflect(std::vector<elements::Element<N>> &elements, const elements::ElementRealType simsize) {
    //unsigned int i = 0, repeat=0;
    //const unsigned int MAX_REPEAT = 4;
    for(auto &element: elements){
        size_t dim = 0;
        //repeat = 0;
        while(dim < N){
            if(element.position.at(dim) < 0.0)
                reflect<elements::ElementRealType>(0.0, &element.position[dim], &element.velocity[dim], &element.acceleration[dim]);

            if(element.position.at(dim) >= simsize)
                reflect<elements::ElementRealType>(simsize-std::numeric_limits<elements::ElementRealType>::epsilon(), &element.position[dim], &element.velocity[dim], &element.acceleration[dim]);

            dim++;
        }
    }
}
#endif //NBMPI_PHYSICS_HPP
