//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_LJPOTENTIAL_HPP
#define NBMPI_LJPOTENTIAL_HPP

#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>

#include "params.hpp"
#include "physics.hpp"
#include "utils.hpp"
#include "zoltan_fn.hpp"

namespace algorithm {

    constexpr Integer EMPTY = -1;

    template<int N>
    void CLL_init(const elements::Element<N> *local_elements, Integer local_n_elements,
                  const elements::Element<N> *remote_elements, Integer remote_n_elements,
                  Integer lc[N], Real rc,
                  Integer *head,
                  Integer *lscl) {
        Integer lcxyz = lc[0] * lc[1];
        if constexpr (N == 3) {
            lcxyz *= lc[2];
        }
        Integer c;
        for (size_t i = 0; i < lcxyz; ++i) head[i] = EMPTY;

        for (size_t i = 0; i < local_n_elements; ++i) {
            c = position_to_cell<N>(local_elements[i].position, rc, lc[0], lc[1]);
            lscl[i] = head[c];
            head[c] = i;
        }

        for (size_t i = 0; i < remote_n_elements; ++i) {
            c = position_to_cell<N>(remote_elements[i].position, rc, lc[0], lc[1]);
            lscl[i+local_n_elements] = head[c];
            head[c] = i+local_n_elements;
        }
    }

    template<int N>
    void CLL_init(const elements::Element<N> *local_elements, Integer local_n_elements,
                  const elements::Element<N> *remote_elements, Integer remote_n_elements,
                  const BoundingBox<N>& bbox, Real rc,
                  Integer *head,
                  Integer *lscl) {
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        Integer lcxyz = std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto prev, auto v){ return prev*v; }),
                c;
        for (size_t i = 0; i < lcxyz; ++i) head[i] = EMPTY;

        for (size_t i = 0; i < local_n_elements; ++i) {
            c = position_to_local_cell_index<N>(local_elements[i].position, rc, bbox, lc[0], lc[1]);
            lscl[i] = head[c];
            head[c] = i;
        }

        for (size_t i = 0; i < remote_n_elements; ++i) {
            c = position_to_local_cell_index<N>(remote_elements[i].position, rc, bbox, lc[0], lc[1]);
            lscl[i+local_n_elements] = head[c];
            head[c] = i+local_n_elements;
        }
    }
    template<int N>
    void CLL_update(std::initializer_list<std::pair<elements::Element<N>*, size_t>>&& elements,
                    const BoundingBox<N>& bbox, Real rc,
                    Integer *head, Integer *lscl) {
        auto lc = get_cell_number_by_dimension<N>(bbox, rc);
        Integer c, acc = 0;
        for(const auto& span : elements){
            auto el_ptr = span.first;
            auto n_els  = span.second;
            for (size_t i = 0; i < n_els; ++i) {
                c = position_to_local_cell_index<N>(el_ptr[i].position, rc, bbox, lc[0], lc[1]);
                lscl[i+acc] = head[c];
                head[c] = i+acc;
            }
            acc += n_els;
        }
    }
    template<int N>
    void CLL_init(std::initializer_list<std::pair<elements::Element<N>*, size_t>>&& elements,
                  const BoundingBox<N>& bbox, Real rc,
                  Integer *head,
                  Integer *lscl) {
        Integer lcxyz = get_total_cell_number<N>(bbox, rc);
        for (size_t i = 0; i < lcxyz; ++i) head[i] = EMPTY;
        CLL_update(std::move(elements), bbox, rc, head, lscl);
    }

    template<int N>
    inline void CLL_append(const Integer i, const Integer  cell, const elements::Element<N>& element, std::vector<Integer>* head, std::vector<Integer>* lscl) {
        lscl->at(i) = head->at(cell);
        head->at(cell) = i;
    }

    template<int N>
    inline void CLL_append(const Integer i, const elements::Element<N>& element, const BoundingBox<N>& bbox, const std::array<Integer,N>& lc, const Real rc, std::vector<Integer>* head, std::vector<Integer>* lscl) {
        auto c = position_to_local_cell_index<N>(element.position, rc, bbox, lc[0], lc[1]);
        lscl->at(i) = head->at(c);
        head->at(c) = i;
    }

    template<int N>
    inline void CLL_append_all(const std::vector<Integer>& ids, const std::vector<elements::Element<N>>& elements, const BoundingBox<N>& bbox, const std::array<Integer,N>& lc, const Real rc, std::vector<Integer>* head, std::vector<Integer>* lscl) {
        for(auto i : ids) CLL_append(i, elements.at(i), bbox, lc, rc, head, lscl);
    }

    Integer CLL_compute_forces3d(std::vector<Real>* acc,
                                const elements::Element<3> *elements, Integer n_elements,
                                const elements::Element<3> *remote_elements,
                                const BoundingBox<3>& bbox, Real rc,
                                const Integer *head, const Integer *lscl,
                                const sim_param_t* params) {
        auto lc = get_cell_number_by_dimension<3>(bbox, rc);
        std::array<Real, 3> delta_dim;
        Real delta;
        Real sig2 = params->sig_lj*params->sig_lj;
        Integer c, c1, ic[3], ic1[3], j;
        elements::Element<3> source, receiver;
        Integer cmplx = n_elements;
        std::fill(acc->begin(), acc->begin()+n_elements, (Real) 0.0);
        for (size_t i = 0; i < n_elements; ++i) {
            c = position_to_local_cell_index<3>(elements[i].position, rc, bbox, lc[0], lc[1]);
            receiver = elements[i];
            for (auto d = 0; d < 3; ++d)
                ic[d] = c / lc[d];
            for (ic1[0] = ic[0] - 1; ic1[0] < (ic[0]+1); ic1[0]++) {
                for (ic1[1] = ic[1] - 1; ic1[1] < ic[1] + 1; ic1[1]++) {
                    for (ic1[2] = ic[2] - 1; ic1[2] < ic[2] + 1; ic1[2]++) {
                        /* this is for bounce back, to avoid heap-buffer over/under flow */
                        if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) || (ic1[2] < 0 || ic1[2] >= lc[2])) continue;
                        c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                        j = head[c1];
                        while(j != EMPTY) {
                            if(i < j) {
                                delta = 0.0;
                                source = j < n_elements ? elements[j] : remote_elements[j - n_elements];
                                for (int dim = 0; dim < 3; ++dim)
                                    delta_dim[dim] = receiver.position.at(dim) - source.position.at(dim);
                                for (int dim = 0; dim < 3; ++dim)
                                    delta += (delta_dim[dim] * delta_dim[dim]);
                                Real C_LJ = compute_LJ_scalar<Real>(delta, params->eps_lj, sig2);
                                for (int dim = 0; dim < 3; ++dim) {
                                    acc->at(3*i + dim) += (C_LJ * delta_dim[dim]);
                                }
                                cmplx++;
                            }
                            j = lscl[j];
                        }
                    }
                }
            }
        }
        return cmplx;
    }

    void CLL_compute_forces2d(elements::Element<2> *elements, Integer n_elements, Integer lc[2], Real rc,
                              Integer *head, Integer *lscl, sim_param_t* params) {
        std::stringstream str; str << __func__ << " is not implemented ("<<__FILE__<<":"<<__LINE__ <<")"<< std::endl;
        throw std::runtime_error(str.str());
    }

    void CLL_compute_forces2d(elements::Element<2> *elements, Integer n_elements, const BoundingBox<2>& bbox, Real rc,
                              Integer *head, Integer *lscl, sim_param_t* params) {
        std::stringstream str; str << __func__ << " is not implemented ("<<__FILE__<<":"<<__LINE__ <<")"<< std::endl;
        throw std::runtime_error(str.str());
    }

    template<int N>
    Integer CLL_compute_forces(elements::Element<N> *elements, Integer n_elements,
                            const elements::Element<N> *remote_elements,
                            const BoundingBox<N>& bbox, Real rc,
                            const Integer *head, const Integer *lscl,
                            const sim_param_t* params) {
        if constexpr(N==3) {
            return CLL_compute_forces3d(elements, n_elements, remote_elements, bbox, rc, head, lscl, params);
        }else {
            CLL_compute_forces2d(elements,n_elements, bbox, rc, head, lscl, params);
            return 0;
        }
    }

    template<int N>
    Integer CLL_compute_forces(std::vector<Real>* acc, const elements::Element<N> *elements, Integer n_elements,
                               const elements::Element<N> *remote_elements,
                               const BoundingBox<N>& bbox, Real rc,
                               const Integer *head, const Integer *lscl,
                               const sim_param_t* params) {
        if constexpr(N==3) {
            return CLL_compute_forces3d(acc, elements, n_elements, remote_elements, bbox, rc, head, lscl, params);
        }else {
            CLL_compute_forces2d(elements, n_elements, bbox, rc, head, lscl, params);
            return 0;
        }
    }

    template<int N>
    void CLL_compute_forces(elements::Element<N> *elements, Integer n_elements,
                            const elements::Element<N> *remote_elements, Integer remote_n_elements,
                            const Integer lc[N], Real rc,
                            const Integer *head, const Integer *lscl,
                            const sim_param_t* params) {
        if constexpr(N==3) {
            CLL_compute_forces3d(elements, n_elements, remote_elements, remote_n_elements, lc, rc, head, lscl, params);
        }else {
            CLL_compute_forces2d(elements,n_elements, lc, rc, head, lscl, params);
        }
    }
}

auto MPI_TIME       = MPI_DOUBLE;
auto MPI_COMPLEXITY = MPI_LONG_LONG;

namespace lj {
    namespace {
        std::vector<Real> acc;
    }
    template<int N>
    Complexity compute_one_step (
            std::vector<elements::Element<N>>&        elements,
            const std::vector<elements::Element<N>>& remote_el,
            std::vector<Integer> *head,                         //the particle linked list
            std::vector<Integer> *lscl,                         //the cell starting point
            BoundingBox<N>& bbox,                               //bounding box of particles
            const Borders& borders,                             //bordering cells and neighboring processors
            const sim_param_t *params) {                        //simulation parameters

        const Real cut_off_radius = params->rc; // cut_off
        const Real dt = params->dt;
        const size_t nb_elements = elements.size();

        if(const auto n_cells = get_total_cell_number<N>(bbox, params->rc); head->size() < n_cells) {
            std::cout << "resize head buffer" << n_cells << std::endl;
            head->resize(n_cells);
        }
        if(const auto n_force_elements = N*elements.size(); acc.size() < n_force_elements) {
            std::cout << "resize force acceleration buffer " << n_force_elements << std::endl;
            acc.resize(N*n_force_elements);
        }
        if(const auto n_particles = elements.size()+remote_el.size();  lscl->size() < n_particles) {
            std::cout << "resize cell linkedlist " << n_particles << std::endl;
            lscl->resize(n_particles);
        }
        algorithm::CLL_init<N>({ {elements.data(), nb_elements}, {elements.data(), remote_el.size()} }, bbox, cut_off_radius, head->data(), lscl->data());

        Complexity cmplx = algorithm::CLL_compute_forces<N>(&acc, elements.data(), nb_elements, remote_el.data(), bbox, cut_off_radius, head->data(), lscl->data(), params);

        leapfrog2<N>(dt, acc, elements);
        leapfrog1<N>(dt, cut_off_radius, acc, elements);
        apply_reflect<N>(elements, params->simsize);

        return cmplx;
    };
}
#endif //NBMPI_LJPOTENTIAL_HPP
