//
// Created by xetql on 21.03.18.
//

#ifndef NBMPI_METRICS_H
#define NBMPI_METRICS_H

#include <utility>
#include <vector>
#include <list>
#include <cmath>
#include <limits>
#include <deque>
#include <numeric>
#include <iostream>

#include <gsl/gsl_histogram.h>
#include <gsl/gsl_statistics.h>
#include <mpi.h>
#include "report.hpp"
#include "utils.hpp"

#ifndef DELTA_LB_CALL
#define DELTA_LB_CALL 100
#endif

#ifndef DATASET_SCIENTIFIC_PRECISION
#define DATASET_SCIENTIFIC_PRECISION 8
#endif

template<class T>
struct SlidingWindow {
    std::deque<T> data_container;
    size_t window_max_size;

    SlidingWindow(size_t window_max_size) : window_max_size(window_max_size) {};

    inline void add(const T &data) {
        if (data_container.size() < window_max_size)
            data_container.push_back(data); // add while not full
        else {                              // when full
            data_container.pop_front();     // delete oldest data
            data_container.push_back(data); // push new data
        }
    }
};

namespace metric {

template<class RealType>
struct LBMetrics{
    /*
     * return {
            gini_times, gini_complexities, gini_communications, // LB for times, complexity, and communications
            *std::max_element(times.begin(), times.end()),
            //(RealType) gsl_stats_variance(&window_gini_times->data_container.front(), 1, window_gini_times->data_container.size()),
            //(RealType) gsl_stats_variance(&window_gini_complexities->data_container.front(), 1, window_gini_times->data_container.size()),
            //(RealType) gsl_stats_variance(&window_times->data_container.front(), 1, window_gini_times->data_container.size()),
            //(RealType) gsl_stats_variance(&window_gini_communications->data_container.front(), 1, window_gini_times->data_container.size()),
            (RealType) variance<RealType>(mu_interaction_times),
            //slope_gini_times, slope_gini_complexity, slope_times, slope_gini_communications,
            macd_gini_times, macd_gini_complexity, macd_times, macd_gini_communications
    };
     * */
    const std::vector<RealType> metrics;

    LBMetrics(const std::vector<RealType> metrics) : metrics(metrics){};

    RealType get_gini_times(){
        return metrics.at(0);
    }
    RealType get_gini_complexities(){
        return metrics.at(1);
    }
    RealType get_gini_communications(){
        return metrics.at(2);
    }
    RealType get_last_time_per_iteration(){
        return metrics.at(3);
    }
    RealType get_variance_avg_time_per_iteration(){
        return metrics.at(4);
    }
    RealType get_macd_gini_times(){
        return metrics.at(5);
    }
    RealType get_macd_gini_complexities(){
        return metrics.at(6);
    }
    RealType get_macd_gini_communications(){
        return metrics.at(8);
    }
    RealType get_macd_times(){
        return metrics.at(7);
    }

};

class FeatureContainer {
public:
    virtual std::vector<double> get_features() = 0;
    virtual int get_target() = 0;
};

namespace topology {

/**
* Compute the weighted load histogram of a particle PE.
* This histogram tells whether the particle distribution is heterogeneous.
* In particular, it gives an idea on the complexity a particle can expect in a given cell from the cell-linked-list
* algorithm.
* Moreover, the weighted load tells if the complexity impacts few or many particles. Of course, one prefer
* impacting few particles by an high complexity than the reverse case.
* If the histogram has a positive skewness the distribution is heterogeneous, because a lot of cell have a small
* complexity and only few cells have a high complexity.
* @tparam N Particle Dimension
* @tparam MapType Particle Linked List structure type
* @param number_of_cell_per_row Number of CLL per row
* @param npart Number of particle attribute to the PE
* @param plist The CLL structure
*/
template<class FloatingPointPrecision, int N, class MapType>
std::vector<FloatingPointPrecision> compute_cells_loads(int number_of_cell_per_row, int npart, MapType &plist) {
    int number_of_cell_per_col = number_of_cell_per_row,
            total_cells = number_of_cell_per_col * number_of_cell_per_row;

    //FloatingPointPrecision avg = (FloatingPointPrecision) npart / (number_of_cell_per_row * number_of_cell_per_col);
    int diff = total_cells - plist.size();
    int xcellidx;
    int ycellidx;
    int zcellidx;
    std::vector<FloatingPointPrecision> loads(diff, 0.0);
    for (auto const &cell : plist) {
        int cellidx = cell.first;
        linear_to_grid(cellidx, number_of_cell_per_row, number_of_cell_per_row, xcellidx, ycellidx, zcellidx);
        int particle_complexity = 0;
        FloatingPointPrecision weight = plist[cellidx].get()->size(); // the weight of this cell

        // Explore neighboring cells
        constexpr int zstart = N == 3 ? -1 : 0; // if in 2D, there is only 1 depth
        constexpr int zstop = N == 3 ? 2 : 1; // so, iterate only through [0,1)
        for (int neighborx = -1; neighborx < 2; neighborx++) {
            for (int neighbory = -1; neighbory < 2; neighbory++) {
                for (int neighborz = zstart; neighborz < zstop; neighborz++) {
                    // Check boundary conditions
                    if (xcellidx + neighborx < 0 || xcellidx + neighborx >= number_of_cell_per_row) continue;
                    if (ycellidx + neighbory < 0 || ycellidx + neighbory >= number_of_cell_per_row) continue;
                    if (zcellidx + neighborz < 0 || zcellidx + neighborz >= number_of_cell_per_row) continue;
                    int ncellidx = (xcellidx + neighborx) +
                                   number_of_cell_per_row * (ycellidx + neighbory) +
                                   number_of_cell_per_row * number_of_cell_per_row * (zcellidx + neighborz);
                    if (plist.find(ncellidx) != plist.end())
                        particle_complexity += plist[ncellidx].get()->size();
                }
            }
        }
        FloatingPointPrecision cell_weighted_load = weight * particle_complexity; //between 0 and npart.
        loads.push_back(cell_weighted_load);
    }
    return loads;
};

}// end namespace topology

namespace load_balancing {

template<typename RealType>
RealType compute_gini_index(std::vector<RealType> const& revenues){
    const int pop_size = revenues.size();
    std::vector<int> world(pop_size);
    std::iota(world.begin(), world.end(), 1);
    double h = 1.0 / (RealType) pop_size;
    RealType total_workload_sec = std::accumulate(revenues.begin(), revenues.end(), 0.0); //summed time + communications
    std::vector<RealType> workload_ratios = functional::map<RealType>(revenues, [&total_workload_sec](auto v){return v / total_workload_sec;});
    std::sort(workload_ratios.begin(), workload_ratios.end());
    std::vector<RealType> cumulative_workload_ratios = functional::scan_left(workload_ratios, [](auto acc, auto v){ return acc + v;}, (RealType) 0.0);
    const int nb_ratios = cumulative_workload_ratios.size();
    RealType gini_area = 0.0;
    for(size_t i = 1; i < nb_ratios; ++i)
        gini_area += h * (cumulative_workload_ratios[i-1] + cumulative_workload_ratios[i]) / 2.0;
    RealType gini_idx = (0.5 - gini_area) / 0.5;
    return gini_idx < std::numeric_limits<RealType>::epsilon() ? 0 : gini_idx;
}

template<typename Container>
double load_imbalance_standard_metric(Container const &timings) {
    return 100.0 *
           (std::max(timings.begin(), timings.end()) / gsl_stats_mean(&timings.front(), timings.size(), 1) - 1.0);
}

}// end namespace load_balancing

namespace load_dynamic {

template<typename Realtype, typename Container>
inline Realtype compute_ema(int size, Realtype alpha, const Container &Y) {
    const int position = Y.size() - size;
    const size_t starting_el = position >= 0 ? position : 0;
    Realtype acc = Y[starting_el];
    for (size_t t = starting_el + 1; t < Y.size(); ++t) acc = alpha * Y[t] + (1 - alpha) * acc;
    return acc;
}

template<typename Realtype, typename Container>
inline Realtype compute_ma(int size, const Container &Y) {
    auto begin_it = size > Y.size() ? Y.begin() : Y.end() - size;
    return std::accumulate(begin_it, Y.end(), 0.0) / Y.size();
}

/**
* Moving Average Convergence Divergence, will tell us if the "local" trend follow the "long term" trend.
* + : Tell us that it increase compared to long term
* - : Tell us that it decrease ""
* @param Y the window
* @return MACD indicator
*/
template<typename Realtype, typename Container>
inline Realtype compute_macd_ema(const Container &Y, const int sz_small_ema = 12, const int sz_big_ema = 26,
                                 const Realtype alpha = 0.95) {
    return compute_ema(sz_small_ema, alpha, Y) - compute_ema(sz_big_ema, alpha, Y);
}

template<typename Container>
inline double compute_macd_ma(const Container &Y, const int sz_small_ema = 12, const int sz_big_ema = 26) {
    return compute_ma(sz_small_ema, Y) - compute_ma(sz_big_ema, Y);
}

}// end namespace load_dynamic

std::vector<float>
compute_metrics(std::shared_ptr<SlidingWindow<double>> &window_times,
                std::shared_ptr<SlidingWindow<double>> &window_gini_times,
                std::shared_ptr<SlidingWindow<double>> &window_gini_complexities,
                std::shared_ptr<SlidingWindow<double>> &window_gini_communications,
                float true_iteration_time, std::vector<double> times,
                int sent, int received, float complexity, int my_rank, MPI_Comm comm, int exec_rank = 0) {

    int nproc = 0;
    MPI_Comm_size(comm, &nproc);

    std::vector<float> communications(nproc);
    float fsent = (float) (sent + received);

    MPI_Gather(&fsent, 1, MPI_FLOAT, &communications.front(), 1, MPI_FLOAT, exec_rank, comm);
    std::vector<float> complexities(nproc);
    MPI_Gather(&complexity, 1, MPI_FLOAT, &complexities.front(), 1, MPI_FLOAT, exec_rank, comm);

    if (my_rank == exec_rank) {
        float gini_times = (float) load_balancing::compute_gini_index<double>(times);
        float gini_complexities = load_balancing::compute_gini_index(complexities);
        float gini_communications = load_balancing::compute_gini_index(communications);

        float skewness_times = (float) gsl_stats_skew(&times.front(), 1, times.size());
        float skewness_complexities = gsl_stats_float_skew(&complexities.front(), 1, complexities.size());
        float skewness_communications = gsl_stats_float_skew(&communications.front(), 1, communications.size());

        window_times->add(true_iteration_time);
        window_gini_complexities->add(gini_complexities);
        window_gini_times->add(gini_times);
        window_gini_communications->add(gini_communications);

        // Generate y from 0 to 1 and store in a vector
        std::vector<float> it(window_gini_times->data_container.size());
        std::iota(it.begin(), it.end(), 0);

        float slope_gini_times = statistic::linear_regression<float>(it, window_gini_times->data_container).first;
        float macd_gini_times = metric::load_dynamic::compute_macd_ema<float>(window_gini_times->data_container, 12, 26,
                                                                              2.0 /
                                                                              (window_gini_times->data_container.size() +
                                                                               1));
        float slope_gini_complexity = statistic::linear_regression<float>(it,
                                                                          window_gini_complexities->data_container).first;
        float macd_gini_complexity = metric::load_dynamic::compute_macd_ema<float>(
                window_gini_complexities->data_container, 12, 26,
                1.0 / (window_gini_complexities->data_container.size() + 1));
        float slope_gini_communications = statistic::linear_regression<float>(it,
                                                                              window_gini_communications->data_container).first;
        float macd_gini_communications = metric::load_dynamic::compute_macd_ema<float>(
                window_gini_communications->data_container, 12, 26,
                1.0 / (window_gini_complexities->data_container.size() + 1));
        float slope_times = statistic::linear_regression<float>(it, window_times->data_container).first;
        float macd_times = metric::load_dynamic::compute_macd_ema<float>(window_times->data_container, 12, 26, 1.0 /
                                                                                                               (window_times->data_container.size() +
                                                                                                                1));
        return {
                gini_times, gini_complexities, gini_communications,
                slope_gini_times, slope_gini_complexity, slope_times, slope_gini_communications,
                macd_gini_times, macd_gini_complexity, macd_times, macd_gini_communications, 0.0
        };
    }
    return {};
}

template<class RealType, class Container>
RealType variance(Container c){
    const size_t s = c.size();
    RealType var = 0.0, mu = std::accumulate(c.begin(), c.end(), 0) / s;
    for(size_t i = 0; i < s; ++i){
        var += std::pow(c[i] - mu, 2);
    }
    return var / s;
};

template<class RealType>
std::vector<RealType>
all_compute_metrics(std::shared_ptr<SlidingWindow<RealType>> window_times,
                    std::shared_ptr<SlidingWindow<RealType>> window_gini_times,
                    std::shared_ptr<SlidingWindow<RealType>> window_gini_complexities,
                    std::shared_ptr<SlidingWindow<RealType>> window_gini_communications,
                    RealType true_iteration_time, std::vector<RealType> times, RealType mu_interaction_time,
                    int sent, int received, int complexity, MPI_Comm comm) {
    int nproc = 0;
    int rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    std::vector<RealType> communications(nproc);
    RealType fsent = (RealType) (sent + received);
    if(std::is_same<RealType, float>::value)
        MPI_Allgather(&fsent, 1, MPI_FLOAT, &communications.front(), 1, MPI_FLOAT, comm);
    else
        MPI_Allgather(&fsent, 1, MPI_DOUBLE, &communications.front(), 1, MPI_DOUBLE, comm);

    std::vector<RealType> complexities(nproc);
    RealType cmplx = (RealType) complexity;
    if(std::is_same<RealType, float>::value)
        MPI_Allgather(&cmplx, 1, MPI_FLOAT, &complexities.front(), 1, MPI_FLOAT, comm);
    else
        MPI_Allgather(&cmplx, 1, MPI_DOUBLE, &complexities.front(), 1, MPI_DOUBLE, comm);

    std::vector<RealType> mu_interaction_times(nproc);
    if(std::is_same<RealType, float>::value)
        MPI_Allgather(&mu_interaction_time, 1, MPI_FLOAT, &mu_interaction_times.front(), 1, MPI_FLOAT, comm);
    else
        MPI_Allgather(&mu_interaction_time, 1, MPI_DOUBLE, &mu_interaction_times.front(), 1, MPI_DOUBLE, comm);

#ifdef DEBUG
    if(!rank) {
    //    std::for_each(complexities.begin(), complexities.end(), [](auto const& el){std::cout << (int) el << " ";});
    }
#endif

    RealType gini_times = load_balancing::compute_gini_index(times);
    RealType gini_complexities   = load_balancing::compute_gini_index(complexities);
    RealType gini_communications = load_balancing::compute_gini_index(communications);

    window_times->add(true_iteration_time);
    window_gini_complexities->add(gini_complexities);
    window_gini_times->add(gini_times);
    window_gini_communications->add(gini_communications);

    // Generate y from 0 to 1 and store in a vector
    std::vector<RealType> it(window_gini_times->data_container.size());
    std::iota(it.begin(), it.end(), 0);

    RealType slope_gini_times = statistic::linear_regression<RealType>(it, window_gini_times->data_container).first;
    RealType macd_gini_times = metric::load_dynamic::compute_macd_ema(window_gini_times->data_container, 12, 26,
                                                                      2.0/(window_gini_times->data_container.size()+1));
    RealType slope_gini_complexity = statistic::linear_regression<RealType>(it, window_gini_complexities->data_container).first;
    //RealType macd_gini_complexity = metric::load_dynamic::compute_macd_ema(
    //        window_gini_complexities->data_container, 12, 26,
    //        2.0 / (window_gini_complexities->data_container.size() + 1));
    RealType slope_gini_communications = statistic::linear_regression<RealType>(it, window_gini_communications->data_container).first;
    //RealType macd_gini_communications = metric::load_dynamic::compute_macd_ema(
    //        window_gini_communications->data_container, 12, 26,
    //        2.0 / (window_gini_complexities->data_container.size() + 1));
    RealType slope_times = statistic::linear_regression<RealType>(it, window_times->data_container).first;
    //RealType macd_times = metric::load_dynamic::compute_macd_ema(window_times->data_container, 12, 26,
    //        2.0 / (window_times->data_container.size() + 1));

    
    return {
            gini_times, gini_complexities, gini_communications, // LB for times, complexity, and communications
            //*std::max_element(times.begin(), times.end()),
            //(RealType) gsl_stats_variance(&window_gini_times->data_container.front(), 1, window_gini_times->data_container.size()),
            //(RealType) gsl_stats_variance(&window_gini_complexities->data_container.front(), 1, window_gini_times->data_container.size()),
            //(RealType) gsl_stats_variance(&window_times->data_container.front(), 1, window_gini_times->data_container.size()),
            //(RealType) gsl_stats_variance(&window_gini_communications->data_container.front(), 1, window_gini_times->data_container.size()),
            //(RealType) variance<RealType>(mu_interaction_times),
            //slope_gini_times, slope_gini_complexity, slope_times, slope_gini_communications,
            //macd_gini_times, macd_gini_complexity, macd_times, macd_gini_communications
    };
}

namespace io {

void write_load_balancing_reports(std::ofstream &dataset, std::string fname, int ts_idx, float gain,
                                  std::vector<float> &dataset_entry, int rank, const sim_param_t *params,
                                  int exec_rank = 0) {

    if (rank == exec_rank) {
        int npframe = params->npframe;
        std::cout << " Gain within " << ((ts_idx) - DELTA_LB_CALL) << " and "
                  << (ts_idx) << ": " << gain << " s."
                  << std::endl;
        dataset_entry[dataset_entry.size() - 1] = gain;
        if (!dataset.is_open()) dataset.open(fname, std::ofstream::out | std::ofstream::app | std::ofstream::binary);
        write_report_data_bin<float>(dataset, ts_idx - DELTA_LB_CALL, dataset_entry, rank);
        dataset.close();
    }

}

template<class FeatureContainer>
void write_dataset(std::ofstream &dataset, std::string fname,
                   std::list<std::shared_ptr<FeatureContainer>> fcontainers,
                   int rank,  double total_time = 0.0, int exec_rank = 0 ) {

    if (rank == exec_rank) {
        if (!dataset.is_open()) dataset.open(fname, std::ofstream::out | std::ofstream::app);
        for (auto const &features_container : fcontainers) {
            for(auto const& feature: features_container->get_features()){
                dataset << std::fixed << std::setprecision(DATASET_SCIENTIFIC_PRECISION) << feature << " ";
            }
            dataset << features_container->get_target() << std::endl;
        }
        dataset << total_time << std::endl;
        dataset.close();
    }

}

}// end namespace metric::io
}// end namespace metric


#endif //NBMPI_METRICS_H
