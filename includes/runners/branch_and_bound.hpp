//
// Created by xetql on 27.06.18.
//

#ifndef NBMPI_BRANCH_AND_BOUND_HPP
#define NBMPI_BRANCH_AND_BOUND_HPP

#include <sstream>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <zoltan.h>
#include <cstdlib>
#include <gsl/gsl_statistics.h>

#include "../astar.hpp"
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

#ifndef WINDOW_SIZE
#define WINDOW_SIZE 30
#endif

#ifndef N_FEATURES
#define N_FEATURES 14
#endif

#ifndef N_LABEL
#define N_LABEL 1
#endif

#ifndef TICK_FREQ
#define TICK_FREQ 1 // MPI_Wtick()
#endif

template<int N> using Domain = std::vector<partitioning::geometric::Domain<N> >;
template<int N> using LBNode = Node<MESH_DATA<N>, Domain<N>>;
template<int N> using LBSolutionPath = std::list<std::shared_ptr<LBNode<N> > >;

template<int N> using NodeQueue = std::multiset<std::shared_ptr<LBNode<N> >, Compare<MESH_DATA<N>, Domain<N>> >;

template<int N>
inline bool is_a_solution(const std::shared_ptr<LBNode<N>> node, const int LAST_ITERATION) {
    return node->get_node_type() == NodeType::Computing && node->end_it >= LAST_ITERATION;
}

template<int N>
inline void remove_unnecessary_nodes(NodeQueue<N>& queue, std::shared_ptr<LBNode<N> > node) {
    for(auto it = queue.begin(); it != queue.end(); ) {
        auto current_node = *it;
        if(current_node->get_node_type() == NodeType::Partitioning && node->end_it == current_node->end_it){
            it = queue.erase(it);
        } else{
            ++it;
        }
    }
}

template<int N>
std::vector<LBSolutionPath<N>> Astar_runner(
        MESH_DATA<N> *p_mesh_data,
        Zoltan_Struct *p_load_balancer,
        sim_param_t *params,
        const MPI_Comm comm = MPI_COMM_WORLD) {
    // MPI Init ...
    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    const int nframes = params->nframes;
    const int npframe = params->npframe;
    const int NB_BEST_SOLUTIONS = (int) params->nb_best_path,
            LAST_ITERATION = nframes * npframe;

    int number_of_visited_node = 0;

    double it_start, true_iteration_time, my_iteration_time;
    MESH_DATA<N> *mesh_data, tmp_data;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    Domain<N> domain_boundaries = retrieve_domain_boundaries<N>(p_load_balancer, nproc, params);
    std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;
    std::multiset<std::shared_ptr<LBNode<N> >, Compare<MESH_DATA<N>, Domain<N>> > queue;

    //std::shared_ptr<SlidingWindow<double>> window_gini_times, window_gini_complexities, window_times, window_gini_communications;
    std::shared_ptr<SlidingWindow<double>>
            window_gini_times = std::make_shared<SlidingWindow<double>>(params->npframe / 2),
            window_gini_complexities = std::make_shared<SlidingWindow<double>>(params->npframe / 2),
            window_times = std::make_shared<SlidingWindow<double>>(params->npframe / 2),
            window_gini_communications = std::make_shared<SlidingWindow<double>>(params->npframe / 2);

    std::vector<double>
            dataset_entry(N_FEATURES + N_LABEL),
            features(N_FEATURES + N_LABEL),
            times(nproc),
            optimal_frame_time_lookup_table(nframes);

    std::vector<bool> tried_to_load_balance(nframes, false);

    MPI_Barrier(comm);
    //load_balancing::geometric::migrate_particles<N>(p_mesh_data->els, domain_boundaries, datatype, comm);
    zoltan_load_balance<N>(p_mesh_data, domain_boundaries, p_load_balancer, nproc, params, datatype, comm);
    MPI_Barrier(comm);

    std::shared_ptr<LBNode<N> > current_node = std::make_shared<LBNode<N>>(*p_mesh_data, domain_boundaries, p_load_balancer, npframe), solution;
    std::vector<std::shared_ptr<LBNode<N> > > solutions;

    current_node->metrics_before_decision = dataset_entry;
    current_node->last_metric = dataset_entry;

    double child_cost;

    int complexity, received, sent;

    while (solutions.size() < NB_BEST_SOLUTIONS) {
        auto children = current_node->get_children();
        number_of_visited_node++;
        for(std::shared_ptr<LBNode<N> >& child : children) {
            if(!child) continue;

            /*window_gini_times = child->window_gini_times;
            window_gini_complexities = child->window_gini_complexities;
            window_times = child->window_times;
            window_gini_communications = child->window_gini_communications;*/

            mesh_data = &child->mesh_data;
            domain_boundaries = child->domain;

            auto load_balancer = child->lb;

            child_cost = 0;
            int frame = 1 + (child->start_it / npframe), frame_id = (child->start_it / npframe);

            switch(child->get_node_type()) {
                case NodeType::Partitioning:
                    if(!tried_to_load_balance[frame_id]) {
                        if(!rank) std::cout << child << std::endl;

                        MPI_Barrier(comm);
                        double partitioning_start_time = MPI_Wtime();
                        zoltan_load_balance<N>(mesh_data, domain_boundaries, load_balancer, nproc, params, datatype, comm);
                        double my_partitioning_time = MPI_Wtime() - partitioning_start_time;
                        MPI_Barrier(comm);

                        MPI_Allreduce(&my_partitioning_time, &child_cost, 1, MPI_DOUBLE, MPI_MAX, comm);

                        child->last_metric = child->metrics_before_decision;
                        child->domain = domain_boundaries; // update the partitioning
                        child->set_cost(child_cost);     //set how much time it costed
                        queue.insert(child);
                    }
                    break;
                case NodeType::Computing:
                    if(child->get_decision() == NodeLBDecision::DoNothing || !tried_to_load_balance[frame_id]){
                        if (child->get_decision() != NodeLBDecision::DoNothing) {
                            tried_to_load_balance[frame_id] = true;
                            remove_unnecessary_nodes(queue, child);
                        }

                        child->end_it += npframe;
                        if(!rank) std::cout << child << std::endl;

                        std::tuple<int, int, int> computation_info;
                        std::vector<double> dataset_entry(N_FEATURES + N_LABEL);

                        domain_boundaries = retrieve_domain_boundaries<N>(load_balancer, nproc, params);

                        for (int i = 0; i < npframe; ++i) {
                            MPI_Barrier(comm);
                            const double __start = MPI_Wtime();

                            load_balancing::geometric::migrate_particles<N>(mesh_data->els, domain_boundaries, datatype, comm);

                            MPI_Barrier(comm);

                            //computation_info = lennard_jones::compute_one_step<N>(mesh_data, plklist, domain_boundaries, datatype, params, comm, frame);

                            const double my_iteration_time = MPI_Wtime() - __start;

                            //std::tie(complexity, received, sent) = computation_info;
                            complexity = 0; received = 0; sent = 0;

                            MPI_Allgather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, comm);
                            true_iteration_time = *std::max_element(times.begin(), times.end());
                            dataset_entry = metric::all_compute_metrics(window_times, window_gini_times,
                                                                        window_gini_complexities, window_gini_communications,
                                                                        true_iteration_time, times, 0.0, sent, received, complexity, comm);
                            if(!rank) {
                                for(auto const& feature: dataset_entry){
                                    std::cout << std::setw(6) << std::fixed << std::setprecision(3) << feature << ",";
                                }
                                std::cout << std::endl;
                            }
                            child_cost += true_iteration_time;
                        }

                        child->last_metric = {};
                        std::copy(dataset_entry.begin(), dataset_entry.end(), std::back_inserter(child->last_metric));
                        child->last_metric.push_back(dataset_entry.at(0) - child->metrics_before_decision.at(0));
                        child->last_metric.push_back(dataset_entry.at(1) - child->metrics_before_decision.at(1));
                        child->last_metric.push_back(dataset_entry.at(2) - child->metrics_before_decision.at(2));
                        child->last_metric.push_back(dataset_entry.at(3) - child->metrics_before_decision.at(3));

                        child->domain = domain_boundaries; //update the partitioning
                        child->set_cost(child_cost);     //set how much time it costed
                        queue.insert(child);
                    }
                    break;
            }
            if(!rank) std::cout << child->cost() << std::endl;
            MPI_Barrier(comm);

        }
        if(!rank) std::cout << "QUEUE SIZE="<<queue.size() << std::endl;
        do {

            current_node = *queue.begin();                  // Next best node
            queue.erase(queue.begin());                     // Remove it from the queue
            if (is_a_solution(current_node, LAST_ITERATION))// if it is a solution node
                solutions.push_back(current_node);          // then add to solution list
        // while current node is a solution and we search for more...
        } while(is_a_solution(current_node, LAST_ITERATION) && solutions.size() < NB_BEST_SOLUTIONS);

        MPI_Barrier(comm);
    }

    std::vector<LBSolutionPath<N> > best_paths;
    size_t path_idx = 0;
    while(path_idx < NB_BEST_SOLUTIONS) {

        LBSolutionPath<N> solution_path;
        auto solution = solutions[path_idx];
        while (solution->parent.get() != nullptr) { //reconstruct path
            if(solution->type == NodeType::Computing) solution_path.push_front(solution);
            solution = solution->parent;
        }
        best_paths.push_back(solution_path);
        path_idx++;
    }
    if(!rank) std::cout << "Visited nodes:"<<number_of_visited_node<<std::endl;
    return best_paths;
}

/*
template<int N>
std::list<std::shared_ptr<NodeWithoutParticles<std::vector<partitioning::geometric::Domain<N>>>>> IDAstar_runner(
        const MESH_DATA<N> *p_mesh_data,
        Zoltan_Struct *load_balancer,
        const sim_param_t *params,
        const MPI_Comm comm = MPI_COMM_WORLD) {

    int nproc, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    const int nframes = params->nframes;
    const int npframe = params->npframe;
    double it_start, true_iteration_time, my_iteration_time;

    MESH_DATA<N> mesh_data = *p_mesh_data;

    std::map<int, MESH_DATA<N>> particles_states;
    particles_states.emplace(0, mesh_data);

    using Domain = std::vector<partitioning::geometric::Domain<N>>;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<N>();
    Domain domain_boundaries(nproc);
    {
        int dim;
        double xmin, ymin, zmin, xmax, ymax, zmax;
        // get boundaries of all domains
        for (int part = 0; part < nproc; ++part) {
            Zoltan_RCB_Box(load_balancer, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
            auto domain = partitioning::geometric::borders_to_domain<N>(xmin, ymin, zmin, xmax, ymax, zmax,
                                                                        params->simsize);
            domain_boundaries[part] = domain;
        }
    }
    Domain start_domain_boundaries = domain_boundaries;
    std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<N> > > > plklist;

    std::priority_queue<
            std::shared_ptr<NodeWithoutParticles<Domain> >,
            std::vector<std::shared_ptr<NodeWithoutParticles<Domain> > >,
            CompareNodeWithoutParticles<Domain> > queue;

    std::shared_ptr<SlidingWindow<double>> window_gini_times, window_gini_complexities, window_times, window_gini_communications;
    window_gini_times = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    window_times = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    window_gini_complexities = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    window_gini_communications = std::make_shared<SlidingWindow<double>>(params->npframe / 2);
    std::vector<double> dataset_entry(N_FEATURES + N_LABEL), features(N_FEATURES + N_LABEL);

    std::vector<double> times(nproc);

    std::shared_ptr<NodeWithoutParticles<Domain>> current_node, solution;

    std::list<std::shared_ptr<NodeWithoutParticles<Domain>>> solution_path;
    int it = 0;
    double child_cost, true_child_cost;

    // Compute the optimal time per step
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int ranks[1] = {0};
    MPI_Group foreman_group;
    MPI_Group_incl(world_group, 1, ranks, &foreman_group);
    MPI_Comm foreman_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, foreman_group, 0, &foreman_comm);

    MESH_DATA<N> tmp_data;

    Domain tmp_domain_boundary = {{std::make_pair(0.0, params->simsize), std::make_pair(0.0, params->simsize)}};
    load_balancing::gather_elements_on(nproc, rank, params->npart, mesh_data.els, 0, tmp_data.els,
                                       datatype.elements_datatype, comm);
    MESH_DATA<N> *p_tmp_data = &tmp_data;
    std::vector<double> optimal_frame_time_lookup_table(nframes);
    if (rank == 0) {
        SimpleCSVFormatter frame_formater(',');
        std::ofstream frame_file;
        if(params->record) {
            std::string mkdir_cmd = "mkdir -p data/time-series/"+std::to_string(params->seed);
            system(mkdir_cmd.c_str());
        }

        for(int frame = 0; frame < nframes; frame++){
            double frame_time = 0;
            for(int step = 0; step < npframe; step++){
                it_start = MPI_Wtime();
                load_balancing::geometric::migrate_particles<N>(p_tmp_data->els, tmp_domain_boundary, datatype, foreman_comm);
                auto computation_info = lennard_jones::compute_one_step<N>(p_tmp_data, plklist, tmp_domain_boundary, datatype,
                                                                           params, foreman_comm);
                frame_time  += (MPI_Wtime() - it_start);
            }
            if(params->record){
                frame_file.open("data/time-series/"+std::to_string(params->seed)+"/run_cpp.csv."+std::to_string(frame+1), std::ofstream::out | std::ofstream::trunc);
                frame_formater.write_header(frame_file, params->npframe, params->simsize);
                write_frame_data(frame_file, p_tmp_data->els, frame_formater, params);
                frame_file.close();
            }
            optimal_frame_time_lookup_table[frame] = frame_time / nproc;
        }
    }

    MPI_Bcast(&optimal_frame_time_lookup_table.front(), nframes, MPI_DOUBLE, 0, comm);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double total_optimal_time = std::accumulate(optimal_frame_time_lookup_table.begin(), optimal_frame_time_lookup_table.end(), 0.0);
    if (rank == 0) std::cout << "Optimal time: " << (total_optimal_time) << std::endl;
    double shallowest_possible_solution = total_optimal_time * 5.0;

    MPI_Barrier(comm);

    MPI_Group_free(&foreman_group);
    MPI_Group_free(&world_group);
    if (rank == 0) MPI_Comm_free(&foreman_comm);

    int number_of_visited_node = 0;
    bool solution_found = false;
    while (!solution_found) {
        queue = std::priority_queue<
                std::shared_ptr<NodeWithoutParticles<Domain> >,
                std::vector<std::shared_ptr<NodeWithoutParticles<Domain> > >,
                CompareNodeWithoutParticles<Domain> >();

        current_node = std::make_shared<NodeWithoutParticles<Domain>>(start_domain_boundaries);
        std::fill(dataset_entry.begin(), dataset_entry.end(), 0);
        current_node->metrics_before_decision = dataset_entry;
        current_node->last_metric    = dataset_entry;
        it = 0;
        while (it < nframes * npframe) {
            int number_of_frames_computed;
            auto children = current_node->get_children();
            number_of_visited_node++;
#ifdef DEBUG
            if(!rank){
                std::cout << "Number of visited node: " << number_of_visited_node;
                std::cout << ", Number of node in queue: " << queue.size();
                std::cout << ", Current iteration: " << it <<std::endl;
            }
#endif
            mesh_data = particles_states[it];
            domain_boundaries = children.first->domain;
            load_balancing::geometric::migrate_particles<N>(mesh_data.els, domain_boundaries, datatype, comm);

            child_cost = 0;
            MPI_Barrier(comm);
            for (int i = 0; i < npframe; i++) {
                it_start = MPI_Wtime();
                if (i == 0)
                    zoltan_load_balance<N>(&mesh_data, domain_boundaries, load_balancer, nproc, params, datatype, comm);
                if (i > 0)
                    load_balancing::geometric::migrate_particles<N>(mesh_data.els, domain_boundaries, datatype, comm);
                MPI_Barrier(comm);
                std::tuple<int, int, int> computation_info;

                try {
                    double cpt_step_start_time = MPI_Wtime();
                    computation_info = lennard_jones::compute_one_step<N>(&mesh_data, plklist, domain_boundaries, datatype,
                                                                          params, comm);
                    int complexity = std::get<0>(computation_info),
                            received = std::get<1>(computation_info),
                            sent = std::get<2>(computation_info);
                    double mean_interaction_cpt_time = (MPI_Wtime() - cpt_step_start_time) / complexity;
                    my_iteration_time = MPI_Wtime() - it_start;
                    MPI_Allgather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, comm);
                    true_iteration_time = *std::max_element(times.begin(), times.end());

                    dataset_entry = metric::all_compute_metrics(window_times, window_gini_times,
                                                                window_gini_complexities, window_gini_communications,
                                                                true_iteration_time, times, mean_interaction_cpt_time, sent, received, complexity,
                                                                comm);
#ifdef DEBUG
                    if(!rank){
                        std::cout << std::fixed << std::setprecision(3);
                        std::for_each(dataset_entry.begin(), dataset_entry.end(), [](auto const& el){std::cout << el << " ";});
                        std::cout << std::endl;
                    }
#endif
                    child_cost += true_iteration_time;
                } catch (const std::runtime_error e) {
                    std::cout << "Panic! ";
                    std::cout << children.first << std::endl;
                    throw new std::runtime_error("particle out domain");
                }
            }

            MPI_Allreduce(&child_cost, &true_child_cost, 1, MPI_DOUBLE, MPI_MAX, comm);

            if (particles_states.find(it + npframe) == particles_states.end())
                particles_states[it + npframe] = mesh_data;

            children.first->end_it = it + npframe;
            children.first->node_cost = true_child_cost;
            number_of_frames_computed  = (children.first->end_it / npframe);
            children.first->heuristic_cost = std::accumulate(optimal_frame_time_lookup_table.begin()+(number_of_frames_computed-1), optimal_frame_time_lookup_table.end(), 0);
            children.first->domain = domain_boundaries;
            children.first->path_cost += true_child_cost;
            children.first->last_metric = dataset_entry;

            mesh_data = particles_states[it];
            domain_boundaries = children.second->domain;
            load_balancing::geometric::migrate_particles<N>(mesh_data.els, domain_boundaries, datatype, comm);

            child_cost = 0;
            MPI_Barrier(comm);
            for (int i = 0; i < npframe; i++) {
                it_start = MPI_Wtime();
                load_balancing::geometric::migrate_particles<N>(mesh_data.els, domain_boundaries, datatype, comm);
                MPI_Barrier(comm);
                std::tuple<int, int, int> computation_info;
                try {
                    double cpt_step_start_time = MPI_Wtime();
                    computation_info = lennard_jones::compute_one_step<N>(&mesh_data, plklist, domain_boundaries, datatype,
                                                                          params, comm);
                    int complexity = std::get<0>(computation_info),
                            received = std::get<1>(computation_info),
                            sent = std::get<2>(computation_info);
                    double mean_interaction_cpt_time = (MPI_Wtime() - cpt_step_start_time) / complexity;
                    my_iteration_time = MPI_Wtime() - it_start;
                    MPI_Allgather(&my_iteration_time, 1, MPI_DOUBLE, &times.front(), 1, MPI_DOUBLE, comm);
                    true_iteration_time = *std::max_element(times.begin(), times.end());

                    dataset_entry = metric::all_compute_metrics(window_times, window_gini_times,
                                                                window_gini_complexities, window_gini_communications,
                                                                true_iteration_time, times, mean_interaction_cpt_time, sent, received, complexity,
                                                                comm);
#ifdef DEBUG
                    if(!rank){
                        std::cout << std::fixed << std::setprecision(8);
                        std::for_each(dataset_entry.begin(), dataset_entry.end(), [](auto const& el){std::cout << el << " ";});
                        std::cout << std::endl;
                    }
#endif
                    child_cost += true_iteration_time;
                } catch (const std::runtime_error error) {
                    std::cout << "Panic! ";
                    std::cout << children.second << std::endl;
                    throw new std::runtime_error("particle out domain");
                }
            }

            MPI_Allreduce(&child_cost, &true_child_cost, 1, MPI_DOUBLE, MPI_MAX, comm);

            if (particles_states.find(it + npframe) == particles_states.end())
                particles_states[it + npframe] = mesh_data;

            children.second->end_it = it + npframe;
            children.second->node_cost = true_child_cost;
            number_of_frames_computed  = (children.first->end_it / npframe);
            children.second->heuristic_cost =
                    true_child_cost + std::accumulate(optimal_frame_time_lookup_table.begin()+(number_of_frames_computed-1), optimal_frame_time_lookup_table.end(), 0);
            children.second->domain = domain_boundaries;
            children.second->path_cost += true_child_cost;
            children.second->last_metric = dataset_entry;

            if (children.first->cost() <= shallowest_possible_solution)
                queue.push(children.first);
            if (children.second->cost() <= shallowest_possible_solution)
                queue.push(children.second);
            if(queue.empty()) break;
            current_node = queue.top();
            queue.pop();

            if (current_node->end_it >= nframes * npframe) {
                solution = current_node;
                if (!rank) std::cout << solution->cost() << " seconds" << std::endl;
                //retrieve best path
                while (solution->parent.get() != nullptr) {
                    solution_path.push_front(solution);
                    solution = solution->parent;
                }
                return solution_path;
            }
            it = current_node->end_it;
            MPI_Barrier(comm);
        }
        shallowest_possible_solution *= 1.3;
    }
}
*/
#endif //NBMPI_BRANCH_AND_BOUND_HPP
