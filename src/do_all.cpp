//
// Created by xetql on 7/21/18.
//
#include <string>
#include <mpi.h>
#include <random>

#include <zoltan.h>

#include "../includes/runners/simulator.hpp"

#include "../includes/initial_conditions.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/params.hpp"
#include "../includes/runners/branch_and_bound.hpp"


int main(int argc, char **argv) {
    constexpr int DIMENSION = 3;
    sim_param_t params;
    int rank, nproc, dim;
    float ver;
    MESH_DATA<DIMENSION> original_data, mesh_data;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    params.world_size = (unsigned int) nproc;

    //Define the output simulation name
    const std::string SIMULATION_STR_NAME = std::to_string(params.seed) +
                                            "-" + std::to_string(params.nframes) + "x" +
                                            std::to_string(params.npframe) +
                                            "-" + std::to_string(params.world_size) +
                                            "-" + std::to_string(params.npart) +
                                            "-" + std::to_string((params.T0)) +
                                            "-" + std::to_string((params.G)) +
                                            "-" + std::to_string((params.simsize)) +
                                            "-" + std::to_string((params.eps_lj)) +
                                            "-" + std::to_string((params.sig_lj)) +
                                            "-" + std::to_string(params.dt) +
                                            "_" + std::to_string(params.particle_init_conf);

    if (rank == 0) {
        std::cout << "==============================================" << std::endl;
        std::cout << "= Simulation is starting now...                 " << std::endl;
        std::cout << "= Parameters: " << std::endl;
        std::cout << "= Particles: " << params.npart << std::endl;
        std::cout << "= Seed: " << params.seed << std::endl;
        std::cout << "= PEs: " << params.world_size << std::endl;
        std::cout << "= Simulation size: " << params.simsize << std::endl;
        std::cout << "= Number of time-steps: " << params.nframes << "x" << params.npframe << std::endl;
        std::cout << "= Initial conditions: " << std::endl;
        std::cout << "= SIG:" << params.sig_lj << std::endl;
        std::cout << "= EPS:  " << params.eps_lj << std::endl;
        std::cout << "= Borders: collisions " << std::endl;
        std::cout << "= Gravity:  " << params.G << std::endl;
        std::cout << "= Temperature: " << params.T0 << std::endl;
        std::cout << "==============================================" << std::endl;
    }

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<DIMENSION>();

    int rc = Zoltan_Initialize(argc, argv, &ver);

    if (rc != ZOLTAN_OK) {
        MPI_Finalize();
        exit(0);
    }
    std::vector<partitioning::geometric::Domain<DIMENSION>> domain_boundaries;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////START PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (rank == 0) {
        const std::string IMPORT_FILENAME
                = std::to_string(params.npart) + "-" +
                  std::to_string(params.particle_init_conf) + "-" +
                  std::to_string(params.simsize) + ".particles";
        if (file_exists(IMPORT_FILENAME)) {
            std::cout << "importing from file ..." << std::endl;
            elements::import_from_file<DIMENSION, elements::ElementRealType>(IMPORT_FILENAME, mesh_data.els);
            std::cout << "Done !" << std::endl;
        } else {
            std::cout << "Generating data ..." << std::endl;
            std::shared_ptr<initial_condition::lennard_jones::RejectionCondition<DIMENSION>> condition;
            const int MAX_TRIAL = 100000;
            int NB_CLUSTERS;
            std::vector<int> clusters;
            using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;
            std::queue<ElementGeneratorCfg> elements_generators;
            switch (params.particle_init_conf) {
                case 1: //uniformly distributed
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(
                                    params.seed, MAX_TRIAL), params.npart));
                    break;
                case 2: //Half full half empty
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                                    params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 3: //Wall of particle
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
                    );
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::ParticleWallElementsGenerator<DIMENSION>>(
                                    params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 4: //cluster(s)
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart));
                    break;
                case 5: //custom various density
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
                    );
                    NB_CLUSTERS = 2;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart / 4);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart / 4));
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                                    params.simsize / 10, false, params.seed, MAX_TRIAL), 3 * params.npart / 4));
                    break;
                case 6: //custom various density
                    condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                            &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
                    );
                    NB_CLUSTERS = 1;
                    clusters.resize(NB_CLUSTERS);
                    std::fill(clusters.begin(), clusters.end(), params.npart);
                    elements_generators.push(std::make_pair(
                            std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                                    clusters, params.seed, MAX_TRIAL), params.npart));

                    break;
                default:
                    MPI_Finalize();
                    throw std::runtime_error("Unknown particle distribution.");
            }
            while (!elements_generators.empty()) {
                ElementGeneratorCfg el_gen = elements_generators.front();
                el_gen.first->generate_elements(mesh_data.els, el_gen.second, condition);
                elements_generators.pop();
                std::cout << el_gen.second << std::endl;
            }
            std::cout << "Done !" << std::endl;
        }
    }

    original_data = mesh_data; //copy data elsewhere for future use
    std::shared_ptr<decision_making::Policy> lb_policy;
    const std::string RESULT_FILENAME = SIMULATION_STR_NAME + ".result";
    const std::string DATASET_FILENAME = SIMULATION_STR_NAME + ".dataset";
    Zoltan_Struct *zz;
    {
        for (unsigned int lb_policy_idx = 1 /* skip the no lb ... */ ; lb_policy_idx <= 4; ++lb_policy_idx) {
            mesh_data = original_data; //recover data from the clean copy
            switch (lb_policy_idx) {
                case 0:
                    lb_policy = std::make_shared<decision_making::NoLBPolicy>();
                    break;
                case 1:
                    lb_policy = std::make_shared<decision_making::PeriodicPolicy>(25);
                    break;
                case 2://load the file created above
                    lb_policy = std::make_shared<decision_making::PeriodicPolicy>(100);
                    break;
                case 3:
                    lb_policy = std::make_shared<decision_making::RandomPolicy>(0.1, params.seed);
                    break;
                case 4://TODO: threshold should be a parameter
                    lb_policy = std::make_shared<decision_making::ThresholdHeuristicPolicy>(0.3);
                    break;
                case 5://load the file created above
                    lb_policy = std::make_shared<decision_making::InFilePolicy>(
                            DATASET_FILENAME, params.nframes, params.npframe);
                    break;
                case 6://neural net policy
                    mlpack::math::RandomSeed(params.seed);
                    lb_policy = std::make_shared<decision_making::NeuralNetworkPolicy>(DATASET_FILENAME, 0);
                    break;
                default:
                    throw std::runtime_error("unknown lb policy");
            }

            if (!rank) lb_policy->print(std::to_string(lb_policy_idx));

            zz = zoltan_create_wrapper(ENABLE_AUTOMATIC_MIGRATION);

            zoltan_load_balance<DIMENSION>(&mesh_data, zz, datatype, MPI_COMM_WORLD, ENABLE_AUTOMATIC_MIGRATION);

            auto time_spent = simulate<DIMENSION>(nullptr, &mesh_data, zz, lb_policy, &params, MPI_COMM_WORLD,
                                                  ENABLE_AUTOMATIC_MIGRATION);

            if (!rank) {
                std::ofstream result;
                result.open(RESULT_FILENAME, std::ofstream::app | std::ofstream::out);
                std::cout << time_spent << std::endl;
                result << time_spent << std::endl;
                result.close();
            }

            Zoltan_Destroy(&zz);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    {
        zz = zoltan_create_wrapper(ENABLE_AUTOMATIC_MIGRATION);
        zoltan_load_balance<DIMENSION>(&mesh_data, zz, datatype, MPI_COMM_WORLD, ENABLE_AUTOMATIC_MIGRATION);

        if (!rank) std::cout << "Run A* simulation to get optimal path..." << std::endl;
        auto astar_optimal_paths = Astar_runner<DIMENSION>(&mesh_data, zz, &params, MPI_COMM_WORLD, ENABLE_AUTOMATIC_MIGRATION);

        std::ofstream dataset;
        if (!rank && file_exists(DATASET_FILENAME)) std::remove(DATASET_FILENAME.c_str());

        size_t sol_idx = 0;
        for (auto const& solution : astar_optimal_paths) {
            if (!rank){
                std::cout << "Solution ("<<sol_idx<<"):" << std::endl;
                for(auto const& node : solution){
                    if(node->type == NodeType::Computing)
                        std::cout << std::setprecision(10) << "frame time: " << node->get_node_cost() << " ? "<< (node->decision==NodeLBDecision::LoadBalance ? "1" : "0") << std::endl;
                }
                arma::mat feat, target;
                std::tie(feat, target) = to_armadillo_mat(solution, 13);
                feat.save(DATASET_FILENAME+"-features-"+std::to_string(sol_idx)+".mat", arma::raw_ascii);
                target.save(DATASET_FILENAME+"-targets-"+std::to_string(sol_idx)+".mat", arma::raw_ascii);
            }
            metric::io::write_dataset(dataset, DATASET_FILENAME, solution, rank, (*(std::next(solution.end(), -1)))->cost());
            sol_idx++;
        }
        Zoltan_Destroy(&zz);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////FINISHED A* COMPUTATION, START COMPARISON WITH HEURISTICS//////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (!rank) std::cout << "Start to compare best path with heuristics..." << std::endl;
    if(!rank && file_exists(RESULT_FILENAME)) std::remove(RESULT_FILENAME.c_str());

    {
        for (unsigned int lb_policy_idx = 5 /* skip the no lb ... */ ; lb_policy_idx <= 6  ; ++lb_policy_idx) {
            mesh_data = original_data; //recover data from the clean copy
            switch (lb_policy_idx) {
                case 0:
                    lb_policy = std::make_shared<decision_making::NoLBPolicy>();
                    break;
                case 1:
                    lb_policy = std::make_shared<decision_making::PeriodicPolicy>(25);
                    break;
                case 2://load the file created above
                    lb_policy = std::make_shared<decision_making::PeriodicPolicy>(100);
                    break;
                case 3:
                    lb_policy = std::make_shared<decision_making::RandomPolicy>(0.1, params.seed);
                    break;
                case 4://TODO: threshold should be a parameter
                    lb_policy = std::make_shared<decision_making::ThresholdHeuristicPolicy>(0.3);
                    break;
                case 5://load the file created above
                    lb_policy = std::make_shared<decision_making::InFilePolicy>(
                            DATASET_FILENAME, params.nframes, params.npframe);
                    break;
                case 6://neural net policy
                    mlpack::math::RandomSeed(params.seed);
                    lb_policy = std::make_shared<decision_making::NeuralNetworkPolicy>(DATASET_FILENAME, 0);
                    break;
                default:
                    throw std::runtime_error("unknown lb policy");
            }

            if(!rank) lb_policy->print(std::to_string(lb_policy_idx));

            zz = zoltan_create_wrapper(ENABLE_AUTOMATIC_MIGRATION);

            zoltan_load_balance<DIMENSION>(&mesh_data, zz, datatype, MPI_COMM_WORLD, ENABLE_AUTOMATIC_MIGRATION);

            auto time_spent = simulate<DIMENSION>(nullptr, &mesh_data, zz,  lb_policy, &params, MPI_COMM_WORLD, ENABLE_AUTOMATIC_MIGRATION);

            if (!rank) {
                std::ofstream result;
                result.open(RESULT_FILENAME, std::ofstream::app | std::ofstream::out);
                std::cout << time_spent << std::endl;
                result << time_spent << std::endl;
                result.close();
            }

            Zoltan_Destroy(&zz);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}