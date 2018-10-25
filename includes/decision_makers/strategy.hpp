//
// Created by xetql on 02.07.18.
//

#ifndef NBMPI_STRATEGY_HPP
#define NBMPI_STRATEGY_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/core/optimizers/sgd/update_policies/vanilla_update.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/prereqs.hpp>

#include <random>
#include <queue>
#include "../metrics.hpp"
namespace decision_making {
    class Policy {
    public:
        virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) = 0;

        virtual void print(std::string prefix) {
            std::cout << prefix << " ";
        }
    };

    class RandomPolicy : public Policy {
        const float lb_probability;
        std::mt19937 gen = std::mt19937(0);
        std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(0.0, 1.0);
    public:
        RandomPolicy(float lb_probability, int seed = 0 /* seed MUST be the same on all MPI ranks */) :
            lb_probability(lb_probability) {
            gen.seed(seed);
        }

        virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) override {
            return dist(gen) < lb_probability;
        }

        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "Random Policy:" << std::endl;
        }

    };

    class ThresholdHeuristicPolicy : public Policy{
        const float threshold;
        inline bool is_greater_than_threshold(std::unique_ptr<metric::LBMetrics<double>> mc) {
            if (mc)
                return mc->get_gini_times() > threshold;
            else return false;
        }

    public:
        ThresholdHeuristicPolicy(float threshold) : threshold(threshold){};
        virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) override {
            return is_greater_than_threshold(std::move(mc));
        }
        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "Threshold Policy:" << std::endl;
        }
    };

    class InFilePolicy : public Policy {

    public:
        std::queue<bool> decisions;
        int period;
        InFilePolicy(std::string filename, int nframes, int npframe) {
            period = npframe;
            /* Read the targets of dataset files and apply decision at each frame */
            decisions = std::queue<bool>();
            std::ifstream dataset;
            dataset.open(filename, std::ofstream::in);
            if(!dataset.good()) throw std::runtime_error("bad repr. file");
            std::string line;
            std::string buf;
            int decision_cnt = 0;
            bool clear = false;
            while (std::getline(dataset, line)) {
                if(clear) decisions = std::queue<bool>();
                std::stringstream ss(line);       // Insert the string into a stream
                std::vector<float> tokens; // Create vector to hold our words
                while (ss >> buf) tokens.push_back(std::stof(buf.c_str()));

                if(tokens.size() == 1) {
                    clear = true;
                    continue;
                } else {
                    clear = false;
                }

                decisions.push( *(tokens.end() - 1) > 0 );

            }
            dataset.close();
        }

        virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) override {
            if(it % period == 0) {
                auto decision = decisions.front();
                decisions.pop();
                return decision;
            } else return false;
        }

        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "InFile Policy:" << std::endl;
        }
    };

    class PeriodicPolicy : public Policy{
        const int period;
    public:
        PeriodicPolicy(int period) : period(period) {}
        virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) override {
            return (it % period) == 0 && it > 0;
        }
        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "Periodic Policy ("<< std::to_string(period) << "):" << std::endl;
        }
    };

    class NoLBPolicy : public Policy{
    public:
        NoLBPolicy() {}
        virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) override {
            return false;
        }
    };

    class NeuralNetworkPolicy : public Policy {
        mlpack::ann::MeanSquaredError<> mse;
        mlpack::ann::RandomInitialization rand_init;
        mlpack::ann::FFN<mlpack::ann::MeanSquaredError<>, mlpack::ann::RandomInitialization> model;
        mlpack::optimization::RMSProp optimizer;
        std::shared_ptr<arma::mat> ds;
        arma::mat inputs, targets;
    public:

        NeuralNetworkPolicy(const std::string& ds_filename, int idx) {
            inputs.load(ds_filename+"-features-"+std::to_string(idx)+".mat", arma::raw_ascii);
            targets.load(ds_filename+"-targets-"+std::to_string(idx)+".mat", arma::raw_ascii);
            auto tin = inputs.t().eval();
            auto ttar= targets.t().eval();
            model.Add<mlpack::ann::Linear<> >(tin.n_rows, 6);
            model.Add<mlpack::ann::LeakyReLU<> >();
            model.Add<mlpack::ann::Linear<> >(6, ttar.n_rows);
            model.Add<mlpack::ann::LeakyReLU<> >();
            model.Train(tin, ttar);
        }

        void train() {
            model.Train(inputs, targets, optimizer);
        }

        /**
         * Query the model
         */
        virtual inline bool should_load_balance(int it, std::unique_ptr<metric::LBMetrics<double>> mc) override {
            if(!mc) return false;
            arma::mat features(mc->metrics);
            arma::mat responses(1, 1);
            model.Predict(features, responses);
            return responses(0,0) >= 0;
        }

        void print(std::string prefix) override {
            Policy::print(prefix);
            std::cout << "NeuralNetwork Policy:" << std::endl;
        }
    };

} // end of namespace decision_making

#endif //NBMPI_STRATEGY_HPP
