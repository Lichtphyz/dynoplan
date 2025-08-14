#include <iostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iterator>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <bits/stdc++.h>
// fcl
#include "fcl/broadphase/broadphase_collision_manager.h"
#include <fcl/fcl.h>
// BOOST
#include <boost/program_options.hpp>
#include <boost/program_options.hpp>
#include <boost/heap/d_ary_heap.hpp>
// DYNOPLAN
#include <dynoplan/optimization/ocp.hpp>
#include "dynoplan/tdbastar/compute_heu.hpp"
#include "dynoplan/nigh_custom_spaces.hpp"
#include "dynoplan/ompl/robots.h"

namespace fs = std::filesystem;
using namespace dynoplan;
#define DYNOBENCH_BASE "../dynoplan/dynobench/"

std::vector<Eigen::VectorXd> sampleFree(
    int N,
    const Eigen::VectorXd &p_lb,
    const Eigen::VectorXd &p_ub,
    const std::shared_ptr<dynobench::Model_robot> &robot)
{
  static std::random_device rd;
  static std::mt19937 gen(rd());

  const int nx = static_cast<int>(robot->nx);
  const int pos_dim = static_cast<int>(p_lb.size());

  if (nx < pos_dim)
  {
    throw std::runtime_error("Robot state has fewer dimensions than position bounds.");
  }
  if (robot->x_lb.size() != nx || robot->x_ub.size() != nx)
  {
    throw std::runtime_error("Robot bounds do not match state dimension.");
  }
  // Precompute position distributions (from environment bounds)
  std::vector<std::uniform_real_distribution<>> pos_dists;
  pos_dists.reserve(pos_dim);
  for (int i = 0; i < pos_dim; ++i)
  {
    pos_dists.emplace_back(p_lb[i], p_ub[i]);
  }

  // Precompute distributions for the rest (from robot bounds)
  std::vector<std::uniform_real_distribution<>> other_dists;
  other_dists.reserve(nx - pos_dim);
  for (int i = pos_dim; i < nx; ++i)
  {
    // other_dists.emplace_back(robot->x_lb[i], robot->x_ub[i]);
    other_dists.emplace_back(0, 0); // velocity part
  }

  // Special handling for unicycle heading
  const bool is_unicycle = (robot->name.find("unicycle") != std::string::npos) && (nx >= 3);
  std::uniform_real_distribution<> dist_theta(-M_PI, M_PI);

  std::vector<Eigen::VectorXd> samples;
  samples.reserve(N);

  for (int n = 0; n < N; ++n)
  {
    Eigen::VectorXd x(nx);

    // Sample positions from environment bounds
    for (int i = 0; i < pos_dim; ++i)
    {
      x[i] = pos_dists[i](gen);
    }

    // Sample other dimensions from robot bounds
    for (int i = pos_dim; i < nx; ++i)
    {
      x[i] = other_dists[i - pos_dim](gen);
    }

    // Override heading if unicycle
    if (is_unicycle)
    {
      x[2] = dist_theta(gen);
    }

    samples.push_back(std::move(x));
  }

  return samples;
}

void compute_heuristics(int N,
                        dynobench::Problem &problem,
                        Options_tdbastar &options_tdbastar,
                        size_t &robot_id,
                        ompl::NearestNeighbors<std::shared_ptr<AStarNode>> **heuristic_result)
{
  const bool use_non_counter_time = true;
  Options_tdbastar options_tdbastar_local = options_tdbastar;
  std::shared_ptr<dynobench::Model_robot>
      robot = dynobench::robot_factory(
          (problem.models_base_path + problem.robotTypes[robot_id] + ".yaml").c_str(),
          problem.p_lb, problem.p_ub);
  dynobench::Problem problem_local;
  problem_local.robotType = problem.robotTypes[robot_id];
  problem_local.start = problem.starts[robot_id];
  problem_local.goal = problem.goals[robot_id];
  problem_local.p_lb = problem.p_lb;
  problem_local.p_ub = problem.p_ub;
  problem_local.obstacles = problem.obstacles;
  problem_local.models_base_path = problem.models_base_path;
  problem_local.robotTypes.push_back(problem_local.robotType);
  problem_local.starts.push_back(problem_local.start);
  problem_local.goals.push_back(problem_local.goal);
  size_t robot_id_local = 0;
  bool debug = false;
  load_env(*robot, problem_local); // problem can be MRS
  // sample goal states to cover the env.
  std::vector<Eigen::VectorXd> free_samples;
  free_samples = sampleFree(N, problem_local.p_lb, problem_local.p_ub, robot);
  CHECK(options_tdbastar.motions_ptr,
        "motions should be loaded before calling dbastar");
  std::vector<Motion> &motions = *options_tdbastar.motions_ptr;
  auto check_motions = [&]
  {
    for (size_t idx = 0; idx < motions.size(); ++idx)
    {
      if (motions[idx].idx != idx)
      {
        return false;
      }
    }
    return true;
  };
  assert(check_motions());
  // Variables that will get updated during idbA* search
  bool finished = false;
  size_t it = 0;
  size_t num_solutions = 0;
  double non_counter_time = 0;
  bool solved_db = false;
  bool solved = false;
  // helper function to get time stamp since beginning of search (millisec.)
  auto start = std::chrono::steady_clock::now();
  auto get_time_stamp_ms = [&]
  {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };

  // take care of where to save
  ompl::NearestNeighbors<std::shared_ptr<AStarNode>> *T_n = nullptr;
  T_n = nigh_factory2<std::shared_ptr<AStarNode>>(
      problem_local.robotTypes[robot_id_local], robot);
  *heuristic_result = T_n;
  std::vector<std::shared_ptr<AStarNode>> all_nodes;
  for (auto &goal : free_samples)
  {
    problem_local.goal = goal;
    problem_local.goals.at(0) = goal;
    std::cout << "goal: " << std::endl;
    std::cout << problem_local.goal.format(dynobench::FMT) << std::endl;
    solved = false;
    solved_db = false;
    it = 0;
    finished = false;
    while (!finished)
    {
      // Choose the value of delta and number of primitives for the search
      if (it == 0)
      {
        options_tdbastar_local.delta = options_tdbastar.delta;
        options_tdbastar_local.max_motions = options_tdbastar.max_motions;
      }
      else
      {
        if (solved_db)
        {
          options_tdbastar_local.delta *= 0.9; // delta_rate
        }
        else
        {
          // basically do not reduce delta if I have not solved with db first
          // (because this often means that we just need more primitives)
          options_tdbastar_local.delta *= .9999;
        }
        std::cout << "delta: " << options_tdbastar_local.delta << std::endl;
        // We always add more primitives
        options_tdbastar_local.max_motions *= 1.5; // num_primitives_rate
        std::cout << "max motions: " << options_tdbastar_local.max_motions << std::endl;
      }

      double delta_cost = 0;
      CSTR_(delta_cost);
      options_tdbastar_local.maxCost = 1e8;
      // Run dbastar
      std::cout << "*** Running DB-astar ***" << std::endl;
      dynobench::Trajectory traj_db, traj;
      Out_info_tdb out_info_tdb;
      std::string id_db = gen_random(6);
      options_tdbastar_local.outFile = "/tmp/dynoplan/i_db_" + id_db + ".yaml";

      Stopwatch sw;
      // dbastar(problem_local, options_tdbastar_local, traj_db, out_info_tdb);
      tdbastar(problem_local, options_tdbastar, traj_db,
               /*constraints*/ {}, out_info_tdb, robot_id_local, /*reverse_search*/ false,
               nullptr, nullptr);
      non_counter_time += sw.elapsed_ms() - out_info_tdb.time_search;
      solved_db = out_info_tdb.solved;

      std::cout << "warning: using as time only the search!" << std::endl;
      traj_db.time_stamp =
          get_time_stamp_ms() - int(use_non_counter_time) * non_counter_time;

      if (out_info_tdb.solved)
      {
        if (debug)
        {
          // write trajectory to file for debugging
          std::string filename = "/tmp/dynoplan/i_traj_db.yaml";
          std::string filename_id = "/tmp/dynoplan/i_traj_db_" + id_db + ".yaml";
          std::cout << "saving traj to: " << filename << std::endl;
          std::cout << "saving traj to: " << filename_id << std::endl;
          create_dir_if_necessary(filename.c_str());
          create_dir_if_necessary(filename_id.c_str());
          std::ofstream out(filename_id);
          traj_db.to_yaml_format(out);
          std::filesystem::copy(
              filename_id, filename.c_str(),
              std::filesystem::copy_options::overwrite_existing);
        }
        // Start Trajectory optimization
        std::cout << "***Trajectory Optimization -- START ***" << std::endl;
        Result_opti result;
        Stopwatch stopwatch;
        Options_trajopt options_trajopt;
        trajectory_optimization(problem_local, traj_db, options_trajopt, traj, result);
        non_counter_time +=
            stopwatch.elapsed_ms() - std::stof(result.data.at("time_ddp_total"));
        std::cout << "***Trajectory Optimization -- DONE ***" << std::endl;

        traj.time_stamp =
            get_time_stamp_ms() - int(use_non_counter_time) * non_counter_time;
        if (traj.feasible)
        {
          num_solutions++;
          solved = true;

          std::cout << "we have a feasible solution! Cost: " << traj.cost
                    << std::endl;
          all_nodes.push_back(std::make_shared<AStarNode>());
          auto __node = all_nodes.back();
          __node->state_eig = goal;
          // only gScore is needed for forward search
          __node->gScore = traj.cost;
          T_n->add(__node);
          break;
          // update the g-value with cost and move to the next goal state
        }
        else
        {
          std::cout << "Trajectory optimization has failed!" << std::endl;
        }
      }
      else
      {
        std::cout << "Discrete search has failed!" << std::endl;
      }
      it++;
      if (it >= 5)
      {
        finished = true;
      }

      if (get_time_stamp_ms() / 1000. > /*timelimit*/ 60) // sec.
      {
        finished = true;
      }
    }
  }
}