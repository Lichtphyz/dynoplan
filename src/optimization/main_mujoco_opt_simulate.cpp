#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>

#include <boost/program_options.hpp>
#include "dynoplan/optimization/opt_simulate_mujoco.hpp"                 


namespace po = boost::program_options;
namespace fs = std::filesystem;

using namespace dynobench;

int main(int argc, char** argv) {
  try {
    // --- CLI ---
    std::string env_file, init_file, dynobench_base, results_path, cfg_file;
    bool do_optimize = false;
    bool do_visualize = false;
    bool view_init = false;

    po::options_description desc("main_mujoco_opt_simulate options");
    desc.add_options()
      ("help,h", "Show help")
      ("env_file",       po::value<std::string>(&env_file), "Environment YAML")
      ("init_file",      po::value<std::string>(&init_file), "Initial guess YAML")
      ("results_path",   po::value<std::string>(&results_path)->default_value("../result_opt.yaml"), "Path to save optimized solution YAML (written only if -o succeeds)")
      ("dynobench_base", po::value<std::string>(&dynobench_base), "DynoBench base directory (contains models/)")
      ("cfg_file",       po::value<std::string>(&cfg_file)->default_value(""), "optimization parameters, see optimization/options.hpp")
      ("optimize,o",     po::bool_switch(&do_optimize)->default_value(false), "Optimize the problem")
      ("visualize,v",    po::bool_switch(&do_visualize)->default_value(false), "Save videos; does not require -o")
      ("view_init,i",    po::bool_switch(&view_init)->default_value(false), "view the initial guess, does not require -o")
      ("views",           po::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{"auto"}, "auto"), "Views: 'auto' or list of side top front diag")
      ("repeats",         po::value<int>()->default_value(2), "Number of repeats inside each video (default 2)")
      ("video_prefix",    po::value<std::string>()->default_value(""), "Optional video base; outputs base_<view>.mp4")
      ;
      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);
      
      if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
      }
      
    // If neither optimize nor visualize requested, do nothing (as specified)
    if (!do_optimize && !do_visualize) {
      std::cout << "Nothing to do: neither -o/--optimize nor -v/--visualize specified.\n";
      std::cout << "Run with --help to see options.\n";
      return 0;
    }

    // Validate required inputs when actions are requested
    auto need = [&](const char* name, const std::string& val) {
      if (val.empty()) {
        throw std::runtime_error(std::string("Missing required option --") + name);
      }
    };
    need("env_file", env_file);
    need("init_file", init_file);
    need("dynobench_base", dynobench_base);

    // Normalize paths
    auto as_abs = [](const std::string& p)->std::string {
      try { return fs::weakly_canonical(fs::path(p)).string(); }
      catch(...) { return p; }
    };
    env_file       = as_abs(env_file);
    init_file      = as_abs(init_file);
    dynobench_base = as_abs(dynobench_base);

    dynobench::Trajectory sol, sol_broken;
    bool feasible = false;
    // --- OPTIMIZATION ---
    if (do_optimize) {
      std::string anytime_dummy = ""; // unused
      
      std::cout << "Optimizing and saving in: " << results_path << std::endl;
      feasible = execute_optMujoco(env_file, init_file, results_path, anytime_dummy,
                                  sol, dynobench_base, /*sum_robots_cost*/ false, sol_broken, cfg_file);
      if (!feasible) {
        std::cerr << "Optimization failed.\n";
        sol_broken.to_yaml_format(results_path.c_str());
        if (!do_visualize) return 2;
      } else {
        sol.to_yaml_format(results_path.c_str());
      }
    }


    // --- VISUALIZATION ---
    if (do_visualize) {
    // Choose what to visualize:
    // If optimizing, visualize the "broken" trajectory; else visualize the initial guess.
    dynobench::Trajectory sol_to_show;
    if (do_optimize) {
      if (feasible) {
        sol_to_show = sol;
        std::cout << "Visualizing the optimized trajectory.\n";
      } else {
        sol_to_show = sol_broken;
        std::cout << "Visualizing the failed to optimize trajectory.\n";
      }
    } else {
      std::cout << "Visualizing the INITIAL GUESS (no optimization).\n";
      dynobench::Trajectory init_guess;
      init_guess.read_from_yaml(init_file.c_str());
      sol_to_show = init_guess;
    }

      // Video base name:
      std::string base = vm["video_prefix"].as<std::string>();
      if (base.empty()) {
        fs::path init_p(init_file);
        base = (init_p.parent_path() / (init_p.stem().string() + "_viz")).string();
      }
      bool view_ghost = view_init;
      if (!do_optimize && !view_init && do_visualize) {
        // Visualize a feasible solution and dont show the initial guess
        sol_to_show.read_from_yaml(results_path.c_str());
        feasible = true;
      }
      // Views and repeats
      auto views = vm["views"].as<std::vector<std::string>>();
      int repeats = vm["repeats"].as<int>();

      if (views.size() == 1 && views[0] == "auto") {
        // Use AUTO fan-out inside execute_simMujoco
        std::string video_path = base + ".mp4"; // AUTO strips .mp4 and appends suffixes
        std::cout << "Writing videos to base: " << base << "_{side,top,front,diag}.mp4 (repeats=" << repeats << ")\n";
        execute_simMujoco(env_file, init_file, sol_to_show, dynobench_base,
                          video_path, "auto", repeats, view_ghost, feasible);
      } else {
        // Explicit list of views
        for (const auto& v : views) {
          std::string out = base + "_" + v + ".mp4";
          std::cout << "Writing: " << out << " (repeats=" << repeats << ")\n";
          execute_simMujoco(env_file, init_file, sol_to_show, dynobench_base,
                            out, v, repeats, view_ghost, feasible);
        }
      }
    }

    std::cout << "Done.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
