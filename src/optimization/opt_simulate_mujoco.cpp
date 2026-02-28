#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include "dynobench/mujoco_quadrotors_payload.hpp"
#include "dynobench/mujoco_quadrotor.hpp"
#include "dynoplan/optimization/ocp.hpp"
#include "dynobench/dyno_macros.hpp"
#include "dynobench/general_utils.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>

using namespace dynoplan;
using dynobench::Model_MujocoQuadsPayload;
using dynobench::Model_MujocoQuad;

static void repaint_model_geoms(mjModel* m, float rr, float gg, float bb, float aa) {
    for (int i = 0; i < m->ngeom; ++i) {
        float* c = m->geom_rgba + 4*i;
        c[0] = rr;
        c[1] = gg;
        c[2] = bb;
        c[3] = aa;
    }
}

static void apply_camera_preset(mjvCamera& cam,
                                const Eigen::Vector3d& env_center,
                                const Eigen::Vector3d& env_size,
                                const std::string& camera_view) {
    // Center the camera on the environment
    cam.lookat[0] = env_center.x();
    cam.lookat[1] = env_center.y();
    cam.lookat[2] = env_center.z();

    // Distance heuristic from XY extent
    const double xy_diag   = std::sqrt(env_size.x()*env_size.x() + env_size.y()*env_size.y());
    const double base_dist = std::max(1.0, xy_diag * 0.65);

    if (camera_view == "top") {
        cam.azimuth   = 0;      // azimuth irrelevant for top
        cam.elevation = -90;    // straight down
        cam.distance  = std::max(env_size.x(), env_size.y()) * 0.8;
    } else if (camera_view == "front") {
        cam.azimuth   = 180;      // +x looking toward -x
        cam.elevation = -15;
        cam.distance  = base_dist;
    } else if (camera_view == "side") {
        cam.azimuth   = 90;     // +y looking toward -y
        cam.elevation = -15;
        cam.distance  = base_dist;
    } else if (camera_view == "diag") {
        cam.azimuth   = 202.5;     // 
        cam.elevation = -45;
        cam.distance  = base_dist;
    } else { // default / unknown
        cam.azimuth   = 45;
        cam.elevation = -35;
        cam.distance  = base_dist;
    }
}

void execute_simMujoco(std::string &env_file,
                       std::string &initial_guess_file,
                       dynobench::Trajectory &sol,
                       const std::string &dynobench_base,
                       const std::string &video_path,
                       const std::string &camera_view,
                       int num_repeats, bool view_ghost, bool feasible) {
        
    std::cout << " view ghost! "<<view_ghost << std::endl;

    // AUTO mode: if camera_view == "auto", write side/top/front with repeats
    if (camera_view == "auto") {
        std::string base = video_path;
        if (base.size() > 4 && base.substr(base.size()-4) == ".mp4")
            base = base.substr(0, base.size()-4);
        if (base.empty()) base = "out";
        std::vector<std::string> views = {"side","top","front","diag"};
        for (const auto& v : views) {
            std::string out = base + "_" + v + ".mp4";
            execute_simMujoco(env_file, initial_guess_file, sol, dynobench_base, out, v, num_repeats, view_ghost, feasible);
        }
        return;
    }


    std::string models_base_path = dynobench_base + "/models/";
    dynobench::Problem  problem(env_file.c_str());
    problem.models_base_path = models_base_path;
    dynobench::Trajectory init_guess;
    init_guess.read_from_yaml(initial_guess_file.c_str());

    auto base_live  = dynobench::robot_factory((models_base_path+problem.robotType+".yaml").c_str(),
            problem.p_lb, problem.p_ub);
    auto base_ghost = dynobench::robot_factory((models_base_path+problem.robotType+".yaml").c_str(),
            problem.p_lb, problem.p_ub);
    
    YAML::Node env = YAML::LoadFile(env_file);
    std::string robotName = env["robots"][0]["type"].as<std::string>();
    auto maxNode = env["environment"]["max"];
    auto minNode = env["environment"]["min"];

    Eigen::Vector3d env_max(maxNode[0].as<double>(),
                    maxNode[1].as<double>(),
                    maxNode[2].as<double>());

    Eigen::Vector3d env_min(minNode[0].as<double>(),
                    minNode[1].as<double>(),
                    minNode[2].as<double>());

    // center of environment
    Eigen::Vector3d env_center = 0.5 * (env_max + env_min);

    // size of environment
    Eigen::Vector3d env_size = env_max - env_min;
    bool is_payload = (startsWith(robotName, "mujocoquadspayload"));

    if (is_payload) {
        auto* live  = dynamic_cast<Model_MujocoQuadsPayload*>(base_live.get());
        auto* ghost = dynamic_cast<Model_MujocoQuadsPayload*>(base_ghost.get());
        if (!live || !ghost) throw std::runtime_error("Failed to cast to Model_MujocoQuadsPayload");

        // Ghost color & transparency
        if (view_ghost) {
            repaint_model_geoms(ghost->m, 1.0f, 0.0f, 0.0f, 0.5f);
        }

        if (!glfwInit()) throw std::runtime_error("GLFW init failed");
        GLFWwindow* win = glfwCreateWindow(1920, 1080, "MuJoCo video", nullptr, nullptr);
        if (!win) throw std::runtime_error("Failed to create GLFW window");
        glfwMakeContextCurrent(win);
        glfwSwapInterval(1);

        live->init_mujoco_viewer();
        if (view_ghost) {
            ghost->init_mujoco_viewer();
    
            for (int g = 0; g < 6; ++g) ghost->opt_.geomgroup[g] = 0;
            ghost->opt_.geomgroup[1] = 1;
            ghost->opt_.geomgroup[2] = 1;
            ghost->opt_.geomgroup[3] = 1;
            ghost->opt_.geomgroup[5] = 1;
        }

        // --- Camera setup ---
        // Read environment bounds
        auto maxNode = env["environment"]["max"];
        auto minNode = env["environment"]["min"];

        Eigen::Vector3d env_max(maxNode[0].as<double>(),
                                maxNode[1].as<double>(),
                                maxNode[2].as<double>());

        Eigen::Vector3d env_min(minNode[0].as<double>(),
                                minNode[1].as<double>(),
                                minNode[2].as<double>());

        // Read start & goal positions from YAML
        auto startNode = env["robots"][0]["start"];
        auto goalNode  = env["robots"][0]["goal"];

        Eigen::Vector3d start_pos(startNode[0].as<double>(),
                                startNode[1].as<double>(),
                                startNode[2].as<double>());

        Eigen::Vector3d goal_pos(goalNode[0].as<double>(),
                                goalNode[1].as<double>(),
                                goalNode[2].as<double>());

        // Expand environment bounds to include start and goal
        env_min = env_min.cwiseMin(start_pos).cwiseMin(goal_pos);
        env_max = env_max.cwiseMax(start_pos).cwiseMax(goal_pos);

        // Compute new center and size
        Eigen::Vector3d env_center = 0.5 * (env_max + env_min);
        Eigen::Vector3d env_size   = env_max - env_min;

        apply_camera_preset(live->cam_, env_center, env_size, camera_view);

        Eigen::VectorXd x_next(live->nx), x_live = problem.start;
        Eigen::VectorXd x_next_ghost(ghost->nx), x_ghost = problem.start;

        size_t T = init_guess.actions.size();
        size_t U = sol.actions.size();

        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        w = (w / 2) * 2;
        h = (h / 2) * 2;
        std::vector<unsigned char> rgb(w*h*3);
        std::vector<float> depth(w*h);

        std::string cmd = "ffmpeg -y -f rawvideo -pixel_format rgb24 "
            "-video_size " + std::to_string(w) + "x" + std::to_string(h) +
            " -framerate 60 -i - -vf \"vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
            "-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \"" + video_path + "\"";

        FILE* ffmpeg = popen(cmd.c_str(), "w");
        if (!ffmpeg) throw std::runtime_error("Failed to open FFmpeg pipe");
        
        int repeats = num_repeats;
        for (int r = 0; r < repeats; ++r) {
            Eigen::VectorXd x_next(live->nx), x_live = problem.start;
            if (view_ghost) {
                Eigen::VectorXd x_next_ghost(ghost->nx), x_ghost = problem.start;
            }

            for (size_t k = 0; k < T; ++k) {
                if (k < U)
                    if (feasible) {
                        live->step(x_next, x_live, sol.actions[k], live->ref_dt);
                        x_live.swap(x_next);
                    } else {
                        live->get_x0(sol.states[k]);
                    }
                if (view_ghost) {
                    ghost->get_x0(init_guess.states[k]);
                }
                
                mjv_updateScene(live->m, live->d, &live->opt_, nullptr, &live->cam_, mjCAT_ALL, &live->scn_);
                if (view_ghost) {
                    mjv_updateScene(ghost->m, ghost->d, &ghost->opt_, nullptr, &live->cam_, mjCAT_ALL, &ghost->scn_);
                    mjv_addGeoms(ghost->m, ghost->d, &ghost->opt_, nullptr, mjCAT_ALL, &live->scn_);
                }
                mjr_render({0, 0, w, h}, &live->scn_, &live->con_);

                mjr_readPixels(rgb.data(), depth.data(), {0, 0, w, h}, &live->con_);
                fwrite(rgb.data(), 3, w*h, ffmpeg);

                glfwSwapBuffers(win);
                glfwPollEvents();
            }
        }
        pclose(ffmpeg);
        glfwTerminate();
    }
    else {
        auto* live  = dynamic_cast<Model_MujocoQuad*>(base_live.get());
        auto* ghost = dynamic_cast<Model_MujocoQuad*>(base_ghost.get());
        if (!live || !ghost) throw std::runtime_error("Failed to cast to Model_MujocoQuad");

        repaint_model_geoms(ghost->m, 1.0f, 0.0f, 0.0f, 0.5f);

        if (!glfwInit()) throw std::runtime_error("GLFW init failed");
        GLFWwindow* win = glfwCreateWindow(1920, 1080, "MuJoCo video", nullptr, nullptr);
        if (!win) throw std::runtime_error("Failed to create GLFW window");
        glfwMakeContextCurrent(win);
        glfwSwapInterval(1);

        live->init_mujoco_viewer();
        if (view_ghost) {
            ghost->init_mujoco_viewer();
            for (int g = 0; g < 6; ++g) ghost->opt_.geomgroup[g] = 0;
            ghost->opt_.geomgroup[1] = 1;
            ghost->opt_.geomgroup[2] = 1;
            ghost->opt_.geomgroup[3] = 1;
            ghost->opt_.geomgroup[5] = 1;
        }

        // --- Camera setup ---
        // Read environment bounds
        auto maxNode = env["environment"]["max"];
        auto minNode = env["environment"]["min"];

        Eigen::Vector3d env_max(maxNode[0].as<double>(),
                                maxNode[1].as<double>(),
                                maxNode[2].as<double>());

        Eigen::Vector3d env_min(minNode[0].as<double>(),
                                minNode[1].as<double>(),
                                minNode[2].as<double>());

        // Read start & goal positions from YAML
        auto startNode = env["robots"][0]["start"];
        auto goalNode  = env["robots"][0]["goal"];

        Eigen::Vector3d start_pos(startNode[0].as<double>(),
                                startNode[1].as<double>(),
                                startNode[2].as<double>());

        Eigen::Vector3d goal_pos(goalNode[0].as<double>(),
                                goalNode[1].as<double>(),
                                goalNode[2].as<double>());

        // Expand environment bounds to include start and goal
        env_min = env_min.cwiseMin(start_pos).cwiseMin(goal_pos);
        env_max = env_max.cwiseMax(start_pos).cwiseMax(goal_pos);

        // Compute new center and size
        Eigen::Vector3d env_center = 0.5 * (env_max + env_min);
        Eigen::Vector3d env_size   = env_max - env_min;

        apply_camera_preset(live->cam_, env_center, env_size, camera_view);

        size_t T = init_guess.actions.size();
        size_t U = sol.actions.size();

        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        w = (w / 2) * 2;
        h = (h / 2) * 2;
        std::vector<unsigned char> rgb(w*h*3);
        std::vector<float> depth(w*h);

        std::string cmd = "ffmpeg -y -f rawvideo -pixel_format rgb24 "
            "-video_size " + std::to_string(w) + "x" + std::to_string(h) +
            " -framerate 60 -i - -vf \"vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
            "-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \"" + video_path + "\"";

        FILE* ffmpeg = popen(cmd.c_str(), "w");
        if (!ffmpeg) throw std::runtime_error("Failed to open FFmpeg pipe");
        int repeats = num_repeats;
        for (int r = 0; r < repeats; ++r) {
            Eigen::VectorXd x_next(live->nx), x_live = problem.start;
            if (view_ghost) {
                Eigen::VectorXd x_next_ghost(ghost->nx), x_ghost = problem.start;
            }

            for (size_t k = 0; k < T; ++k) {
                if (k < U)
                    if (feasible) {
                        live->step(x_next, x_live, sol.actions[k], live->ref_dt);
                        x_live.swap(x_next);
                    } else {
                        live->get_x0(sol.states[k]);
                    }
                if (view_ghost) {
                    ghost->get_x0(init_guess.states[k]);
                }

                mjv_updateScene(live->m, live->d, &live->opt_, nullptr, &live->cam_, mjCAT_ALL, &live->scn_);
                if (view_ghost) {
                    mjv_updateScene(ghost->m, ghost->d, &ghost->opt_, nullptr, &live->cam_, mjCAT_ALL, &ghost->scn_);
                    mjv_addGeoms(ghost->m, ghost->d, &ghost->opt_, nullptr, mjCAT_ALL, &live->scn_);
                }
                mjr_render({0, 0, w, h}, &live->scn_, &live->con_);
                mjr_readPixels(rgb.data(), depth.data(), {0, 0, w, h}, &live->con_);
                fwrite(rgb.data(), 3, w*h, ffmpeg);

                glfwSwapBuffers(win);
                glfwPollEvents();
            }
        }
        pclose(ffmpeg);
        glfwTerminate();
    }
}


bool execute_optMujoco(std::string &env_file,
                       std::string &init_file,
                       std::string &results_file,
                       std::string &output_file_anytime,
                       dynobench::Trajectory &sol,
                       const std::string &dynobench_base,
                       bool sum_robots_cost, dynobench::Trajectory &sol_broken, std::string cfg_file) 
{
    std::string models_base_path = dynobench_base + "/models/";
    Result_opti result;
    Options_trajopt options_trajopt;
    if (cfg_file == "") {
        options_trajopt.solver_id = 1;
        options_trajopt.max_iter = 100;
        options_trajopt.noise_level = 1e-4;
        options_trajopt.collision_weight = 250;
        options_trajopt.weight_goal = 600.;
        options_trajopt.time_ref = 0.5;
        options_trajopt.time_weight = 0.7;
    } else { 
        options_trajopt.read_from_yaml(cfg_file.c_str());
    }
    dynobench::Problem problem(env_file.c_str());
    problem.models_base_path = models_base_path;
    dynobench::Trajectory init_guess;
    init_guess.read_from_yaml(init_file.c_str());

    std::cout << "optimizing trajectory..." << std::endl;
    trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
    sol_broken.states = result.xs_out;
    sol_broken.actions = result.us_out;


    if (result.feasible) {
        std::cout << "Optimization done. Results in: " << results_file << std::endl;
        return true;
    } else {
        std::cout << "Optimization failed." << std::endl;
        return false;
    }
}



// bool execute_optILMujoco(std::string &env_file,
//                        std::string &init_file,
//                        std::string &results_file,
//                        std::string &output_file_anytime,
//                        dynobench::Trajectory &sol,
//                        const std::string &dynobench_base,
//                        bool sum_robots_cost, dynobench::Trajectory &sol_broken, std::string cfg_file, const size_t N_opt) 
// {
//     std::string models_base_path = dynobench_base + "/models/";
//     Result_opti result;
//     Options_trajopt options_trajopt;
//     if (cfg_file == "") {
//         options_trajopt.solver_id = 1;
//         options_trajopt.max_iter = 100;
//         options_trajopt.noise_level = 1e-4;
//         options_trajopt.collision_weight = 250;
//         options_trajopt.weight_goal = 600.;
//         options_trajopt.time_ref = 0.5;
//         options_trajopt.time_weight = 0.7;
//     } else { 
//         options_trajopt.read_from_yaml(cfg_file.c_str());
//     }
//     dynobench::Problem problem(env_file.c_str());
//     problem.models_base_path = models_base_path;
//     dynobench::Trajectory init_guess;
//     init_guess.read_from_yaml(init_file.c_str());
//     while (init_guess.states.size() < N_opt+1) {
//         init_guess.states.push_back(init_guess.states.back());
//     }
    
//     if (init_guess.states.size() > N_opt+1) {
//         init_guess.states.resize(N_opt+1);
//     }
//     while (init_guess.actions.size() < N_opt) {
//         init_guess.actions.push_back(init_guess.actions.back());
//     }
//     if (init_guess.actions.size() > N_opt) {
//         init_guess.actions.resize(N_opt);
//     }    
//     std::cout << "optimizing trajectory..." << std::endl;
//     optimize_N_steps(problem, init_guess, options_trajopt, sol, result);
//     sol_broken.states = result.xs_out;
//     sol_broken.actions = result.us_out;


//     if (result.feasible) {
//         std::cout << "Optimization done. Results in: " << results_file << std::endl;
//         return true;
//     } else {
//         sol.states = result.xs_out;
//         sol.actions = result.us_out;
//         std::cout << "Optimization failed." << std::endl;
//         return false;
//     }
// }

bool execute_optILMujoco(std::string &env_file,
                       std::string &init_file,
                       std::string &results_file,
                       std::string &output_file_anytime,
                       dynobench::Trajectory &sol,
                       const std::string &dynobench_base,
                       bool sum_robots_cost, dynobench::Trajectory &sol_broken,
                       std::string cfg_file, const size_t N_opt) 
{
    std::string models_base_path = dynobench_base + "/models/";
    Result_opti result;
    Options_trajopt options_trajopt;
    if (cfg_file == "") {
        options_trajopt.solver_id = 1;
        options_trajopt.max_iter = 100;
        options_trajopt.noise_level = 1e-4;
        options_trajopt.collision_weight = 250;
        options_trajopt.weight_goal = 600.;
        options_trajopt.time_ref = 0.5;
        options_trajopt.time_weight = 0.7;
    } else { 
        options_trajopt.read_from_yaml(cfg_file.c_str());
    }

    dynobench::Problem problem(env_file.c_str());
    problem.models_base_path = models_base_path;

    dynobench::Trajectory init_guess;
    init_guess.read_from_yaml(init_file.c_str());

    // while (init_guess.states.size() < N_opt + 1) {
    //     init_guess.states.push_back(init_guess.states.back());
    // }
    // if (init_guess.states.size() > N_opt + 1) {
    //     init_guess.states.resize(N_opt + 1);
    // }

    // while (init_guess.actions.size() < N_opt) {
    //     init_guess.actions.push_back(init_guess.actions.back());
    // }
    // if (init_guess.actions.size() > N_opt) {
    //     init_guess.actions.resize(N_opt);
    // }

    std::cout << "optimizing trajectory..." << std::endl;
    optimize_N_steps(problem, init_guess, options_trajopt, sol, result);

    sol_broken.states = result.xs_out;
    sol_broken.actions = result.us_out;

    if (result.feasible) {
        std::cout << "Optimization done. Results in: " << results_file << std::endl;
        return true;
    } else {
        std::cout << "Optimization failed. Falling back to first 2 steps." << std::endl;

        // ---- fallback: keep first 2 steps (1 actions, 2 states) if available ----
        const size_t K = 1;

        if (result.xs_out.size() >= K + 1 && result.us_out.size() >= K) {
            sol.states.assign(result.xs_out.begin(), result.xs_out.begin() + (K + 1));
            sol.actions.assign(result.us_out.begin(), result.us_out.begin() + K);

            // keep these consistent (optional but nice)
            sol.start = problem.start;
            sol.goal  = problem.goal;
            // sol.cost  = sol.actions.size() * model.ref_dt; // if you have ref_dt here; otherwise omit
            sol.feasible = false; // it's still an infeasible solution overall
        } else {
            // Not enough output; last-resort: just keep init guess first 2 steps
            if (init_guess.states.size() >= K + 1 && init_guess.actions.size() >= K) {
                sol.states.assign(init_guess.states.begin(), init_guess.states.begin() + (K + 1));
                sol.actions.assign(init_guess.actions.begin(), init_guess.actions.begin() + K);
                sol.feasible = false;
            } else {
                // absolute minimum: just keep x0
                sol.states = {problem.start};
                sol.actions.clear();
                sol.feasible = false;
            }
        }

        return true;
    }
}
