// ---------------------------------------------------------------------------
// main_simulate_mujoco.cpp   – interactive viewer + ghost overlay
// ---------------------------------------------------------------------------
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include "crocoddyl/core/utils/timer.hpp"
#include "dynoplan/optimization/ocp.hpp"
#include "dynobench/mujoco_quadrotors_payload.hpp"

using namespace dynoplan;
using dynobench::Model_MujocoQuadsPayload;

// The interactive part was written by ChatGPT by providing it with a python piece
// of code written by me and asking it to rewrite it cpp.
// ----------------------------------------------------------------------------
// globals for mouse/keyboard state
// ----------------------------------------------------------------------------
static bool   g_button_left  = false;
static bool   g_button_right = false;
static double g_last_x       = 0.0;
static double g_last_y       = 0.0;

// ----------------------------------------------------------------------------
// grab our robot pointer
// ----------------------------------------------------------------------------
static Model_MujocoQuadsPayload* rob(GLFWwindow* w) {
    return static_cast<Model_MujocoQuadsPayload*>(glfwGetWindowUserPointer(w));
}

// ----------------------------------------------------------------------------
// keyboard: ESC to quit
// ----------------------------------------------------------------------------
static void keyboard_cb(GLFWwindow* win, int key, int /*scancode*/, int action, int /*mods*/) {
    if ((action == GLFW_PRESS || action == GLFW_REPEAT) && key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(win, true);
    }
}

// ----------------------------------------------------------------------------
// mouse button: track left/right
// ----------------------------------------------------------------------------
static void mouse_button_cb(GLFWwindow* win, int btn, int act, int /*mods*/) {
    if (act == GLFW_PRESS) {
        g_button_left  = (btn == GLFW_MOUSE_BUTTON_LEFT);
        g_button_right = (btn == GLFW_MOUSE_BUTTON_RIGHT);
        glfwGetCursorPos(win, &g_last_x, &g_last_y);
    }
    else if (act == GLFW_RELEASE) {
        g_button_left = g_button_right = false;
    }
}

// ----------------------------------------------------------------------------
// cursor move: rotate/zoom
// ----------------------------------------------------------------------------
static void cursor_pos_cb(GLFWwindow* win, double x, double y) {
    double dx = x - g_last_x;
    double dy = y - g_last_y;
    g_last_x = x;
    g_last_y = y;

    auto* r = rob(win);
    if (!r) return;

    if (g_button_left) {
        mjv_moveCamera(r->m, mjMOUSE_ROTATE_H, dx/100.0, dy/100.0, &r->scn_, &r->cam_);
    }
    else if (g_button_right) {
        mjv_moveCamera(r->m, mjMOUSE_ZOOM, dx/100.0, dy/100.0, &r->scn_, &r->cam_);
    }
}

// ----------------------------------------------------------------------------
// scroll wheel: always zoom
// ----------------------------------------------------------------------------
static void scroll_cb(GLFWwindow* win, double /*xoff*/, double yoff) {
    auto* r = rob(win);
    if (!r) return;
    mjv_moveCamera(r->m, mjMOUSE_ZOOM, 0.0, yoff/10.0, &r->scn_, &r->cam_);
}

// ----------------------------------------------------------------------------
// repaint all geoms in a model to (r,g,b,a)
// ----------------------------------------------------------------------------
static void repaint_model_geoms(mjModel* m, float rr, float gg, float bb, float aa) {
    for (int i = 0; i < m->ngeom; ++i) {
        float* c = m->geom_rgba + 4*i;
        c[0]=rr; c[1]=gg; c[2]=bb; c[3]=aa;
    }
}

int main(int argc, const char** argv) {
    std::string env_file, init_file, models_base_path, results_file;
    std::string cfg_file = "";

    po::options_description desc("Options");
    set_from_boostop(desc, VAR_WITH_NAME(env_file));
    set_from_boostop(desc, VAR_WITH_NAME(init_file));
    set_from_boostop(desc, VAR_WITH_NAME(cfg_file));
    set_from_boostop(desc, VAR_WITH_NAME(results_file));
    set_from_boostop(desc, VAR_WITH_NAME(models_base_path));
    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.count("help")) { std::cout<<desc<<"\n"; return 0; }
    } catch (const po::error& e) {
        std::cerr<<e.what()<<"\n"<<desc<<"\n";
        return 1;
    }
    
    Result_opti result;
    Options_trajopt options_trajopt;
    // options_trajopt.solver_id = 1;
    // options_trajopt.collision_weight = 100.;
    // options_trajopt.weight_goal = 1000.;
    // options_trajopt.max_iter = 100;
    if (cfg_file != "") {
    options_trajopt.read_from_yaml(cfg_file.c_str());
    }


    // 2) load problem + optimize trajectory
    dynobench::Problem  problem(env_file.c_str());
    problem.models_base_path = models_base_path;
    dynobench::Trajectory init_guess;
    init_guess.read_from_yaml(init_file.c_str());
    dynobench::Trajectory traj_out;
    trajectory_optimization(problem, init_guess, options_trajopt, traj_out, result);

    CSTR_(results_file);
    std::ofstream results(results_file);

    results << "time_stamp: " << get_time_stamp() << std::endl;
    results << "env_file: " << env_file << std::endl;
    results << "init_file: " << init_file << std::endl;
    results << "cfg_file: " << cfg_file << std::endl;
    results << "results_file: " << results_file << std::endl;
    results << "options trajopt:" << std::endl;

    options_trajopt.print(results, "  ");
    result.write_yaml(results);

    // saving only the trajectory
    std::string file = results_file + ".trajopt.yaml";
    std::ofstream out(file);
    traj_out.to_yaml_format(out);


    std::cout << "result: " << result.feasible << std::endl;
    bool simulate = true;
    if (simulate) {    
        // ------------------------ simulator ---------------------//
        // 3) create live + ghost copies
        auto base_live  = dynobench::robot_factory(
            (models_base_path+problem.robotType+".yaml").c_str(),
            problem.p_lb, problem.p_ub);
        auto base_ghost = dynobench::robot_factory(
            (models_base_path+problem.robotType+".yaml").c_str(),
            problem.p_lb, problem.p_ub);

        auto* live  = dynamic_cast<Model_MujocoQuadsPayload*>(base_live.get());
        auto* ghost = dynamic_cast<Model_MujocoQuadsPayload*>(base_ghost.get());
        // ghost->m->opt.disableflags |= 1 << mjDSBL_CONTACT;
        // tint ghost
        repaint_model_geoms(ghost->m, 1.0f, 0.0f, 0.0f, 0.5f);

        // 4) setup env + GLFW + viewer
        // load_env(*live,  problem);
        // load_env(*ghost, problem);

        if (!glfwInit()) {
            std::cerr<<"GLFW init failed\n";
            return 1;
        }
        GLFWwindow* win = glfwCreateWindow(1200, 1200, "MuJoCo viewer", nullptr, nullptr);
        if (!win) {
            std::cerr<<"No window\n";
            glfwTerminate();
            return 1;
        }
        glfwMakeContextCurrent(win);
        glfwSwapInterval(1);

        // install callbacks
        glfwSetWindowUserPointer(win, live);
        glfwSetKeyCallback       (win, keyboard_cb);
        glfwSetMouseButtonCallback(win, mouse_button_cb);
        glfwSetCursorPosCallback (win, cursor_pos_cb);
        glfwSetScrollCallback    (win, scroll_cb);

        // init viewers
        live ->init_mujoco_viewer();
        ghost->init_mujoco_viewer();

        // hide ghost background geoms (group 0), show only 1,2,3,5:
        for (int g=0; g<6; ++g) ghost->opt_.geomgroup[g] = 0;
        ghost->opt_.geomgroup[1] = 1;
        ghost->opt_.geomgroup[2] = 1;
        ghost->opt_.geomgroup[3] = 1;
        ghost->opt_.geomgroup[5] = 1;
        std::cout << "states 0: " << " : " << traj_out.states.size() << std::endl;
        std::cout << "result: " << result.feasible<< std::endl;
      
        // 5) loop
        const size_t T = init_guess.actions.size();
        const size_t U = result.us_out.size();
        while (!glfwWindowShouldClose(win)) {
            Eigen::VectorXd x_next(live->nx), x_live = problem.start;
            Eigen::VectorXd x_next_ghost(ghost->nx), x_ghost = problem.start;
            for (size_t k=0; k<T; ++k) {
                if(k < U) {
                    if (result.feasible) {
                        live->step(x_next, x_live, traj_out.actions[k], live->ref_dt);
                        x_live.swap(x_next);
                    } else {
                        // std::cout << "optimization failed!" << std::endl;
                        live->step(x_next, x_live, result.us_out[k], live->ref_dt);
                        x_live.swap(x_next);
                    }
                }
                ghost->step(x_next_ghost, x_ghost, init_guess.actions[k], ghost->ref_dt);
                x_ghost.swap(x_next_ghost);
                
                // merge + render
                int w,h; glfwGetFramebufferSize(win, &w, &h);
                mjv_updateScene(live->m,  live->d,  &live->opt_,  nullptr, &live->cam_,  mjCAT_ALL, &live->scn_);
                mjv_updateScene(ghost->m, ghost->d, &ghost->opt_, nullptr, &live->cam_, mjCAT_ALL, &ghost->scn_);
                mjv_addGeoms   (ghost->m, ghost->d, &ghost->opt_, nullptr, mjCAT_ALL, &live->scn_);
                mjr_render({0,0,w,h}, &live->scn_, &live->con_);

                glfwSwapBuffers(win);
                glfwPollEvents();
            }
        }

        glfwTerminate();
        return 0;
    }
}
