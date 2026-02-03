#include "dynoplan/optimization/ocp.hpp"
#include "dynobench/dyno_macros.hpp"
#include "dynobench/math_utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <filesystem>

#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"

#include "dynobench/general_utils.hpp"
#include "dynobench/joint_robot.hpp"
#include "dynobench/quadrotor_payload_n.hpp"
#include "dynobench/robot_models.hpp"
#include "dynoplan/optimization/croco_models.hpp"

using vstr = std::vector<std::string>;
using V2d = Eigen::Vector2d;
using V3d = Eigen::Vector3d;
using V4d = Eigen::Vector4d;
using Vxd = Eigen::VectorXd;
using V1d = Eigen::Matrix<double, 1, 1>;

using dynobench::Trajectory;

namespace dynoplan {

using dynobench::enforce_bounds;
using dynobench::FMT;

class CallVerboseDyno : public crocoddyl::CallbackAbstract {
public:
  using Traj =
      std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>;
  std::vector<Traj> trajs;

  explicit CallVerboseDyno() = default;
  ~CallVerboseDyno() override = default;

  void operator()(crocoddyl::SolverAbstract &solver) override {
    std::cout << "adding trajectory" << std::endl;
    trajs.push_back(std::make_pair(solver.get_xs(), solver.get_us()));
  }

  void store() {
    std::string timestamp = get_time_stamp();
    std::string folder = "iterations/" + timestamp;
    std::filesystem::create_directories(folder);

    for (size_t i = 0; i < trajs.size(); i++) {
      auto &traj = trajs.at(i);
      dynobench::Trajectory tt;
      tt.states = traj.first;
      tt.actions = traj.second;

      std::stringstream ss;
      ss << std::setfill('0') << std::setw(4) << i;
      std::ofstream out(folder + "/it" + ss.str() + ".yaml");
      tt.to_yaml_format(out);
    }
  }
};

#if 0
void read_from_file(File_parser_inout &inout) {


  YAML::Node init;
  if (inout.init_guess != "") {
    init = load_yaml_safe(inout.init_guess);
  }
  YAML::Node env = load_yaml_safe(inout.env_file);

  if (!env["robots"]) {
    CHECK(false, AT);
    // ...
  }

  Problem problem;
  problem.read_from_yaml(inout.env_file.c_str());

  inout.name = problem.robotType;

  Vxd uzero;

  // if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0",
  //               "car_first_order_with_1_trailers_0"},
  //          inout.name))
  //   dt = .1;
  // else if (__in(vstr{"quad2d", "quadrotor_0", "acrobot"}, inout.name))
  //   dt = .01;
  // else
  //   CHECK(false, AT);

  // load the yaml file

  std::string base_path = "../models/";
  std::string suffix = ".yaml";
  inout.robot_model_file = base_path + inout.name + suffix;
  std::cout << "loading file: " << inout.robot_model_file << std::endl;
  YAML::Node robot_model = load_yaml_safe(inout.robot_model_file);
  CHECK(robot_model["dt"], AT);
  inout.dt = robot_model["dt"].as<double>();
  std::cout << STR_(inout.dt) << std::endl;

  // // load the collision checker
  // if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0",
  //               "car_first_order_with_1_trailers_0", "quad2d",
  //               "quadrotor_0"},
  //          inout.name)) {
  //   inout.cl = mk<CollisionChecker>();
  //   inout.cl->load(inout.env_file);
  // } else {
  //   std::cout << "this robot doesn't have collision checking " << std::endl;
  //   inout.cl = nullptr;
  // }

  std::vector<std::vector<double>> states;
  // std::vector<Vxd> xs_init;
  // std::vector<Vxd> us_init;

  size_t N;
  std::vector<std::vector<double>> actions;

  inout.start = problem.start;
  inout.goal = problem.goal;

  if (inout.xs.size() && inout.us.size()) {
    std::cout << "using xs and us as init guess " << std::endl;
  } else if (inout.init_guess != "") {
    if (!inout.new_format) {

      CHECK(init["result"], AT);
      CHECK(init["result"][0], AT);
      get_states_and_actions(init["result"][0], inout.xs, inout.us);

    } else {

      std::vector<Vxd> _xs_init;
      std::vector<Vxd> _us_init;

      CHECK(init["result2"], AT);
      CHECK(init["result2"][0], AT);

      get_states_and_actions(init["result2"][0], _xs_init, _us_init);

      std::cout << "Reading results in the new format " << std::endl;

      std::vector<double> _times;

      for (const auto &time : init["result2"][0]["times"]) {
        _times.push_back(time.as<double>());
      }

      // 0 ... 3.5
      // dt is 1.
      // 0 1 2 3 4

      // we use floor in the time to be more agressive
      std::cout << STR_(inout.dt) << std::endl;

      double total_time = _times.back();

      size_t num_time_steps = std::ceil(total_time / inout.dt);

      Vxd times = Vxd::Map(_times.data(), _times.size());

      std::vector<Vxd> xs_init_new;
      std::vector<Vxd> us_init_new;

      size_t nx = states.at(0).size();
      size_t nu = actions.at(0).size();

      auto ts =
          Vxd::LinSpaced(num_time_steps + 1, 0, num_time_steps * inout.dt);

      std::cout << "taking samples at " << ts.transpose() << std::endl;

      for (size_t ti = 0; ti < num_time_steps + 1; ti++) {
        Vxd xout(nx);
        Vxd Jout(nx);

        if (ts(ti) > times.tail(1)(0))
          xout = _xs_init.back();
        else
          dynobench::linearInterpolation(times, _xs_init, ts(ti), xout, Jout);
        xs_init_new.push_back(xout);
      }

      auto times_u = times.head(times.size() - 1);
      for (size_t ti = 0; ti < num_time_steps; ti++) {
        Vxd uout(nu);
        Vxd Jout(nu);
        if (ts(ti) > times_u.tail(1)(0))
          uout = _us_init.back();
        else
          linearInterpolation(times_u, _us_init, ts(ti), uout, Jout);
        us_init_new.push_back(uout);
      }

      N = num_time_steps;

      std::cout << "N: " << N << std::endl;
      std::cout << "us:  " << us_init_new.size() << std::endl;
      std::cout << "xs: " << xs_init_new.size() << std::endl;

      inout.xs = xs_init_new;
      inout.us = us_init_new;

      std::ofstream debug_file("debug.txt");

      for (auto &x : inout.xs) {
        debug_file << x.format(FMT) << std::endl;
      }
      debug_file << "---" << std::endl;

      for (auto &u : inout.us) {
        debug_file << u.format(FMT) << std::endl;
      }
    }
  } else {
    std::cout << "Warning: "
              << "no init guess " << std::endl;
    std::cout << "using T: " << inout.T << " start " << inout.start.format(FMT)
              << std::endl;

    Vxd x0 = inout.start;

    if (options_trajopt.ref_x0) {
      std::cout << "Warning: using a ref x0, instead of start" << std::endl;

      if (startsWith(inout.name, "unicycle1")) {
        ;
      } else if (startsWith(inout.name, "unicycle2")) {
        x0.segment(3, 2) = Vxd::Zero(2);
        ;
      } else if (startsWith(inout.name, "quad2d")) {
        x0.segment(2, 4) = Vxd::Zero(4);
      } else if (startsWith(inout.name, "quadrotor_0")) {
        x0.segment(3, 4) << 0, 0, 0, 1;
        x0.segment(7, 6) = Vxd::Zero(6);
      } else if (startsWith(inout.name, "acrobot")) {
        x0.segment(2, 2) = Vxd::Zero(2);
      } else if (startsWith(inout.name, "unicycle_first_order_0")) {
        ;
      } else if (startsWith(inout.name, "quad3d")) {
        x0.segment(3, 3).setZero(); // quaternion imaginary
        x0.segment(7, 6).setZero(); // velocities
      } else {
        ERROR_WITH_INFO("not implemented");
      }
      inout.xs = std::vector<Vxd>(inout.T + 1, x0);
    }
    // lets use slerp.
    else if (options_trajopt.interp) {

      if (inout.name == "quadrotor_0") {

        auto x_s = inout.start.segment(0, 3);
        auto x_g = inout.goal.segment(0, 3);
        // auto  x_g = Eigen::Quaterniond(inout.goal.segment(3,4));

        auto q_s = Eigen::Quaterniond(inout.start.segment<4>(3));
        auto q_g = Eigen::Quaterniond(inout.goal.segment<4>(3));

        inout.xs = std::vector<Vxd>(inout.T + 1, x0);

        for (size_t i = 0; i < inout.xs.size(); i++) {

          double dt = double(i) / (inout.xs.size() - 1);
          auto q_ = q_s.slerp(dt, q_g);
          auto x = x_s + dt * (x_g - x_s);
          inout.xs.at(i).segment(0, 3) = x;
          inout.xs.at(i).segment(3, 4) = q_.coeffs();
        }
      } else {
        CHECK(false, AT);
      }

    } else {
      inout.xs = std::vector<Vxd>(inout.T + 1, x0);
    }

    if (startsWith(inout.name, "unicycle")) {
      uzero = Vxd::Zero(2);
    } else if (startsWith(inout.name, "car")) {
      uzero = Vxd::Zero(2);
    } else if (startsWith(inout.name, "quad2d")) {
      uzero = Vxd::Ones(2);
    } else if (startsWith(inout.name, "quad3d")) {
      uzero = Vxd::Ones(4);
    } else if (inout.name == "acrobot") {
      uzero = Vxd::Zero(1);
    } else {
      CHECK(false, AT);
    }

    inout.us = std::vector<Vxd>(inout.T, uzero);
  }

  bool verbose = false;
  if (verbose) {

    std::cout << "states " << std::endl;
    for (auto &x : inout.xs)
      std::cout << x.format(FMT) << std::endl;

    std::cout << "actions " << std::endl;
    for (auto &u : inout.us)
      std::cout << u.format(FMT) << std::endl;
  }
}
#endif

void convert_traj_with_variable_time(const std::vector<Vxd> &xs,
                                     const std::vector<Vxd> &us,
                                     std::vector<Vxd> &xs_out,
                                     std::vector<Vxd> &us_out, const double &dt,
                                     const dynobench::StateDyno &state) {
  DYNO_CHECK(xs.size(), AT);
  DYNO_CHECK(us.size(), AT);
  DYNO_CHECK_EQ(xs.size(), us.size() + 1, AT);

  size_t N = us.size();
  size_t nx = xs.front().size();

  size_t nu = us.front().size();
  double total_time =
      std::accumulate(us.begin(), us.end(), 0., [&dt](auto &a, auto &b) {
        return a + dt * b(b.size() - 1);
      });

  std::cout << "original total time: " << dt * us.size() << std::endl;
  std::cout << "new total_time: " << total_time << std::endl;

  size_t num_time_steps = std::ceil(total_time / dt);
  std::cout << "number of time steps " << num_time_steps << std::endl;
  std::cout << "new total time " << num_time_steps * dt << std::endl;
  double scaling_factor = num_time_steps * dt / total_time;
  std::cout << "scaling factor " << scaling_factor << std::endl;
  DYNO_CHECK_GEQ(scaling_factor, 1, AT);
  DYNO_CHECK_GEQ(scaling_factor, 1., AT);

  // now I have to sample at every dt
  // TODO: lOOK for Better solution than the trick with scaling

  auto times = Vxd(N + 1);
  times.setZero();
  for (size_t i = 1; i < static_cast<size_t>(times.size()); i++) {
    times(i) = times(i - 1) + dt * us.at(i - 1)(nu - 1);
  }
  // std::cout << times.transpose() << std::endl;

  // TODO: be careful with SO(2)
  std::vector<Vxd> x_out, u_out;
  for (size_t i = 0; i < num_time_steps + 1; i++) {
    double t = i * dt / scaling_factor;
    Vxd out(nx);
    Vxd Jout(nx);
    linearInterpolation(times, xs, t, state, out, Jout);
    x_out.push_back(out);
  }

  std::vector<Vxd> u_nx_orig(us.size());
  std::transform(us.begin(), us.end(), u_nx_orig.begin(),
                 [&nu](auto &s) { return s.head(nu - 1); });

  for (size_t i = 0; i < num_time_steps; i++) {
    double t = i * dt / scaling_factor;
    Vxd out(nu - 1);
    // std::cout << " i and time and num_time_steps is " << i << " " << t << "
    // "
    //           << num_time_steps << std::endl;
    Vxd J(nu - 1);
    linearInterpolation(times.head(times.size() - 1), u_nx_orig, t, state, out,
                        J);
    u_out.push_back(out);
  }

  std::cout << "u out " << u_out.size() << std::endl;
  std::cout << "x out " << x_out.size() << std::endl;

  xs_out = x_out;
  us_out = u_out;
}

// std::vector<ReportCost> report_problem(ptr<crocoddyl::ShootingProblem>
// problem,
//                                        const std::vector<Vxd> &xs,
//                                        const std::vector<Vxd> &us,
//                                        const char *file_name) {
//   std::vector<ReportCost> reports;
//
//   for (size_t i = 0; i < problem->get_runningModels().size(); i++) {
//     auto &x = xs.at(i);
//     auto &u = us.at(i);
//     auto p = boost::static_pointer_cast<ActionModelDyno>(
//         problem->get_runningModels().at(i));
//     std::vector<ReportCost> reports_i = get_report(
//         p, [&](ptr<Cost> f, Eigen::Ref<Vxd> r) { f->calc(r, x, u); });
//
//     for (auto &report_ii : reports_i)
//       report_ii.time = i;
//     reports.insert(reports.end(), reports_i.begin(), reports_i.end());
//   }
//
//   auto p =
//       boost::static_pointer_cast<ActionModelDyno>(problem->get_terminalModel());
//   std::vector<ReportCost> reports_t = get_report(
//       p, [&](ptr<Cost> f, Eigen::Ref<Vxd> r) { f->calc(r, xs.back()); });
//
//   for (auto &report_ti : reports_t)
//     report_ti.time = xs.size() - 1;
//   ;
//
//   reports.insert(reports.begin(), reports_t.begin(), reports_t.end());
//
//   // write down the reports.
//   //
//
//   std::string one_space = " ";
//   std::string two_space = "  ";
//   std::string four_space = "    ";
//
//   create_dir_if_necessary(file_name);
//
//   std::ofstream reports_file(file_name);
//   for (auto &report : reports) {
//     reports_file << "-" << one_space << "name: " << report.name << std::endl;
//     reports_file << two_space << "time: " << report.time << std::endl;
//     reports_file << two_space << "cost: " << report.cost << std::endl;
//     reports_file << two_space << "type: " << static_cast<int>(report.type)
//                  << std::endl;
//     if (report.r.size()) {
//       reports_file << two_space << "r: " << report.r.format(FMT) <<
//       std::endl;
//     }
//   }
//
//   return reports;
// }

// bool check_problem(ptr<crocoddyl::ShootingProblem> problem,
//                    ptr<crocoddyl::ShootingProblem> problem2,
//                    const std::vector<Vxd> &xs, const std::vector<Vxd> &us) {
//
//   bool equal = true;
//   // for (auto &x : xs) {
//   //   CSTR_V(x);
//   //   CSTR_(x.size());
//   // }
//   // std::cout << "us" << std::endl;
//   // for (auto &u : us) {
//   //
//   //   CSTR_(u.size());
//   //   CSTR_V(u);
//   // }
//
//   problem->calc(xs, us);
//   problem->calcDiff(xs, us);
//   auto data_running = problem->get_runningDatas();
//   auto data_terminal = problem->get_terminalData();
//
//   // now with finite diff
//   problem2->calc(xs, us);
//   problem2->calcDiff(xs, us);
//   auto data_running_diff = problem2->get_runningDatas();
//   auto data_terminal_diff = problem2->get_terminalData();
//
//   double tol = 1e-3;
//   bool check;
//
//   check = dynobench::check_equal(data_terminal_diff->Lx, data_terminal->Lx,
//   tol, tol); WARN(check, std::string("LxT:") + AT);
//
//   if (!check)
//     equal = false;
//
//   check = dynobench::check_equal(data_terminal_diff->Lxx, data_terminal->Lxx,
//   tol, tol); if (!check)
//     equal = false;
//   WARN(check, std::string("LxxT:") + AT);
//
//   DYNO_CHECK_EQ(data_running_diff.size(), data_running.size(), AT);
//   for (size_t i = 0; i < data_running_diff.size(); i++) {
//     auto &d = data_running.at(i);
//     auto &d_diff = data_running_diff.at(i);
//     CSTR_V(xs.at(i));
//     CSTR_V(us.at(i));
//     check = dynobench::check_equal(d_diff->Fx, d->Fx, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Fx:") + AT);
//     check = dynobench::check_equal(d_diff->Fu, d->Fu, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Fu:") + AT);
//     check = dynobench::check_equal(d_diff->Lx, d->Lx, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Lx:") + AT);
//     check = dynobench::check_equal(d_diff->Lu, d->Lu, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Lu:") + std::to_string(i) + ":" + AT);
//     check = dynobench::check_equal(d_diff->Fx, d->Fx, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Fx:") + AT);
//     check = dynobench::check_equal(d_diff->Fu, d->Fu, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Fu:") + AT);
//     check = dynobench::check_equal(d_diff->Lxx, d->Lxx, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Lxx:") + AT);
//     check = dynobench::check_equal(d_diff->Lxu, d->Lxu, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Lxu:") + AT);
//     check = dynobench::check_equal(d_diff->Luu, d->Luu, tol, tol);
//     if (!check)
//       equal = false;
//     WARN(check, std::string("Luu:") + AT);
//   }
//   return equal;
// }

void write_states_controls(const std::vector<Eigen::VectorXd> &xs,
                           const std::vector<Eigen::VectorXd> &us,
                           std::shared_ptr<dynobench::Model_robot> model_robot,
                           const dynobench::Problem &problem,
                           const char *filename) {

  // store the init guess:
  dynobench::Trajectory __traj;
  __traj.actions = us;
  __traj.states = xs;

  // {
  //   std::ofstream out(filename + std::string(".raw.yaml"));
  //   out << "states:" << std::endl;
  //   for (auto &x : xs) {
  //     out << "- " << x.format(FMT) << std::endl;
  //   }
  //   out << "actions:" << std::endl;
  //   for (auto &u : us) {
  //     out << "- " << u.format(FMT) << std::endl;
  //   }
  // }

  if (__traj.actions.front().size() > model_robot->nu) {
    for (size_t i = 0; i < __traj.actions.size(); i++) {
      __traj.actions.at(i) =
          Eigen::VectorXd(__traj.actions.at(i).head(model_robot->nu));
    }
  }
  if (__traj.states.front().size() > model_robot->nx) {
    for (size_t i = 0; i < __traj.states.size(); i++) {
      __traj.states.at(i) =
          Eigen::VectorXd(__traj.states.at(i).head(model_robot->nx));
    }
  }

  // }

  __traj.start = problem.start;
  __traj.goal = problem.goal;

  // create directory if necessary
  if (const std::filesystem::path path =
          std::filesystem::path(filename).parent_path();
      !path.empty()) {
    std::filesystem::create_directories(path);
  }

  std::ofstream init_guess(filename);
  CSTR_(filename);

  std::cout << "Check traj in controls " << std::endl;
  __traj.check(model_robot, false);
  std::cout << "Check traj in controls -- DONE " << std::endl;
  __traj.to_yaml_format(init_guess);
}

void fix_problem_quaternion(Eigen::VectorXd &start, Eigen::VectorXd &goal,
                            std::vector<Eigen::VectorXd> &xs_init,
                            std::vector<Eigen::VectorXd> &us_init) {
  std::cout << "WARNING: "
            << "i use quaternion interpolation, but distance is "
               "euclidean norm"
            << std::endl;

  // flip the start state if necessary
  double d1 = (xs_init.front().segment<4>(3) - start.segment<4>(3)).norm();
  double d2 = (xs_init.front().segment<4>(3) + start.segment<4>(3)).norm();

  if (d2 < d1) {
    std::cout << "WARNING: " << "i flip the start state" << std::endl;
    xs_init.front().segment<4>(3) *= -1.;
  }

  for (size_t j = 0; j < xs_init.size() - 1; j++) {
    Eigen::Quaterniond qa(xs_init.at(j).segment<4>(3)),
        qb(xs_init.at(j + 1).segment<4>(3)), qres;
    double t = 1;
    qres = qa.slerp(t, qb);
    std::cout << "qres " << qres.coeffs().format(FMT) << std::endl;
    xs_init.at(j + 1).segment<4>(3) = qres.coeffs();
  }

  // check the goal state...

  double d1g = (xs_init.back().segment<4>(3) - goal.segment<4>(3)).norm();
  double d2g = (xs_init.back().segment<4>(3) + goal.segment<4>(3)).norm();

  if (d2g < d1g) {

    WARN_WITH_INFO("quad3d -- I flip the quaternion of the goal state");
    goal.segment<4>(3) *= -1.;
  }
};

void add_extra_time_rate(std::vector<Eigen::VectorXd> &us_init) {
  std::vector<Vxd> us_init_time(us_init.size());
  size_t nu = us_init.front().size();
  for (size_t i = 0; i < us_init.size(); i++) {
    Vxd u(nu + 1);
    u.head(nu) = us_init.at(i);
    u(nu) = 1.;
    us_init_time.at(i) = u;
  }
  us_init = us_init_time;
};

void add_extra_state_time_rate(std::vector<Eigen::VectorXd> &xs_init,
                               Eigen::VectorXd &start) {
  std::vector<Vxd> xs_init_time(xs_init.size());
  size_t nx = xs_init.front().size();
  for (size_t i = 0; i < xs_init_time.size(); i++) {
    Vxd x(nx + 1);
    x.head(nx) = xs_init.at(i);
    x(nx) = 1.;
    xs_init_time.at(i) = x;
  }
  xs_init = xs_init_time;
  Eigen::VectorXd old_start = start;
  start.resize(nx + 1);
  start << old_start, 1.;
};

void check_problem_with_finite_diff(
    Options_trajopt options, Generate_params gen_args,
    ptr<crocoddyl::ShootingProblem> problem_croco, const std::vector<Vxd> &xs,
    const std::vector<Vxd> &us) {
  std::cout << "Checking with finite diff " << std::endl;
  options.use_finite_diff = true;
  options.disturbance = 1e-5;
  std::cout << "gen problem " << STR_(AT) << std::endl;
  size_t nx, nu;
  dynobench::Trajectory ref_traj;
  ptr<crocoddyl::ShootingProblem> problem_fdiff =
      generate_problem(gen_args, options, ref_traj);
  check_problem(problem_croco, problem_fdiff, xs, us);
};

void add_noise(double noise_level, std::vector<Eigen::VectorXd> &xs,
               std::vector<Eigen::VectorXd> &us,
               std::shared_ptr<dynobench::Model_robot> model_robot) {
  size_t nx = xs.at(0).size();
  size_t nu = us.at(0).size();
  for (size_t i = 0; i < xs.size(); i++) {
    DYNO_CHECK_EQ(static_cast<size_t>(xs.at(i).size()), nx, AT);
    xs.at(i) += noise_level * Vxd::Random(nx);
    model_robot->ensure(xs.at(i));
  }

  for (size_t i = 0; i < us.size(); i++) {
    DYNO_CHECK_EQ(static_cast<size_t>(us.at(i).size()), nu, AT);
    us.at(i) += noise_level * Vxd::Random(nu);
  }
};

void mpc_adaptative_warmstart(
    size_t counter, size_t window_optimize_i, std::vector<Vxd> &xs,
    std::vector<Vxd> &us, std::vector<Vxd> &xs_warmstart,
    std::vector<Vxd> &us_warmstart,
    std::shared_ptr<dynobench::Model_robot> model_robot, bool shift_repeat,
    ptr<dynobench::Interpolator> path, ptr<dynobench::Interpolator> path_u,
    double max_alpha) {
  size_t _nx = model_robot->nx;
  size_t _nu = model_robot->nu;
  size_t nu = us_warmstart.front().size();
  double dt = model_robot->ref_dt;

  if (counter) {
    std::cout << "new warmstart" << std::endl;
    xs = xs_warmstart;
    us = us_warmstart;
    DYNO_CHECK_GE(nu, 0, AT);
    size_t missing_steps = window_optimize_i - us.size();

    Vxd u_last = Vxd::Zero(nu);

    u_last.head(model_robot->nu) = model_robot->u_0;

    Vxd x_last = xs.back();

    // TODO: Sample the interpolator to get new init guess.

    if (shift_repeat) {
      for (size_t i = 0; i < missing_steps; i++) {
        us.push_back(u_last);
        xs.push_back(x_last);
      }
    } else {

      std::cout << "filling window by sampling the trajectory" << std::endl;
      Vxd last = xs_warmstart.back().head(_nx);

      auto it = std::min_element(path->x.begin(), path->x.end(),
                                 [&](const auto &a, const auto &b) {
                                   return model_robot->distance(a, last) <
                                          model_robot->distance(b, last);
                                 });

      size_t last_index = std::distance(path->x.begin(), it);
      double alpha_of_last = path->times(last_index);
      std::cout << STR_(last_index) << std::endl;
      // now I

      Vxd out(_nx);
      Vxd J(_nx);

      Vxd out_u(_nu);
      Vxd J_u(_nu);

      for (size_t i = 0; i < missing_steps; i++) {
        {
          path->interpolate(std::min(alpha_of_last + (i + 1) * dt, max_alpha),
                            out, J);
          xs.push_back(out);
        }

        {
          path_u->interpolate(std::min(alpha_of_last + i * dt, max_alpha - dt),
                              out_u, J_u);
          us.push_back(out_u);
        }
      }
    }

  } else {
    std::cout << "first iteration -- using first" << std::endl;

    if (window_optimize_i + 1 < xs_warmstart.size()) {
      xs = std::vector<Vxd>(xs_warmstart.begin(),
                            xs_warmstart.begin() + window_optimize_i + 1);
      us = std::vector<Vxd>(us_warmstart.begin(),
                            us_warmstart.begin() + window_optimize_i);
    } else {
      std::cout << "Optimizing more steps than required" << std::endl;
      xs = xs_warmstart;
      us = us_warmstart;

      size_t missing_steps = window_optimize_i - us.size();
      Vxd u_last = Vxd::Zero(nu);

      u_last.head(model_robot->nu) = model_robot->u_0;

      Vxd x_last = xs.back();

      // TODO: Sample the interpolator to get new init guess.

      for (size_t i = 0; i < missing_steps; i++) {
        us.push_back(u_last);
        xs.push_back(x_last);
      }
    }
  }
};

void warmstart_mpc(std::vector<Vxd> &xs, std::vector<Vxd> &us,
                   std::vector<Vxd> &xs_init_rewrite,
                   std::vector<Vxd> &us_init_rewrite, size_t counter,
                   size_t window_optimize_i, size_t window_shift) {
  xs = std::vector<Vxd>(xs_init_rewrite.begin() + counter * window_shift,
                        xs_init_rewrite.begin() + counter * window_shift +
                            window_optimize_i + 1);

  us = std::vector<Vxd>(us_init_rewrite.begin() + counter * window_shift,
                        us_init_rewrite.begin() + counter * window_shift +
                            window_optimize_i);
};

void warmstart_mpcc(std::vector<Eigen::VectorXd> &xs_warmstart,
                    std::vector<Eigen::VectorXd> &us_warmstart, int counter,
                    int window_optimize_i, std::vector<Eigen::VectorXd> &xs,
                    std::vector<Eigen::VectorXd> &us,
                    std::shared_ptr<dynobench::Model_robot> model_robot,
                    bool shift_repeat,
                    boost::shared_ptr<dynobench::Interpolator> path,
                    boost::shared_ptr<dynobench::Interpolator> path_u,
                    double dt, double max_alpha,
                    std::vector<Eigen::VectorXd> &xs_init,
                    std::vector<Eigen::VectorXd> &us_init) {
  std::cout << "warmstarting " << std::endl;

  std::vector<Vxd> xs_i;
  std::vector<Vxd> us_i;

  size_t _nx = model_robot->nx;
  size_t _nu = model_robot->nu;

  std::cout << STR(counter, ":") << std::endl;

  if (counter) {
    std::cout << "reusing solution from last iteration "
                 "(window swift)"
              << std::endl;
    xs_i = xs_warmstart;
    us_i = us_warmstart;

    size_t missing_steps = window_optimize_i - us_i.size();

    Vxd u_last = Vxd::Zero(model_robot->nu + 1);
    u_last.head(model_robot->nu) = model_robot->u_0;
    Vxd x_last = xs_i.back();

    // TODO: Sample the interpolator to get new init guess.

    if (shift_repeat) {
      std::cout << "filling window with last solution " << std::endl;
      for (size_t i = 0; i < missing_steps; i++) {
        us_i.push_back(u_last);
        xs_i.push_back(x_last);
      }
    } else {
      // get the alpha  of the last one.
      std::cout << "filling window by sampling the trajectory" << std::endl;
      Vxd last = xs_warmstart.back().head(model_robot->nx);

      auto it = std::min_element(path->x.begin(), path->x.end(),
                                 [&](const auto &a, const auto &b) {
                                   return model_robot->distance(a, last) <=
                                          model_robot->distance(b, last);
                                 });

      size_t last_index = std::distance(path->x.begin(), it);
      double alpha_of_last = path->times(last_index);
      std::cout << STR_(last_index) << std::endl;
      // now I

      Vxd out(_nx);
      Vxd J(_nx);

      Vxd out_u(_nu);
      Vxd J_u(_nu);

      Vxd uu(_nu + 1);
      Vxd xx(_nx + 1);
      for (size_t i = 0; i < missing_steps; i++) {
        {
          double alpha = std::min(alpha_of_last + (i + 1) * dt, max_alpha);
          path->interpolate(alpha, out, J);
          xx.head(_nx) = out;
          xx(_nx) = alpha;
          xs_i.push_back(xx);
        }

        {
          path_u->interpolate(std::min(alpha_of_last + i * dt, max_alpha - dt),
                              out_u, J_u);
          uu.head(_nu) = out_u;
          uu(_nu) = dt;
          us_i.push_back(uu);
        }
      }
    }
  } else {
    std::cout << "first iteration, using initial guess trajectory" << std::endl;

    Vxd x(_nx + 1);
    Vxd u(_nu + 1);
    for (size_t i = 0; i < window_optimize_i + 1; i++) {

      x.head(_nx) = xs_init.at(i);
      x(_nx) = dt * i;
      xs_i.push_back(x);

      if (i < window_optimize_i) {
        u.head(_nu) = us_init.at(i);
        u(_nu) = dt;
        us_i.push_back(u);
      }
    }
  }
  xs = xs_i;
  us = us_i;
};

void solve_for_fixed_penalty(
    Generate_params &gen_args, Options_trajopt &options_trajopt_local,
    const std::vector<Eigen::VectorXd> &xs_init,
    const std::vector<Eigen::VectorXd> &us_init, bool check_with_finite_diff,
    size_t N, const std::string &name, size_t &ddp_iterations, double &ddp_time,
    std::vector<Eigen::VectorXd> &xs_out, std::vector<Eigen::VectorXd> &us_out,
    std::shared_ptr<dynobench::Model_robot> model_robot,
    const dynobench::Problem &problem, const std::string folder_tmptraj,
    bool store_iterations, boost::shared_ptr<CallVerboseDyno> callback_dyno) {
  // geneate problem
  dynobench::Trajectory traj_ref;
  traj_ref.states = xs_init;
  traj_ref.actions = us_init;
  ptr<crocoddyl::ShootingProblem> problem_croco =
      generate_problem(gen_args, options_trajopt_local, traj_ref);

  size_t nu = model_robot->nu;
  if (gen_args.free_time) {
    nu++;
  }

  // warmstart
  std::vector<Vxd> xs, us;
  if (options_trajopt_local.use_warmstart) {
    std::cout << "using warmstart" << AT << std::endl;
    xs = xs_init;
    us = us_init;
  } else {
    xs = std::vector<Eigen::VectorXd>(N + 1, gen_args.start);
    Eigen::VectorXd u0 = Vxd(nu);
    u0 << model_robot->u_0, 1;
    us = std::vector<Vxd>(N, u0);
  }

  // check finite diff
  if (!options_trajopt_local.use_finite_diff && check_with_finite_diff) {
    check_problem_with_finite_diff(options_trajopt_local, gen_args,
                                   problem_croco, xs, us);
  }

  // add noise
  if (options_trajopt_local.noise_level > 0.) {
    add_noise(options_trajopt_local.noise_level, xs, us, model_robot);
  }

  // store init guess
  // report_problem(problem_croco, xs, us, "/tmp/dynoplan/report-0.yaml");
  std::cout << "solving with croco " << AT << std::endl;

  std::string random_id = gen_random(6);
  {
    std::string filename = folder_tmptraj + "init_guess_" + random_id + ".yaml";
    // write_states_controls(xs, us, model_robot, problem, filename.c_str());
  }

  // solve
  crocoddyl::SolverBoxFDDP ddp(problem_croco);
  ddp.set_th_stop(options_trajopt_local.th_stop);
  ddp.set_th_acceptnegstep(options_trajopt_local.th_acceptnegstep);

  if (options_trajopt_local.CALLBACKS) {
    std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
    cbs.push_back(mk<crocoddyl::CallbackVerbose>());
    if (store_iterations) {
      cbs.push_back(callback_dyno);
    }
    ddp.setCallbacks(cbs);
  }

  std::cout << "CROCO optimize" << AT << std::endl;
  crocoddyl::Timer timer;
  ddp.solve(xs, us, options_trajopt_local.max_iter, false,
            options_trajopt_local.init_reg);
  std::cout << "time: " << timer.get_duration() << std::endl;

  if (store_iterations)
    callback_dyno->store();
  std::cout << "CROCO optimize -- DONE" << std::endl;
  ddp_iterations += ddp.get_iter();
  ddp_time += timer.get_duration();
  xs_out = ddp.get_xs();
  us_out = ddp.get_us();

  // report after
  // std::string filename = folder_tmptraj + "opt_" + random_id + ".yaml";

  for (auto &x : xs_out) {
    model_robot->ensure(x);
  }
  std::cout << "solving with croco -- DONE " << AT << std::endl;
  // write_states_controls(xs_out, us_out, model_robot, problem, filename.c_str());
  // report_problem(problem_croco, xs_out, us_out, "/tmp/dynoplan/report-1.yaml");
};

void __trajectory_optimization(
    const dynobench::Problem &t_problem,
    std::shared_ptr<dynobench::Model_robot> &model_robot,
    const dynobench::Trajectory &init_guess,
    const Options_trajopt &options_trajopt, dynobench::Trajectory &traj,
    Result_opti &opti_out) {

  dynobench::Problem problem = t_problem;

  model_robot->ensure(problem.start); // precission issues with quaternions
  model_robot->ensure(problem.goal);

  Options_trajopt options_trajopt_local = options_trajopt;

  std::vector<SOLVER> solvers{SOLVER::traj_opt,
                              SOLVER::traj_opt_free_time_proxi};

  CHECK(__in_if(solvers,
                [&](const SOLVER &s) {
                  return s ==
                         static_cast<SOLVER>(options_trajopt_local.solver_id);
                }),
        "solver_id not in solvers");

  const bool modify_to_match_goal_start = false;
  const bool store_iterations = false;
  const std::string folder_tmptraj = "/tmp/dynoplan/";

  std::cout
      << "WARNING: "
      << "Cleaning data in opti_out at beginning of __trajectory_optimization"
      << std::endl;
  opti_out.data.clear();

  auto callback_dyno = mk<CallVerboseDyno>();

  // {
  //   dynobench::Trajectory __init_guess = init_guess;
  //   __init_guess.start = problem.start;
  //   __init_guess.goal = problem.goal;
  //   std::cout << "checking traj input of __trajectory_optimization "
  //             << std::endl;
  //   __init_guess.check(model_robot, false);
  //   std::cout << "checking traj input of __trajectory_optimization -- DONE "
  //             << std::endl;
  // }

  size_t ddp_iterations = 0;
  double ddp_time = 0;

  bool check_with_finite_diff = false;
  // std::string name = problem.robotType;
  std::string name = model_robot->name;
  size_t _nx = model_robot->nx;
  size_t _nu = model_robot->nu;

  bool verbose = false;
  auto xs_init = init_guess.states;
  auto us_init = init_guess.actions;
  DYNO_CHECK_EQ(xs_init.size(), us_init.size() + 1, AT);
  size_t N = init_guess.actions.size();
  auto goal = problem.goal;
  auto start = problem.start;
  double dt = model_robot->ref_dt;

  SOLVER solver = static_cast<SOLVER>(options_trajopt_local.solver_id);

  if (modify_to_match_goal_start) {
    std::cout << "WARNING: " << "i modify last state to match goal"
              << std::endl;
    xs_init.back() = goal;
    xs_init.front() = start;
  }

  // write_states_controls(xs_init, us_init, model_robot, problem,
  //                       (folder_tmptraj + "init_guess.yaml").c_str());

  size_t num_smooth_iterations =
      dt > .05 ? 3 : 5; // TODO: put this as an option in command line

  // std::vector<Eigen::VectorXd> xs_init__ = xs_init;

  if (startsWith(problem.robotType, "quad3d") &&
      !startsWith(problem.robotType, "quad3dpayload")) {
    DYNO_CHECK_EQ(start.size(), 13, "");
    DYNO_CHECK_EQ(goal.size(), 13, "");
    fix_problem_quaternion(start, goal, xs_init, us_init);
  }

  if (options_trajopt_local.smooth_traj) {
    for (size_t i = 0; i < num_smooth_iterations; i++) {
      xs_init = smooth_traj2(xs_init, *model_robot->state);
    }

    for (size_t i = 0; i < num_smooth_iterations; i++) {
      us_init = smooth_traj2(us_init, dynobench::Rn(us_init.front().size()));
    }
  }

  for (auto &x : xs_init) {
    model_robot->ensure(x);
  }

  // write_states_controls(xs_init, us_init, model_robot, problem,
  //                       (folder_tmptraj + "init_guess_smooth.yaml").c_str());

  bool success = false;
  std::vector<Vxd> xs_out, us_out;

  // create_dir_if_necessary(options_trajopt_local.debug_file_name.c_str());
  // std::ofstream debug_file_yaml(options_trajopt_local.debug_file_name);
  // {
  //   debug_file_yaml << "robotType: " << problem.robotType << std::endl;
  //   debug_file_yaml << "N: " << N << std::endl;
  //   debug_file_yaml << "start: " << start.format(FMT) << std::endl;
  //   debug_file_yaml << "goal: " << goal.format(FMT) << std::endl;
  //   debug_file_yaml << "xs0: " << std::endl;
  //   for (auto &x : xs_init)
  //     debug_file_yaml << "  - " << x.format(FMT) << std::endl;

  //   debug_file_yaml << "us0: " << std::endl;
  //   for (auto &x : us_init)
  //     debug_file_yaml << "  - " << x.format(FMT) << std::endl;
  // }

  bool __free_time_mode = solver == SOLVER::traj_opt_free_time_proxi;

  if (solver == SOLVER::traj_opt || __free_time_mode) {

    if (solver == SOLVER::traj_opt_free_time_proxi) {
      add_extra_time_rate(us_init);
    }

    std::vector<Vxd> regs;
    // if (options_trajopt_local.states_reg && solver == SOLVER::traj_opt) {
    //   double state_reg_weight = 100.;
    //   regs = std::vector<Vxd>(xs_init.size() - 1,
    //                           state_reg_weight * Vxd::Ones(_nx));
    // }

    Generate_params gen_args{
        .free_time = __free_time_mode,
        .free_time_linear = false,
        .name = name,
        .N = N,
        .goal = goal,
        .start = start,
        .model_robot = model_robot,
        .states = {xs_init.begin(), xs_init.end() - 1},
        .states_weights = regs,
        .actions = us_init,
        .collisions = options_trajopt_local.collision_weight > 1e-3,
        .reg_control = options_trajopt_local.reg_control,
        .regularize_state = options_trajopt_local.states_reg,
    };

    std::cout << "gen problem " << STR_(AT) << std::endl;

    std::vector<Eigen::VectorXd> _xs_out, _us_out, xs_init_p, us_init_p;

    xs_init_p = xs_init;
    us_init_p = us_init;
    const size_t penalty_iterations = options_trajopt_local.penalty_iterations;
    for (size_t i = 0; i < penalty_iterations; i++) {
      std::cout << "PENALTY iteration " << i << std::endl;
      gen_args.penalty = std::pow(10., double(i) / 2.);

      if (i > 0) {
        options_trajopt_local.noise_level = 0;
      }

      solve_for_fixed_penalty(gen_args, options_trajopt_local, xs_init_p, us_init_p,
                              options_trajopt_local.check_with_finite_diff, N,
                              name, ddp_iterations, ddp_time, _xs_out, _us_out,
                              model_robot, problem, folder_tmptraj,
                              store_iterations, callback_dyno);

      xs_init_p = _xs_out;
      us_init_p = _us_out;
    }

    Trajectory traj;
    traj.start = start;
    traj.goal = goal;
    traj.states.resize(_xs_out.size());
    traj.actions.resize(_us_out.size());

    for (size_t i = 0; i < traj.states.size(); i++)
      traj.states.at(i) = _xs_out.at(i).head(model_robot->nx);

    for (size_t i = 0; i < traj.actions.size(); i++)
      traj.actions.at(i) = _us_out.at(i).head(model_robot->nu);

    if (__free_time_mode) {
      traj.times.resize(_xs_out.size());
      traj.times(0) = 0.;
      for (size_t i = 1; i < static_cast<size_t>(traj.times.size()); i++)
        traj.times(i) = traj.times(i - 1) +
                        _us_out.at(i - 1).tail<1>()(0) * model_robot->ref_dt;
      std::cout << "CHECK traj with non uniform time " << std::endl;
      traj.check(model_robot, false);
      traj.update_feasibility(dynobench::Feasibility_thresholds(), true);
      std::cout << "CHECK traj with non uniform time -- DONE " << std::endl;

    }
    std::cout << "checking traj..." << std::endl;

    traj.check(model_robot, false);
    traj.update_feasibility(dynobench::Feasibility_thresholds(), true);

    success = traj.feasible;

    CSTR_(success);

    if (__free_time_mode) {

      Trajectory traj_resample = traj.resample(model_robot);

      for (auto &s : traj_resample.states) {
        model_robot->ensure(s);
      }

      std::cout << "check traj after resample " << std::endl;
      traj_resample.check(model_robot, true);
      traj_resample.update_feasibility(dynobench::Feasibility_thresholds(),
                                       true);

      xs_out = traj_resample.states;
      us_out = traj_resample.actions;

      if (problem.goal_times.size()) {
        auto ptr_derived =
            std::dynamic_pointer_cast<dynobench::Joint_robot>(model_robot);
        assert(ptr_derived);

        std::cout << "warning: fix the terminal times for the subgoals "
                  << std::endl;

        CSTR_(traj.times.size());
        for (auto &g : ptr_derived->goal_times) {
          std::cout << g << std::endl;
          std::cout << traj.times[g - 1] << std::endl;
          g = int((traj.times[g - 1] / ptr_derived->ref_dt) + 2);
        }

        for (const auto &g : ptr_derived->goal_times) {
          std::cout << "goal time " << g << std::endl;
        }

        int max_goal_time = 0;

        for (auto &t : ptr_derived->goal_times) {
          if (t > max_goal_time) {
            max_goal_time = t;
          }
        }
        DYNO_CHECK_EQ(max_goal_time, xs_out.size(), AT);
        // =======
        //       CSTR_(ts.tail(1)(0))
        //
        //       std::cout << "before resample " << std::endl;
        //       Eigen::VectorXd times;
        //       resample_trajectory(xs_out, us_out, times, xs, us, ts,
        //                           model_robot->ref_dt, model_robot->state);
        //
        //       for (auto &s : xs_out) {
        //         model_robot->ensure(s);
        // >>>>>>> 5a5e37e1549e01c73a2f5264ecd217c4aeabaf40
        //       }
      }
    } else {
      xs_out = _xs_out;
      us_out = _us_out;
    }

    // write out the solution
    // {
    //   debug_file_yaml << "xsOPT: " << std::endl;
    //   for (auto &x : xs_out)
    //     debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    //   debug_file_yaml << "usOPT: " << std::endl;
    //   for (auto &u : us_out)
    //     debug_file_yaml << "  - " << u.format(FMT) << std::endl;
    // }
  }

  // END OF Optimization

  std::ofstream file_out_debug("/tmp/dynoplan/out.yaml");

  opti_out.success = success;
  // in some s
  // opti_out.feasible = feasible;

  opti_out.xs_out = xs_out;
  opti_out.us_out = us_out;
  opti_out.cost = us_out.size() * dt;
  traj.states = xs_out;
  traj.actions = us_out;

  // TODO: check if this is actually necessary!!
  if (traj.actions.front().size() > model_robot->nu) {
    for (size_t i = 0; i < traj.actions.size(); i++) {
      Eigen::VectorXd tmp = traj.actions.at(i).head(model_robot->nu);
      traj.actions.at(i) = tmp;
    }
  }
  if (traj.states.front().size() > model_robot->nx) {
    for (size_t i = 0; i < traj.states.size(); i++) {
      Eigen::VectorXd tmp = traj.states.at(i).head(model_robot->nx);
      traj.states.at(i) = tmp;
    }
  }

  traj.start = problem.start;
  traj.goal = problem.goal;
  traj.cost = traj.actions.size() * model_robot->ref_dt;
  traj.info = "\"ddp_iterations=" + std::to_string(ddp_iterations) +
              ";"
              "ddp_time=" +
              std::to_string(ddp_time) + "\"";

  traj.to_yaml_format(file_out_debug);

  opti_out.data.insert({"ddp_time", std::to_string(ddp_time)});

  if (opti_out.success) {
    double traj_tol = 1e-2;
    double goal_tol = 1e-1;
    double col_tol = 1e-4;
    double x_bound_tol = 1e-2;
    double u_bound_tol = 1e-3;

    traj.to_yaml_format(std::cout);

    std::cout << "Final CHECK" << std::endl;
    CSTR_(model_robot->name);

    traj.check(model_robot, false);
    std::cout << "Final CHECK -- DONE" << std::endl;

    dynobench::Feasibility_thresholds thresholds{.traj_tol = traj_tol,
                                                 .goal_tol = goal_tol,
                                                 .col_tol = col_tol,
                                                 .x_bound_tol = x_bound_tol,
                                                 .u_bound_tol = u_bound_tol};

    traj.update_feasibility(thresholds);

    opti_out.feasible = traj.feasible;

    if (!traj.feasible) {
      std::cout << "WARNING: "
                << "why first feas and now infeas? (could happen using the "
                   "time proxi) "
                << std::endl;

      if (!__free_time_mode &&
          options_trajopt_local.u_bound_scale <= 1 + 1e-8) {
        // ERROR_WITH_INFO("why?");
        std::cout << "WARNING"
                  << "solver says feasible, but check says infeasible!"
                  << std::endl;
        traj.feasible = false;
        opti_out.feasible = false;
        opti_out.success = false;
      }
    }
  } else {
    traj.feasible = false;
    opti_out.feasible = false;
  }
}

void trajectory_optimization(const dynobench::Problem &problem,
                             const Trajectory &init_guess,
                             const Options_trajopt &options_trajopt,
                             Trajectory &traj, Result_opti &opti_out) {

  double time_ddp_total = 0;
  Stopwatch watch;
  Options_trajopt options_trajopt_local = options_trajopt;
  // std::string _base_path = "../../models/";

  std::shared_ptr<dynobench::Model_robot> model_robot;
  model_robot = dynobench::robot_factory(
      (problem.models_base_path + problem.robotType + ".yaml").c_str(),
      problem.p_lb, problem.p_ub);

  load_env(*model_robot, problem);

  size_t _nx = model_robot->nx; // state
  size_t _nu = model_robot->nu;

  Trajectory tmp_init_guess(init_guess), tmp_solution;

  for (auto &s : tmp_init_guess.states)
    model_robot->ensure(s);

  CSTR_(model_robot->ref_dt);

  if (!tmp_init_guess.states.size() && tmp_init_guess.num_time_steps == 0) {
    ERROR_WITH_INFO("define either xs_init or num time steps");
  }

  if (!tmp_init_guess.states.size() && !tmp_init_guess.actions.size()) {

    std::cout << "Warning: no xs_init or us_init has been provided. "
              << std::endl;

    tmp_init_guess.states.resize(init_guess.num_time_steps + 1);

    std::for_each(tmp_init_guess.states.begin(), tmp_init_guess.states.end(),
                  [&](auto &x) {
                    if (options_trajopt_local.ref_x0)
                      x = model_robot->get_x0(problem.start);
                    else
                      x = problem.start;
                  });

    tmp_init_guess.actions.resize(tmp_init_guess.states.size() - 1);
    std::for_each(tmp_init_guess.actions.begin(), tmp_init_guess.actions.end(),
                  [&](auto &x) { x = model_robot->u_0; });

    CSTR_V(tmp_init_guess.states.front());
    CSTR_V(tmp_init_guess.actions.front());
  }

  if (tmp_init_guess.states.size() && !tmp_init_guess.actions.size()) {

    std::cout << "Warning: no us_init has been provided -- using u_0: "
              << model_robot->u_0.format(FMT) << std::endl;

    tmp_init_guess.actions.resize(tmp_init_guess.states.size() - 1);

    std::for_each(tmp_init_guess.actions.begin(), tmp_init_guess.actions.end(),
                  [&](auto &x) { x = model_robot->u_0; });
  }

  if (init_guess.times.size()) {
    std::cout << "i have time stamps, I resample the trajectory" << std::endl;

    {
      std::cout << "check for input" << std::endl;
      Trajectory(init_guess).check(model_robot, false);
      std::cout << "check for input -- DONE" << std::endl;
    }

    resample_trajectory(tmp_init_guess.states, tmp_init_guess.actions,
                        tmp_init_guess.times, init_guess.states,
                        init_guess.actions, init_guess.times,
                        model_robot->ref_dt, model_robot->state);

    for (auto &s : tmp_init_guess.states) {
      model_robot->ensure(s);
    }
  }
  DYNO_CHECK(tmp_init_guess.actions.size(), AT);
  DYNO_CHECK(tmp_init_guess.states.size(), AT);
  DYNO_CHECK_EQ(tmp_init_guess.states.size(), tmp_init_guess.actions.size() + 1,
                AT);

  // check the init guess trajectory

  // std::cout << "Report on the init guess " << std::endl;
  // WARN_WITH_INFO("should I copy the first state in the init guess? -- now yes");
  // tmp_init_guess.start = problem.start;
  // tmp_init_guess.check(model_robot, false);
  // std::cout << "Report on the init guess -- DONE " << std::endl;

  switch (static_cast<SOLVER>(options_trajopt.solver_id)) {

  case SOLVER::traj_opt_free_time: {

    bool do_final_repair_step = true;
    options_trajopt_local.solver_id =
        static_cast<int>(SOLVER::traj_opt_free_time_proxi);
    options_trajopt_local.debug_file_name =
        "/tmp/dynoplan/debug_file_trajopt_freetime_proxi.yaml";
    std::cout << "**\nopti params is " << std::endl;
    options_trajopt_local.print(std::cout);

    __trajectory_optimization(problem, model_robot, tmp_init_guess,
                              options_trajopt_local, tmp_solution, opti_out);
    time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
    CSTR_(time_ddp_total);

    if (!opti_out.success) {
      std::cout << "warning" << " " << "infeasible, will do final repair step either way." << std::endl;
      do_final_repair_step = true;
    }

    if (do_final_repair_step) {

      std::cout << "time proxi was feasible, doing final step " << std::endl;
      options_trajopt_local.solver_id = static_cast<int>(SOLVER::traj_opt);
      options_trajopt_local.debug_file_name =
          "/tmp/dynoplan/debug_file_trajopt_after_freetime_proxi.yaml";


      __trajectory_optimization(problem, model_robot, tmp_solution,
                                options_trajopt_local, traj, opti_out);

      if (problem.goal_times.size()) {
        auto ptr_derived =
            std::dynamic_pointer_cast<dynobench::Joint_robot>(model_robot);
        CHECK(ptr_derived, "multiple goal times only work for joint robot");
        traj.multi_robot_index_goal = ptr_derived->goal_times;

        int max_index = 0;
        for (auto &t : ptr_derived->goal_times) {
          if (t > max_index) {
            max_index = t;
          }
        }
        DYNO_CHECK_EQ(max_index, traj.states.size(), AT);
      }

      time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
      CSTR_(time_ddp_total);
    }
    DYNO_CHECK_EQ(traj.feasible, opti_out.feasible, AT);
  } break;

  default: {
    __trajectory_optimization(problem, model_robot, tmp_init_guess,
                              options_trajopt_local, traj, opti_out);
    time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
    CSTR_(time_ddp_total);
    DYNO_CHECK_EQ(traj.feasible, opti_out.feasible, AT);
  }
  }

  // convert the format if necessary

  if (options_trajopt_local.welf_format) {
    Trajectory traj_welf;
    std::shared_ptr<dynobench::Model_quad3d> robot_derived =
        std::dynamic_pointer_cast<dynobench::Model_quad3d>(model_robot);
    traj_welf = from_quim_to_welf(traj, robot_derived->u_nominal);
    traj = traj_welf;
  }

  double time_raw = watch.elapsed_ms();
  opti_out.data.insert({"time_raw", std::to_string(time_raw)});
  opti_out.data.insert({"time_ddp_total", std::to_string(time_ddp_total)});
}


void optimize_N_steps(const dynobench::Problem &problem,
                             const Trajectory &init_guess,
                             const Options_trajopt &options_trajopt,
                             Trajectory &traj, Result_opti &opti_out) {

  double time_ddp_total = 0;
  Stopwatch watch;
  Options_trajopt options_trajopt_local = options_trajopt;

  std::shared_ptr<dynobench::Model_robot> model_robot;
  model_robot = dynobench::robot_factory(
      (problem.models_base_path + problem.robotType + ".yaml").c_str(),
      problem.p_lb, problem.p_ub);

  load_env(*model_robot, problem);

  Trajectory tmp_init_guess(init_guess), tmp_solution;

  for (auto &s : tmp_init_guess.states)
    model_robot->ensure(s);

  CSTR_(model_robot->ref_dt);

  if (!tmp_init_guess.states.size() && tmp_init_guess.num_time_steps == 0) {
    ERROR_WITH_INFO("define either xs_init or num time steps");
  }

  if (!tmp_init_guess.states.size() && !tmp_init_guess.actions.size()) {

    std::cout << "Warning: no xs_init or us_init has been provided. "
              << std::endl;

    tmp_init_guess.states.resize(init_guess.num_time_steps + 1);

    std::for_each(tmp_init_guess.states.begin(), tmp_init_guess.states.end(),
                  [&](auto &x) {
                    if (options_trajopt_local.ref_x0)
                      x = model_robot->get_x0(problem.start);
                    else
                      x = problem.start;
                  });

    tmp_init_guess.actions.resize(tmp_init_guess.states.size() - 1);
    std::for_each(tmp_init_guess.actions.begin(), tmp_init_guess.actions.end(),
                  [&](auto &x) { x = model_robot->u_0; });

    CSTR_V(tmp_init_guess.states.front());
    CSTR_V(tmp_init_guess.actions.front());
  }

  if (tmp_init_guess.states.size() && !tmp_init_guess.actions.size()) {

    std::cout << "Warning: no us_init has been provided -- using u_0: "
              << model_robot->u_0.format(FMT) << std::endl;

    tmp_init_guess.actions.resize(tmp_init_guess.states.size() - 1);

    std::for_each(tmp_init_guess.actions.begin(), tmp_init_guess.actions.end(),
                  [&](auto &x) { x = model_robot->u_0; });
  }


  DYNO_CHECK(tmp_init_guess.actions.size(), AT);
  DYNO_CHECK(tmp_init_guess.states.size(), AT);
  DYNO_CHECK_EQ(tmp_init_guess.states.size(), tmp_init_guess.actions.size() + 1, AT);
  
  const std::string folder_tmptraj = "/tmp/dynoplan/";
  auto callback_dyno = mk<CallVerboseDyno>();
  size_t ddp_iterations = 0;
  double ddp_time = 0;
  bool verbose = false;
  auto xs_init = tmp_init_guess.states;
  auto us_init = tmp_init_guess.actions;
  size_t _nx = model_robot->nx; // state
  size_t _nu = model_robot->nu;
  std::string name = model_robot->name;
  size_t N = init_guess.actions.size();
  auto goal = problem.goal;
  auto start = problem.start;
  double dt = model_robot->ref_dt;
  const bool store_iterations = false;
  bool success = false;
  SOLVER solver = static_cast<SOLVER>(options_trajopt_local.solver_id);
  bool __free_time_mode = solver == SOLVER::traj_opt_free_time_proxi;
    
  std::vector<Vxd> regs; // placeholder
  Generate_params gen_args{
      .free_time = __free_time_mode,
      .free_time_linear = false,
      .name = name,
      .N = N,
      .goal = goal,
      .start = start,
      .model_robot = model_robot,
      .states = {xs_init.begin(), xs_init.end() - 1},
      .states_weights = regs,
      .actions = us_init,
      .collisions = options_trajopt_local.collision_weight > 1e-3,
      .reg_control = options_trajopt_local.reg_control,
      .regularize_state = options_trajopt_local.states_reg,
  };

  std::cout << "gen problem " << STR_(AT) << std::endl;
  std::vector<Eigen::VectorXd> _xs_out, _us_out, xs_init_p, us_init_p;

  xs_init_p = xs_init;
  us_init_p = us_init;
  const size_t penalty_iterations = options_trajopt_local.penalty_iterations;
  for (size_t i = 0; i < penalty_iterations; i++) {
    std::cout << "PENALTY iteration " << i << std::endl;
    gen_args.penalty = std::pow(10., double(i) / 2.);

    if (i > 0) {
      options_trajopt_local.noise_level = 0;
    }

    solve_for_fixed_penalty(gen_args, options_trajopt_local, xs_init_p, us_init_p,
                            false /*check_with_finite_diff*/, N,
                            name, ddp_iterations, ddp_time, _xs_out, _us_out,
                            model_robot, problem, folder_tmptraj,
                            store_iterations, callback_dyno);

    xs_init_p = _xs_out;
    us_init_p = _us_out;
  }
  
  opti_out.xs_out = _xs_out;
  opti_out.us_out = _us_out;
  opti_out.cost = _us_out.size() * dt;
  traj.states = _xs_out;
  traj.actions = _us_out;
  traj.start = problem.start;
  traj.goal = problem.goal;
  traj.cost = traj.actions.size() * model_robot->ref_dt;
  traj.info = "\"ddp_iterations=" + std::to_string(ddp_iterations) +
              ";"
              "ddp_time=" +
              std::to_string(ddp_time) + "\"";
  

  double traj_tol = 1e-2;
  double goal_tol = 10000.0; // disable goal tol
  double col_tol = 1e-3;
  double x_bound_tol = 1e-2;
  double u_bound_tol = 1e-3;
  dynobench::Feasibility_thresholds thresholds{.traj_tol = traj_tol,
                                                  .goal_tol = goal_tol,
                                                  .col_tol = col_tol,
                                                  .x_bound_tol = x_bound_tol,
                                                  .u_bound_tol = u_bound_tol};
  traj.update_feasibility(thresholds);
  std::cout << "Final CHECK" << std::endl;
  std::cout << "states size " << traj.states.size() << std::endl;
  std::cout << "actions size " << traj.actions.size() << std::endl;
  traj.check(model_robot, true);
  opti_out.feasible = traj.feasible;
  success = traj.feasible;
  opti_out.success = success;
  opti_out.data.insert({"ddp_time", std::to_string(ddp_time)});
  time_ddp_total += std::stod(opti_out.data.at("ddp_time"));
  CSTR_(time_ddp_total);
  DYNO_CHECK_EQ(traj.feasible, opti_out.feasible, AT);
  

  double time_raw = watch.elapsed_ms();
  opti_out.data.insert({"time_raw", std::to_string(time_raw)});
  opti_out.data.insert({"time_ddp_total", std::to_string(time_ddp_total)});
}



void Result_opti::write_yaml(std::ostream &out) {
  out << "feasible: " << feasible << std::endl;
  out << "success: " << success << std::endl;
  out << "cost: " << cost << std::endl; // TODO: why 2*cost is joint_robot?
  if (data.size()) {
    out << "info:" << std::endl;
    for (const auto &[k, v] : data) {
      out << "  " << k << ": " << v << std::endl;
    }
  }
  // TODO: @QUIM @AKMARAL Clarify this!!!
  // out << "result:" << std::endl;
  // // out << "xs_out: " << std::endl;
  // out << "  - states:" << std::endl;
  // for (auto &x : xs_out)
  //   out << "      - " << x.format(FMT) << std::endl;

  // // out << "us_out: " << std::endl;
  // out << "    actions:" << std::endl;
  // for (auto &u : us_out)
  //   out << "      - " << u.format(FMT) << std::endl;
}

void Result_opti::write_yaml_db(std::ostream &out) {
  // CHECK((name != ""), AT);
  out << "feasible: " << feasible << std::endl;
  out << "success: " << success << std::endl;
  out << "cost: " << cost << std::endl;
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (auto &x : xs_out) {
    // if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0",
    //               "car_first_order_with_1_trailers_0", "quad2d"},
    //          name)) {
    //   x(2) = std::remainder(x(2), 2 * M_PI);
    // } else if (name == "acrobot") {
    //   x(0) = std::remainder(x(0), 2 * M_PI);
    //   x(1) = std::remainder(x(1), 2 * M_PI);
    // }
    out << "      - " << x.format(FMT) << std::endl;
  }

  out << "    actions:" << std::endl;
  for (auto &u : us_out) {
    out << "      - " << u.format(FMT) << std::endl;
  }
};

std::vector<Eigen::VectorXd>
smooth_traj2(const std::vector<Eigen::VectorXd> &xs_init,
             const dynobench::StateDyno &state) {
  size_t n = xs_init.front().size();
  size_t ndx = state.ndx;
  DYNO_CHECK_EQ(n, state.nx, AT);
  std::vector<Vxd> xs_out(xs_init.size(), Eigen::VectorXd::Zero(n));

  // compute diff vectors

  Eigen::VectorXd diffA = Eigen::VectorXd::Zero(ndx);
  Eigen::VectorXd diffB = Eigen::VectorXd::Zero(ndx);
  Eigen::VectorXd diffC = Eigen::VectorXd::Zero(ndx);

  xs_out.front() = xs_init.front();
  xs_out.back() = xs_init.back();
  for (size_t i = 1; i < xs_init.size() - 1; i++) {
    state.diff(xs_init.at(i - 1), xs_init.at(i), diffA);
    state.diff(xs_init.at(i - 1), xs_init.at(i + 1), diffB);

    if (i == xs_init.size() - 2) {
      state.integrate(xs_init.at(i - 1), (diffA + diffB / 2.) / 2.,
                      xs_out.at(i));
    } else {
      state.diff(xs_init.at(i - 1), xs_init.at(i + 2), diffC);
      state.integrate(xs_init.at(i - 1), (diffA + diffB / 2. + diffC / 3.) / 3.,
                      xs_out.at(i));
    }
  }
  // smooth the diffs
  return xs_out;
}

} // namespace dynoplan
