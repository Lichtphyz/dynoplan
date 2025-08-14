#pragma once
// DYNOPLAN
#include "dynoplan/tdbastar/tdbastar.hpp"
#include "dynoplan/tdbastar/planresult.hpp"
#include "dynoplan/tdbastar/compute_heu.hpp"
#include "dynoplan/tdbastar/options.hpp"
// DYNOBENCH
#include "dynobench/general_utils.hpp"
#include "dynobench/robot_models_base.hpp"
void compute_heuristics(
    int N, dynobench::Problem &problem,
    dynoplan::Options_tdbastar &options_tdbastar,
    size_t &robot_id,
    ompl::NearestNeighbors<std::shared_ptr<dynoplan::AStarNode>> **heuristic_result);
