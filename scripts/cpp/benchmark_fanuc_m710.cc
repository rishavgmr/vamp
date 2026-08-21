// RSW-2740: VAMP-side counterpart to pRRTC's scripts/benchmark_fanuc_m710.cpp - same test-case
// data (problem.json for start/goal), same RRTCSettings values where VAMP's own settings struct
// has an equivalent field, run the same number of times, for a comparable trial. Not part of
// upstream vamp - ours, for this benchmarking effort only.
//
// Environment representation differs from pRRTC's own SDF-grid-based one: vamp has no generic
// SDF primitive (Sphere/Capsule/Cuboid/Heightfield/Pointcloud only - see vamp's own README), so
// env_cuboids.json (sdf_to_cuboids.py's output - a greedy axis-aligned box decomposition of the
// same voxelized SDF .bin files pRRTC uploads directly) stands in for it here.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/random/halton.hh>
#include <vamp/robots/fanuc_m710.hh>

using Robot = vamp::robots::FanucM710;
static constexpr std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;

namespace
{

    // Test-case data lives in the platform repo, not vamp's own resources/ - same absolute-path
    // convention pRRTC's benchmark_fanuc_m710.cpp uses for the same files.
    const std::string kCaseDir = "/workspaces/platform/src/planners/trajectory_planner/test/data/"
                                 "p2p_benchmark/cases/collins";

    constexpr int kNumRuns = 10;

    // vamp::collision::factory::cuboid::array() takes Euler XYZ angles (rho, theta, phi applied as
    // Rz(phi)*Ry(theta)*Rx(rho), the same intrinsic order as URDF's rpy); env_cuboids.json stores
    // orientation as a quaternion instead (shared with cuboids_to_urdf.py/publish_cuboid_markers.py,
    // which both need xyzw directly), so convert once here rather than changing that shared schema
    // for this one consumer.
    std::array<float, 3> quatToRpy(float x, float y, float z, float w)
    {
        float roll = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
        float pitch = std::asin(std::max(-1.0f, std::min(1.0f, 2 * (w * y - z * x))));
        float yaw = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
        return {roll, pitch, yaw};
    }

    EnvironmentInput load_environment()
    {
        nlohmann::json cuboids_json;
        std::ifstream(kCaseDir + "/env_cuboids.json") >> cuboids_json;

        EnvironmentInput environment;
        for (const auto &[object_name, cuboids] : cuboids_json.items())
        {
            for (const auto &c : cuboids)
            {
                auto center = c.at("center").get<std::array<float, 3>>();
                auto size = c.at("size").get<std::array<float, 3>>();
                auto quat = c.at("quat_xyzw").get<std::array<float, 4>>();
                auto rpy = quatToRpy(quat[0], quat[1], quat[2], quat[3]);
                environment.cuboids.push_back(
                    vamp::collision::factory::cuboid::array(
                        center, rpy, std::array<float, 3>{size[0] / 2.0f, size[1] / 2.0f, size[2] / 2.0f}));
            }
        }
        return environment;
    }

    Robot::ConfigurationArray config_from_json(const nlohmann::json &arr)
    {
        Robot::ConfigurationArray config{};
        for (std::size_t i = 0; i < Robot::dimension; i++)
        {
            config[i] = arr.at(i).get<float>();
        }
        return config;
    }

    nlohmann::json config_to_json(const Robot::Configuration &config)
    {
        nlohmann::json out = nlohmann::json::array();
        const auto &array = config.to_array();
        for (std::size_t i = 0; i < Robot::dimension; i++)
        {
            out.push_back(array[i]);
        }
        return out;
    }

    // PlanningResult::cost is only ever set by RRTC::solve() itself - vamp::planning::simplify()
    // never touches it (confirmed by reading simplify.hh end to end: it only populates .path,
    // .nanoseconds, and .iterations), so a post-shortcut PlanningResult's .cost is always its
    // default-constructed 0, not a real value. Compute cost directly from consecutive-waypoint
    // distances for both pre- and post-shortcut paths instead of trusting either result's .cost
    // field, mirroring pRRTC's own benchmark's independent path_cost() helper.
    float path_cost(const vamp::planning::Path<Robot> &path)
    {
        float cost = 0.0F;
        for (std::size_t i = 1; i < path.size(); i++)
        {
            cost += (path[i] - path[i - 1]).l2_norm();
        }
        return cost;
    }

}  // namespace

auto main(int, char **) -> int
{
    std::cout << std::unitbuf;

    std::cout << "Loading environment cuboids...\n";
    auto env_input = load_environment();
    std::cout << "  " << env_input.cuboids.size() << " cuboids\n";
    env_input.sort();
    auto environment = EnvironmentVector(env_input);

    nlohmann::json problem_json;
    std::ifstream(kCaseDir + "/problem.json") >> problem_json;
    Robot::Configuration start(config_from_json(problem_json.at("start")));
    Robot::Configuration goal(config_from_json(problem_json.at("goal")));

    // Mirrors pRRTC's own settings (scripts/benchmark_fanuc_m710.cpp) wherever RRTCSettings has
    // an equivalent field - range/dynamic_domain/radius/alpha/min_radius all match; max_iterations
    // capped to the same 5000 for a comparable trial (VAMP's own default is 100000).
    vamp::planning::RRTCSettings rrtc_settings;
    rrtc_settings.range = 0.5F;
    rrtc_settings.dynamic_domain = true;
    rrtc_settings.radius = 4.0F;
    rrtc_settings.alpha = 0.0001F;
    rrtc_settings.min_radius = 1.0F;
    rrtc_settings.balance = true;
    rrtc_settings.tree_ratio = 1.0F;
    rrtc_settings.max_iterations = 5000;
    rrtc_settings.max_samples = 5000;

    // SHORTCUT only (no BSPLINE) - matches pRRTC's own postProcessing()/shortcutPath, which only
    // does randomized shortcutting, not spline smoothing.
    vamp::planning::SimplifySettings simplify_settings;
    simplify_settings.operations = {vamp::planning::SHORTCUT};
    // simplify_settings.operations = {};

    int num_success = 0;
    std::vector<float> costs;
    std::vector<double> wall_times_s;
    nlohmann::json runs = nlohmann::json::array();

    // vamp::rng::Halton always starts from the same fixed per-dimension prime bases (see
    // halton.hh's default constructor) and its C++ API has no seed/skip parameter - unlike
    // vamp's own Python scripts' --rng_skip_iterations, which (per vamp's own binding layer, not
    // reproduced here) just discards that many next() draws before planning starts. Reimplement
    // the same idea directly: give each run a non-overlapping slice of the sequence by skipping
    // run_index * rrtc_settings.max_samples draws - that's an upper bound on how many samples a
    // single run could consume, so consecutive runs' windows never overlap. Without this, every
    // run reuses the exact same sequence from the same starting state and produces identical
    // results (same cost, same iteration count every time), unlike pRRTC where GPU scheduling
    // jitter varies results across runs despite a fixed seed.
    for (int i = 0; i < kNumRuns; i++)
    {
        std::cout << "Starting run " << i << "...\n";
        auto rng = std::make_shared<vamp::rng::Halton<Robot>>();
        const std::size_t skip = static_cast<std::size_t>(i) * rrtc_settings.max_samples;
        for (std::size_t s = 0; s < skip; s++)
        {
            rng->next();
        }

        auto t0 = std::chrono::steady_clock::now();
        auto result = RRTC::solve(start, goal, environment, rrtc_settings, rng);
        bool solved = !result.path.empty();
        double cost_before_shortcut = path_cost(result.path);

        vamp::planning::PlanningResult<Robot> simplify_result;
        if (solved)
        {
            simplify_result = vamp::planning::simplify<Robot, rake, Robot::resolution>(
                result.path, environment, simplify_settings, rng);
        }
        auto t1 = std::chrono::steady_clock::now();
        double planning_time_s = std::chrono::duration<double>(t1 - t0).count();
        double cost_after_shortcut = solved ? path_cost(simplify_result.path) : 0.0;

        std::cout << "run " << i << ": solved=" << solved << " cost=" << cost_after_shortcut
                  << " (before shortcut=" << cost_before_shortcut << ") wall_ms=" << planning_time_s * 1e3
                  << " rrtc_iterations=" << result.iterations << "\n";

        nlohmann::json run;
        run["iteration"] = i;
        run["success"] = solved;
        run["planning_time_s"] = planning_time_s;
        run["rrtc_nanoseconds"] = result.nanoseconds;
        run["rrtc_iterations"] = result.iterations;
        if (solved)
        {
            ++num_success;
            costs.push_back(static_cast<float>(cost_after_shortcut));
            wall_times_s.push_back(planning_time_s);
            run["path_cost"] = cost_after_shortcut;
            run["path_cost_before_shortcut"] = cost_before_shortcut;
            nlohmann::json path = nlohmann::json::array();
            for (const auto &cfg : simplify_result.path)
            {
                path.push_back(config_to_json(cfg));
            }
            run["path"] = path;
        }
        runs.push_back(std::move(run));
    }

    nlohmann::json output;
    output["algorithm"] = "vamp_rrtc";
    output["robot"] = "fanuc_m710";
    output["case"] = "collins";
    output["environment_representation"] = "cuboids (from sdf_to_cuboids.py)";
    output["settings"] = {
        {"range", rrtc_settings.range},
        {"dynamic_domain", rrtc_settings.dynamic_domain},
        {"radius", rrtc_settings.radius},
        {"alpha", rrtc_settings.alpha},
        {"min_radius", rrtc_settings.min_radius},
        {"max_iterations", rrtc_settings.max_iterations},
        {"max_samples", rrtc_settings.max_samples},
    };
    output["runs"] = runs;
    output["success_rate"] = static_cast<double>(num_success) / kNumRuns;
    if (!costs.empty())
    {
        output["path_cost_mean"] = std::accumulate(costs.begin(), costs.end(), 0.0) / costs.size();
        output["path_cost_min"] = *std::min_element(costs.begin(), costs.end());
        output["path_cost_max"] = *std::max_element(costs.begin(), costs.end());
        output["planning_time_s_mean"] =
            std::accumulate(wall_times_s.begin(), wall_times_s.end(), 0.0) / wall_times_s.size();
    }

    // Timestamped, matching pRRTC's own benchmark_fanuc_m710.cpp convention exactly (same
    // now_s/output_path construction) - a fixed filename would silently overwrite the previous
    // run's output instead of leaving a trail of runs to compare, the way p2p's and pRRTC's own
    // benchmarks already do in this same directory.
    const auto now_s =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    const std::string out_path =
        "/home/gmruser/.ss_temp/p2p_benchmark_paths/vamp_benchmark_" + std::to_string(now_s) + ".json";
    std::ofstream out_file(out_path);
    out_file << output.dump(2);
    std::cout << "\nWrote " << out_path << "\n";

    return 0;
}
