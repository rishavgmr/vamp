// RSW-2740: VAMP-side benchmark using vamp::collision::SDFGrid directly (src/impl/vamp/collision/
// shapes.hh, sphere_sdf.hh - added for this comparison, not upstream vamp) instead of
// scripts/cpp/benchmark_fanuc_m710.cc's cuboid-decomposition stand-in - queries the exact same
// voxel grids pRRTC uploads (scripts/benchmark_fanuc_m710.cpp's load_sdf_environment()), so this
// is the real apples-to-apples comparison against pRRTC's own SDF-based benchmark. See
// scripts/cpp/benchmark_fanuc_m710.cc's own header comment for why cuboids existed as a fallback.
//
// Case-driven (RSW-2740, "give VAMP the same flexibility as pRRTC" plan): tool spheres, joint
// sampling bounds, the env/workpiece SDF grids, and per-link environment masking are all read at
// runtime from the case directory selected by VAMP_BENCHMARK_CASE (mirrors
// PRRTC_BENCHMARK_CASE/P2P_BENCHMARK_CASE) - see load_tool_attachment/compute_joint_limits/
// load_environment/compute_masked_link_bitmask below (Phase 1 and Phase 3). What's NOT runtime-
// configurable: base_link's own sphere geometry (24 vs 22 spheres depending on the case) and the
// frames[1] mounting rotation, both fused into fanuc_m710.hh's generated code - switching cases
// with materially different base_link/frames[1] needs `vamp_case_setup.py --case <name>` run
// first (regenerates fanuc_m710.hh, requires a rebuild), not just re-running this binary with a
// different VAMP_BENCHMARK_CASE (Phase 2 - see the RSW-2740 plan for why this half stays
// build-time: it's baked into straight-line generated code, not data this binary loads).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include <vamp/collision/environment.hh>
#include <vamp/collision/factory.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/random/case_halton.hh>
#include <vamp/robots/fanuc_m710.hh>

using Robot = vamp::robots::FanucM710;
static constexpr std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;

namespace
{
    const std::string kFixtureDir = "/workspaces/platform/src/planners/trajectory_planner/test/data/"
                                    "p2p_benchmark";

    // Runtime-selectable case directory (RSW-2740), mirroring pRRTC's own
    // PRRTC_BENCHMARK_CASE/case_name() (scripts/benchmark_fanuc_m710.cpp) - defaults to
    // "collins" so existing invocations don't change behavior.
    std::string case_name()
    {
        const char *env = std::getenv("VAMP_BENCHMARK_CASE");
        return env ? std::string(env) : std::string("collins");
    }

    const std::string kCaseName = case_name();
    const std::string kCaseDir = kFixtureDir + "/cases/" + kCaseName;

    constexpr int kNumRuns = 10;

    // Same margins as pRRTC's own scripts/benchmark_fanuc_m710.cpp (kEnvironmentCollisionOffsetM/
    // kWorkpieceCollisionOffsetM/kSelfCollisionOffsetM) - identical margins on both sides is
    // what makes this a fair speed comparison rather than also comparing different safety
    // margins.
    constexpr float kEnvironmentCollisionOffsetM = 0.025F;
    constexpr float kWorkpieceCollisionOffsetM = 0.04F;
    constexpr float kSelfCollisionOffsetM = 0.025F;

    // Mirrors pRRTC's own load_sdf_grid_host() (scripts/benchmark_fanuc_m710.cpp) byte-for-byte -
    // same header layout, same int16-or-float payload auto-detection - except the payload is
    // converted to float here at load time rather than kept as raw bytes, since
    // vamp::collision::SDFGrid<DataT>::data is always std::vector<float> (see shapes.hh's own
    // comment on why: no SIMD benefit to the narrower storage on the CPU/SIMD side).
    struct HostGrid
    {
        uint32_t num_x, num_y, num_z;
        float bounds_lower[3], bounds_upper[3];
        std::vector<float> data;
    };

    HostGrid load_sdf_grid_host(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            throw std::runtime_error("failed to open " + path);
        }

        HostGrid g;
        f.read(reinterpret_cast<char *>(&g.num_x), sizeof(uint32_t));
        f.read(reinterpret_cast<char *>(&g.num_y), sizeof(uint32_t));
        f.read(reinterpret_cast<char *>(&g.num_z), sizeof(uint32_t));
        f.read(reinterpret_cast<char *>(g.bounds_lower), sizeof(float) * 3);
        f.read(reinterpret_cast<char *>(g.bounds_upper), sizeof(float) * 3);
        if (!f)
        {
            throw std::runtime_error(path + ": failed reading header");
        }

        std::vector<uint8_t> payload((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const std::size_t n_voxels = static_cast<std::size_t>(g.num_x) * g.num_y * g.num_z;
        g.data.resize(n_voxels);
        if (payload.size() == n_voxels * 2)
        {
            const auto *raw = reinterpret_cast<const int16_t *>(payload.data());
            for (std::size_t i = 0; i < n_voxels; i++)
            {
                g.data[i] = static_cast<float>(raw[i]);
            }
        }
        else if (payload.size() == n_voxels * 4)
        {
            const auto *raw = reinterpret_cast<const float *>(payload.data());
            std::copy(raw, raw + n_voxels, g.data.begin());
        }
        else
        {
            throw std::runtime_error(path + ": payload size mismatch");
        }
        return g;
    }

    std::string mesh_basename_no_ext(const std::string &path)
    {
        const std::size_t slash = path.find_last_of('/');
        const std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
        const std::size_t dot = base.find_last_of('.');
        return (dot == std::string::npos) ? base : base.substr(0, dot);
    }

    // World-to-local inverse rigid transform for one env object - mirrors pRRTC's own
    // compute_inverse_transform() (scripts/benchmark_fanuc_m710.cpp), just filling
    // SDFGrid<float>'s flat inv_rotation_row*/inv_translation_* fields directly instead of a
    // separate InverseTransform struct.
    void set_grid_transform(
        vamp::collision::SDFGrid<float> &grid,
        const nlohmann::json &position,
        const nlohmann::json &orientation)
    {
        const Eigen::Vector3d t(
            position[0].get<double>(), position[1].get<double>(), position[2].get<double>());
        const Eigen::Quaterniond q(
            orientation[3].get<double>(),
            orientation[0].get<double>(),
            orientation[1].get<double>(),
            orientation[2].get<double>());
        const Eigen::Matrix3d r_inv = q.toRotationMatrix().transpose();
        const Eigen::Vector3d t_inv = -r_inv * t;

        grid.inv_rotation_row0_x = static_cast<float>(r_inv(0, 0));
        grid.inv_rotation_row0_y = static_cast<float>(r_inv(0, 1));
        grid.inv_rotation_row0_z = static_cast<float>(r_inv(0, 2));
        grid.inv_rotation_row1_x = static_cast<float>(r_inv(1, 0));
        grid.inv_rotation_row1_y = static_cast<float>(r_inv(1, 1));
        grid.inv_rotation_row1_z = static_cast<float>(r_inv(1, 2));
        grid.inv_rotation_row2_x = static_cast<float>(r_inv(2, 0));
        grid.inv_rotation_row2_y = static_cast<float>(r_inv(2, 1));
        grid.inv_rotation_row2_z = static_cast<float>(r_inv(2, 2));
        grid.inv_translation_x = static_cast<float>(t_inv(0));
        grid.inv_translation_y = static_cast<float>(t_inv(1));
        grid.inv_translation_z = static_cast<float>(t_inv(2));
    }

    void set_grid_transform_identity(vamp::collision::SDFGrid<float> &grid)
    {
        grid.inv_rotation_row0_x = 1.0F;
        grid.inv_rotation_row0_y = 0.0F;
        grid.inv_rotation_row0_z = 0.0F;
        grid.inv_rotation_row1_x = 0.0F;
        grid.inv_rotation_row1_y = 1.0F;
        grid.inv_rotation_row1_z = 0.0F;
        grid.inv_rotation_row2_x = 0.0F;
        grid.inv_rotation_row2_y = 0.0F;
        grid.inv_rotation_row2_z = 1.0F;
        grid.inv_translation_x = 0.0F;
        grid.inv_translation_y = 0.0F;
        grid.inv_translation_z = 0.0F;
    }

    vamp::collision::SDFGrid<float> build_grid(const HostGrid &h, float offset_scaled)
    {
        float spacing = (h.bounds_upper[0] - h.bounds_lower[0]) / static_cast<float>(h.num_x);
        return vamp::collision::SDFGrid<float>(
            h.bounds_lower[0],
            h.bounds_lower[1],
            h.bounds_lower[2],
            h.bounds_upper[0],
            h.bounds_upper[1],
            h.bounds_upper[2],
            spacing,
            h.num_x,
            h.num_y,
            h.num_z,
            offset_scaled,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F,
            0.0F,
            h.data);
    }

    // RSW-2740 Phase 3: maps a scene_config collisionMask "robot_link" name (e.g. "base_link")
    // to the cricket link_index ccfk_template.hh's per-link blocks are compiled with, via the
    // link_names array cricket now emits (fk_template.hh) specifically to make this lookup
    // possible at runtime.
    int link_name_to_index(const std::string &name)
    {
        for (std::size_t i = 0; i < Robot::link_names.size(); i++)
        {
            if (Robot::link_names[i] == name)
            {
                return static_cast<int>(i);
            }
        }
        throw std::runtime_error("link_name_to_index: unrecognized link name \"" + name + "\"");
    }

    // Builds this env object's masked_link_bitmask from scene_config's own collisionMask (RSW-
    // 2740 Phase 3) - bit i set means link_index i skips checking against this object entirely,
    // mirroring pRRTC's own uploadSDFEnvironment(..., link_env_mask, ...). Absent collisionMask,
    // or an object not named in any entry's masked_env list, keeps bitmask 0 (checked by every
    // link) - matches the pre-existing, correct default for every object besides the ones a case
    // explicitly exempts a link from.
    std::uint32_t
    compute_masked_link_bitmask(const nlohmann::json &scene_json, const std::string &object_name)
    {
        std::uint32_t mask = 0;
        // collisionMask lives under chains[0], not at scene_json's top level (verified directly
        // against collins' own scene_config_m710.json - RSW-2740).
        const auto &chain = scene_json["robots"][0]["chains"][0];
        if (!chain.contains("collisionMask"))
        {
            return mask;
        }
        for (const auto &entry : chain["collisionMask"])
        {
            const auto masked_env = entry["masked_env"].get<std::vector<std::string>>();
            if (std::find(masked_env.begin(), masked_env.end(), object_name) == masked_env.end())
            {
                continue;
            }
            const int link_index = link_name_to_index(entry["robot_link"].get<std::string>());
            mask |= (1u << static_cast<unsigned>(link_index));
        }
        return mask;
    }

    // Loads this case's environment straight from the same scene_config_m710.json/.bin files
    // pRRTC's own load_sdf_environment() reads, driven by the "env" list rather than a hardcoded
    // object list - RESTRICTED_ZONE is deliberately excluded (matches pRRTC's benchmark and
    // benchmark_p2p.cpp's own addIgnoredObject("RESTRICTED_ZONE")).
    EnvironmentInput load_environment()
    {
        nlohmann::json scene_json;
        std::ifstream(kCaseDir + "/scene_config_m710.json") >> scene_json;

        EnvironmentInput environment;

        for (const auto &env_obj : scene_json["env"])
        {
            const std::string name = env_obj["name"].get<std::string>();
            if (name == "RESTRICTED_ZONE")
            {
                continue;
            }

            const std::string bin_path =
                kCaseDir + "/env_meshes/" + mesh_basename_no_ext(env_obj["path"].get<std::string>()) + ".bin";
            auto host = load_sdf_grid_host(bin_path);
            auto grid = build_grid(host, kEnvironmentCollisionOffsetM * 100.0F);
            set_grid_transform(grid, env_obj["position"], env_obj["orientation"]);
            grid.masked_link_bitmask = compute_masked_link_bitmask(scene_json, name);
            grid.name = name;

            std::cout << "  " << name << ": " << host.num_x << "x" << host.num_y << "x" << host.num_z
                      << " voxels\n";
            environment.sdf_grids.push_back(std::move(grid));
        }

        {
            auto host = load_sdf_grid_host(kCaseDir + "/workpiece.bin");
            auto grid = build_grid(host, kWorkpieceCollisionOffsetM * 100.0F);
            set_grid_transform_identity(grid);
            grid.masked_link_bitmask = compute_masked_link_bitmask(scene_json, "workpiece");
            grid.name = "workpiece";
            std::cout << "  workpiece: " << host.num_x << "x" << host.num_y << "x" << host.num_z
                      << " voxels\n";
            environment.sdf_grids.push_back(std::move(grid));
        }

        return environment;
    }

    // --- Tool spheres, loaded at runtime via vamp::collision::Attachment (RSW-2740 Phase 1a) ---
    //
    // A sphere as scanned directly out of a sphere-model yaml, still in the tool's own local
    // frame, exactly as authored (RSW-2740).
    struct RawSphere
    {
        float x, y, z, radius;
    };

    // Identical regex to pRRTC's own scan_spheres_raw() (scripts/benchmark_fanuc_m710.cpp) -
    // deliberately not a real YAML parser for the same reason stated there: this project's
    // benchmark scripts don't otherwise link one, and scanning for center/radius pairs is robust
    // enough for this fixed, simple format regardless of comments, quoting, or link nesting.
    std::vector<RawSphere> scan_spheres_raw(const std::string &content)
    {
        static const std::regex sphere_re(
            R"("?center"?:\s*\[\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\]\s*\n\s*"?radius"?:\s*([-0-9.eE]+))");

        std::vector<RawSphere> spheres;
        for (std::sregex_iterator it(content.begin(), content.end(), sphere_re), end; it != end; ++it)
        {
            const auto &m = *it;
            spheres.push_back({std::stof(m[1]), std::stof(m[2]), std::stof(m[3]), std::stof(m[4])});
        }
        return spheres;
    }

    std::string read_file(const std::string &path)
    {
        std::ifstream f(path);
        if (!f)
        {
            throw std::runtime_error("failed to open " + path);
        }
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    // Builds a tool0-attached Attachment<float> straight from tool_spheres/fine.yaml, with NO
    // manual frame conversion (RSW-2740) - unlike pRRTC, which bakes tool spheres directly into
    // joint 7's sphere array and therefore needs a manual (x, -y, -z) flip to account for the
    // link_6->tool0 fixed joint's rpy="3.1415 0 0" it never actually walks through at runtime,
    // VAMP's fkcc_attach() poses this attachment against tool0's REAL FK-computed frame
    // (set_attachment_pose(environment, to_isometry(&y[...]))), which already includes that
    // rotation because it's a genuine joint in the walked chain. Verified empirically: an
    // independent from-scratch FK computed directly from fine_spheres.urdf's own <joint> origins
    // (Python, no cricket/VAMP involved) agrees with Robot::eefk() to ~1e-4 at q=0, including
    // applying a raw tool0-local test point through both and getting the same world position -
    // so raw yaml coordinates + identity relative_frame is correct, not a coincidence.
    //
    // Uses the FINE tier only (no separate coarse/fine escalation exists for attachments in
    // vamp's generated code, unlike the arm's own two-tier spheres).
    vamp::collision::Attachment<float> load_tool_attachment(const std::string &path)
    {
        const auto raw = scan_spheres_raw(read_file(path));
        if (raw.empty())
        {
            throw std::runtime_error(path + ": found no center/radius sphere entries");
        }

        vamp::collision::Attachment<float> attachment(Eigen::Isometry3f::Identity());
        attachment.spheres.reserve(raw.size());
        for (const auto &s : raw)
        {
            attachment.spheres.emplace_back(vamp::collision::factory::sphere::flat(s.x, s.y, s.z, s.radius));
        }
        return attachment;
    }

    // --- Joint sampling bounds, read at runtime from this case's scene_config (RSW-2740 Phase
    // 1b) ---
    //
    // Mirrors pRRTC's own compute_joint_limits()/uploadJointLimits() (scripts/
    // benchmark_fanuc_m710.cpp): dof 0 (rail) is scene_config's own axisLowerLimits/
    // axisUpperLimits for whichever axis is active; dofs 1-6 are its jointLowerLimits/
    // jointUpperLimits directly. These are case-specific operational bounds, not
    // Robot::s_m/s_a's compile-time hardware limits (baked from robot.urdf, close to but not
    // identical to a case's own tighter bounds) - vamp::rng::CaseHalton scales into these
    // instead.
    std::pair<std::array<float, 7>, std::array<float, 7>> compute_joint_limits()
    {
        nlohmann::json scene_json;
        std::ifstream(kCaseDir + "/scene_config_m710.json") >> scene_json;
        const auto &chain = scene_json["robots"][0]["chains"][0];

        const auto axis_options = chain["axisOptions"].get<std::vector<bool>>();
        int active_axis = -1;
        for (std::size_t i = 0; i < axis_options.size(); ++i)
        {
            if (axis_options[i])
            {
                active_axis = static_cast<int>(i);
                break;
            }
        }
        if (active_axis < 0)
        {
            throw std::runtime_error("compute_joint_limits: no active entry in axisOptions");
        }

        std::array<float, 7> lower{}, upper{};
        lower[0] = chain["axisLowerLimits"][active_axis].get<float>();
        upper[0] = chain["axisUpperLimits"][active_axis].get<float>();

        const auto joint_lower = chain["jointLowerLimits"].get<std::vector<float>>();
        const auto joint_upper = chain["jointUpperLimits"].get<std::vector<float>>();
        if (joint_lower.size() != 6 || joint_upper.size() != 6)
        {
            throw std::runtime_error(
                "compute_joint_limits: expected exactly 6 jointLowerLimits/jointUpperLimits entries");
        }
        for (int i = 0; i < 6; ++i)
        {
            lower[i + 1] = joint_lower[i];
            upper[i + 1] = joint_upper[i];
        }
        return {lower, upper};
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
    std::cout << "Case: " << kCaseName << " (" << kCaseDir << ")\n";

    // RSW-2740 Phase 4: set once, read by every sphere_sphere_self_collision() call for the
    // whole run - see sphere_sphere.hh's own comment for why this is a global rather than
    // threaded through Environment.
    vamp::collision::self_collision_offset_m = kSelfCollisionOffsetM;

    std::cout << "Loading SDF environment...\n";
    auto env_input = load_environment();
    std::cout << "  " << env_input.sdf_grids.size() << " grids\n";

    std::cout << "Loading tool spheres...\n";
    env_input.attachments = load_tool_attachment(kCaseDir + "/tool_spheres/fine.yaml");
    std::cout << "  " << env_input.attachments->spheres.size() << " tool0 spheres\n";

    auto environment = EnvironmentVector(env_input);

    std::cout << "Loading joint sampling bounds...\n";
    const auto [joint_limit_lower, joint_limit_upper] = compute_joint_limits();

    nlohmann::json problem_json;
    std::ifstream(kCaseDir + "/problem.json") >> problem_json;
    Robot::Configuration start(config_from_json(problem_json.at("start")));
    Robot::Configuration goal(config_from_json(problem_json.at("goal")));

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

    vamp::planning::SimplifySettings simplify_settings;
    simplify_settings.operations = {vamp::planning::SHORTCUT};

    int num_success = 0;
    std::vector<float> costs;
    std::vector<double> wall_times_s;
    nlohmann::json runs = nlohmann::json::array();

    for (int i = 0; i < kNumRuns; i++)
    {
        std::cout << "Starting run " << i << "...\n";
        auto rng = std::make_shared<vamp::rng::CaseHalton<Robot>>(joint_limit_lower, joint_limit_upper);
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
    output["case"] = kCaseName;
    output["environment_representation"] = "sdf_grids (vamp::collision::SDFGrid, RSW-2740)";
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

    const auto now_s =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    const std::string out_path =
        "/home/gmruser/.ss_temp/p2p_benchmark_paths/vamp_sdf_benchmark_" + std::to_string(now_s) + ".json";
    std::ofstream out_file(out_path);
    out_file << output.dump(2);
    std::cout << "\nWrote " << out_path << "\n";

    return 0;
}
