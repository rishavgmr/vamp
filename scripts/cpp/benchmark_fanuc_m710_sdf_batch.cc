// RSW-2740 Phase 3: BATCH sibling of benchmark_fanuc_m710_sdf.cc - same env/tool/joint-limit
// loading and the same SDF-based collision-checking (see that file's own header comment for the
// details), but solves a whole batch of (start, goal) pairs from this case's
// problems_batch.json CONCURRENTLY (one std::thread per pair, each solved once) instead of
// repeatedly re-solving a single fixed pair from problem.json. Verifies VAMP's solve()
// reentrancy fix (RSW-2740 Phase 1: SDFGrid::data as shared_ptr, one Environment copy per
// thread) under real concurrent load and reports the actual speedup over solving the same batch
// serially, not just correctness.
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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <string>
#include <thread>
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
    // Not a structured binding (unlike the single-problem benchmark's own version of this line):
    // capturing a structured binding by reference in the solve_one lambda below is a C++20
    // extension, not valid C++17 - plain named variables from .first/.second capture cleanly.
    const auto joint_limits = compute_joint_limits();
    const std::array<float, 7> &joint_limit_lower = joint_limits.first;
    const std::array<float, 7> &joint_limit_upper = joint_limits.second;

    vamp::planning::RRTCSettings rrtc_settings;
    rrtc_settings.range = 0.5F;
    rrtc_settings.dynamic_domain = true;
    rrtc_settings.radius = 4.0F;
    rrtc_settings.alpha = 0.0001F;
    rrtc_settings.min_radius = 1.0F;
    rrtc_settings.balance = true;
    rrtc_settings.tree_ratio = 1.0F;
    // Same as the single-problem benchmark's own 5000/5000. RSW-2740: this was WRONGLY raised to
    // 20000 at one point, based on a flawed diagnosis - a batch of 38 problems was seeing 24
    // exhaust their budget (rrtc_iterations=5001), which looked like "VAMP just needs more
    // samples than pRRTC's same nominal max_iters=5000" (plausible-sounding: pRRTC evaluates 512
    // candidate extensions in parallel per iteration on the GPU, VAMP's CPU search does far less
    // per iteration). That reasoning was never actually tested in isolation - every time the
    // budget was raised, vamp_case_setup.py was also just re-run in the same breath, and ITS
    // real bug (rebuilding only benchmark_fanuc_m710_sdf, silently leaving this target's
    // fanuc_m710.hh stale - see vamp_case_setup.py's own history) was the actual cause of the
    // failures: wrong baked-in robot geometry produced spurious collisions that made otherwise-
    // easy problems look infeasible. Once that build bug was fixed, the ORIGINAL 5000/5000 solves
    // the same 38-problem batch 38/38, with iteration counts nowhere near the ceiling (mostly
    // 0-300). Left at 5000/5000 - don't re-raise this without first confirming
    // fanuc_m710.hh actually matches VAMP_BENCHMARK_CASE for the run that's failing.
    rrtc_settings.max_iterations = 5000;
    rrtc_settings.max_samples = 5000;

    vamp::planning::SimplifySettings simplify_settings;
    simplify_settings.operations = {vamp::planning::SHORTCUT};

    nlohmann::json batch_json;
    std::ifstream(kCaseDir + "/problems_batch.json") >> batch_json;
    const nlohmann::json &problems = batch_json.at("problems");
    const std::size_t num_problems = problems.size();

    std::vector<Robot::Configuration> starts(num_problems);
    std::vector<Robot::Configuration> goals(num_problems);
    for (std::size_t i = 0; i < num_problems; i++)
    {
        starts[i] = Robot::Configuration(config_from_json(problems[i].at("start")));
        goals[i] = Robot::Configuration(config_from_json(problems[i].at("goal")));
    }
    std::cout << "Loaded " << num_problems << " problems from problems_batch.json\n";

    std::mutex cout_mutex;

    // Each call gets its own Environment copy (RSW-2740 Phase 1 - cheap since SDFGrid::data is a
    // shared_ptr) and its own CaseHalton RNG instance (has per-instance mutable state advanced by
    // next(), can't be shared across threads any more than the Environment can). Shared by both
    // passes below - the concurrent pass calls this from N threads at once; the serial baseline
    // pass calls it N times in a row on the main thread. Same logic, so the only difference
    // between the two passes is real: contention or none.
    auto solve_one = [&](std::size_t idx, std::vector<nlohmann::json> &out, const char *phase)
    {
        Robot::Configuration start = starts[idx];
        Robot::Configuration goal = goals[idx];
        EnvironmentVector thread_env = environment;
        auto rng = std::make_shared<vamp::rng::CaseHalton<Robot>>(joint_limit_lower, joint_limit_upper);

        auto t0 = std::chrono::steady_clock::now();
        auto result = RRTC::solve(start, goal, thread_env, rrtc_settings, rng);
        bool solved = !result.path.empty();
        double cost_before_shortcut = solved ? path_cost(result.path) : 0.0;

        vamp::planning::PlanningResult<Robot> simplify_result;
        if (solved)
        {
            simplify_result = vamp::planning::simplify<Robot, rake, Robot::resolution>(
                result.path, thread_env, simplify_settings, rng);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double planning_time_s = std::chrono::duration<double>(t1 - t0).count();
        const double cost_after_shortcut = solved ? path_cost(simplify_result.path) : 0.0;

        nlohmann::json run;
        run["problem_index"] = idx;
        run["success"] = solved;
        run["planning_time_s"] = planning_time_s;
        run["rrtc_iterations"] = result.iterations;
        run["start_config"] = config_to_json(start);
        run["goal_config"] = config_to_json(goal);
        if (solved)
        {
            run["path_cost"] = cost_after_shortcut;
            run["path_cost_before_shortcut"] = cost_before_shortcut;
            // result.size[0]/[1] = start_tree.size()/goal_tree.size() (rrtc.hh) - directly
            // comparable to pRRTC's start_tree_size/goal_tree_size fields, unlike iterations
            // (pRRTC's iteration evaluates num_new_configs candidates in parallel, VAMP's does
            // not - not the same unit of work).
            run["start_tree_size"] = result.size[0];
            run["goal_tree_size"] = result.size[1];
            nlohmann::json path = nlohmann::json::array();
            for (const auto &cfg : simplify_result.path)
            {
                path.push_back(config_to_json(cfg));
            }
            run["path"] = path;
        }
        out[idx] = std::move(run);

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[" << phase << "] problem " << idx << ": solved=" << solved
                       << " cost=" << cost_after_shortcut << " planning_time_ms=" << planning_time_s * 1e3
                       << " rrtc_iterations=" << result.iterations << "\n";
        }
    };

    std::cout << "=== Concurrent pass: launching " << num_problems << " solve() threads at once ===\n";
    std::vector<nlohmann::json> concurrent_results(num_problems);
    const auto batch_t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(num_problems);
    for (std::size_t i = 0; i < num_problems; i++)
    {
        threads.emplace_back(solve_one, i, std::ref(concurrent_results), "concurrent");
    }
    for (auto &t : threads)
    {
        t.join();
    }
    const auto batch_t1 = std::chrono::steady_clock::now();
    const double batch_wall_time_s = std::chrono::duration<double>(batch_t1 - batch_t0).count();

    // RSW-2740: the concurrent pass's own per-thread planning_time_s values are inflated by
    // however much the N threads contended with each other (CPU cores, cache, memory bandwidth) -
    // their sum is NOT a valid estimate of "time to solve these serially". To get an honest
    // speedup number at any batch size, actually solve the same problems one at a time here and
    // compare that real total against batch_wall_time_s.
    std::cout << "=== Serial baseline pass: solving " << num_problems << " problems one at a time ===\n";
    std::vector<nlohmann::json> serial_results(num_problems);
    const auto serial_t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < num_problems; i++)
    {
        solve_one(i, serial_results, "serial");
    }
    const auto serial_t1 = std::chrono::steady_clock::now();
    const double serial_wall_time_s = std::chrono::duration<double>(serial_t1 - serial_t0).count();

    int num_success_concurrent = 0;
    int num_success_serial = 0;
    double concurrent_sum_planning_time_s = 0.0;
    for (const auto &r : concurrent_results)
    {
        concurrent_sum_planning_time_s += r.at("planning_time_s").get<double>();
        if (r.at("success").get<bool>())
        {
            ++num_success_concurrent;
        }
    }
    for (const auto &r : serial_results)
    {
        if (r.at("success").get<bool>())
        {
            ++num_success_serial;
        }
    }

    // The real number: how much the true concurrent wall time beat actually solving the same
    // batch one at a time. >1 means threading them is worth it; <=1 means this machine is
    // already saturated at this batch size and running them sequentially would have been as
    // fast or faster.
    const double speedup = batch_wall_time_s > 0.0 ? serial_wall_time_s / batch_wall_time_s : 0.0;

    std::cout << "\nBatch summary: " << num_success_concurrent << "/" << num_problems
               << " solved concurrently, " << num_success_serial << "/" << num_problems << " solved serially\n"
               << "  batch_wall_time_s (concurrent, true)  = " << batch_wall_time_s << "\n"
               << "  serial_wall_time_s (true baseline)    = " << serial_wall_time_s << "\n"
               << "  concurrent_sum_planning_time_s (diag) = " << concurrent_sum_planning_time_s
               << "  (contention-inflated, NOT a serial estimate)\n"
               << "  speedup (serial_wall_time_s / batch_wall_time_s) = " << speedup << "x\n";

    nlohmann::json output;
    output["algorithm"] = "VAMP";
    output["robot"] = "fanuc_m710";
    output["case"] = kCaseName;
    output["environment_representation"] = "sdf";
    output["num_problems"] = num_problems;
    output["success_rate_concurrent"] = static_cast<double>(num_success_concurrent) / static_cast<double>(num_problems);
    output["success_rate_serial"] = static_cast<double>(num_success_serial) / static_cast<double>(num_problems);
    output["batch_wall_time_s"] = batch_wall_time_s;
    output["serial_wall_time_s"] = serial_wall_time_s;
    output["concurrent_sum_planning_time_s"] = concurrent_sum_planning_time_s;
    output["speedup"] = speedup;
    output["problems_concurrent"] = nlohmann::json(concurrent_results);
    output["problems_serial"] = nlohmann::json(serial_results);

    const auto now_s =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    const std::string out_path =
        "/home/gmruser/.ss_temp/p2p_benchmark_paths/vamp_batch_benchmark_" + std::to_string(now_s) + ".json";
    std::ofstream out_file(out_path);
    out_file << output.dump(2);
    std::cout << "\nWrote " << out_path << "\n";

    return 0;
}
