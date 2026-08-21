#pragma once

#include <vamp/collision/shapes.hh>
#include <vamp/collision/math.hh>

namespace vamp::collision
{

    template <typename DataT>
    inline constexpr auto sphere_sphere_sql2(
        const DataT &ax,
        const DataT &ay,
        const DataT &az,
        const DataT &ar,
        const DataT &bx,
        const DataT &by,
        const DataT &bz,
        const DataT &br) noexcept -> DataT
    {
        auto sum = sql2_3(ax, ay, az, bx, by, bz);
        auto rs = ar + br;
        return sum - rs * rs;
    }

    template <typename DataT>
    inline constexpr auto sphere_sphere_sql2(
        const Sphere<DataT> &a,
        const DataT &x,
        const DataT &y,
        const DataT &z,
        const DataT &r) noexcept -> DataT
    {
        return sphere_sphere_sql2(a.x, a.y, a.z, a.r, x, y, z, r);
    }

    template <typename DataT>
    inline constexpr auto sphere_sphere_sql2(const Sphere<DataT> &a, const Sphere<DataT> &b) noexcept -> DataT
    {
        return sphere_sphere_sql2(a, b.x, b.y, b.z, b.r);
    }

    template <typename DataT>
    inline constexpr auto sphere_sphere_l2(const Sphere<DataT> &a, const Sphere<DataT> &b) noexcept -> DataT
    {
        auto sum = sql2_3(a.x, a.y, a.z, b.x, b.y, b.z).sqrt();
        return sum - (a.r + b.r);
    }

    // RSW-2740: runtime self-collision clearance margin (meters), mirroring pRRTC's own
    // fanucm710_self_collision_offset (uploadSDFEnvironment's self_collision_offset_m param,
    // src/robots/fanuc_m710_benchmark.cuh's rs = r1 + r2 + offset). A plain global, not a
    // parameter, because sphere_sphere_self_collision()'s generated call sites
    // (ccfk_template.hh) pass only sphere coordinates - there's no Environment/robot struct to
    // carry per-case state on. Scoped to arm self-collision only (this file, not
    // sphere_sphere_sql2 above, which plain sphere-vs-query-point checks still use unmodified) -
    // matches pRRTC's own scope exactly: its offset is applied only inside
    // self_collision_check_approx/self_collision_check, never to attachment/tool-vs-arm checks
    // (verified by reading both call sites in fanuc_m710_benchmark.cuh). Default 0 = no margin,
    // same as the unset behavior before this existed.
    inline float self_collision_offset_m = 0.0F;

    template <typename DataT>
    inline constexpr auto sphere_sphere_sql2_self(
        const DataT &ax,
        const DataT &ay,
        const DataT &az,
        const DataT &ar,
        const DataT &bx,
        const DataT &by,
        const DataT &bz,
        const DataT &br) noexcept -> DataT
    {
        auto sum = sql2_3(ax, ay, az, bx, by, bz);
        auto rs = ar + br + DataT::fill(self_collision_offset_m);
        return sum - rs * rs;
    }
}  // namespace vamp::collision
