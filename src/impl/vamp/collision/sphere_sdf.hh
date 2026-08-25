#pragma once

#include <vamp/collision/shapes.hh>
#include <vamp/collision/math.hh>

namespace vamp::collision
{
    // RSW-2740: vectorized port of pRRTC's sdf_lookup() (src/collision/sdf_environment.hh in the
    // pRRTC fork, itself ported from PhysxNode::PxSDFSampleImpl) - same world-to-local transform,
    // same clamp/floor/trilinear-blend math, same x100-scaled distance convention and
    // out-of-bounds clamp-distance penalty. Differs only in how the 8 corner voxel values get
    // read: pRRTC's GPU version reads one query at a time per thread, so it just indexes
    // directly; here, `rake` queries are packed into one DataT, so each of the 8 corners is one
    // DataT::gather() call pulling `rake` different values from the scalar grid.data array at
    // once (the same primitive HeightField's own sphere_heightfield.hh uses for its single
    // corner) - 8 gathers instead of 1, since this does real trilinear interpolation rather than
    // HeightField's nearest-neighbor lookup.
    template <typename DataT>
    inline auto sphere_sdf(
        const SDFGrid<DataT> &grid,
        const DataT &sx,
        const DataT &sy,
        const DataT &sz,
        const DataT &sr) noexcept -> DataT
    {
        using IndexT = IntVector<DataT::num_scalars_per_row, DataT::num_rows>;

        // World -> this grid's local frame. A no-op for an identity-placed grid.
        auto lx = grid.inv_rotation_row0_x * sx + grid.inv_rotation_row0_y * sy + grid.inv_rotation_row0_z * sz +
                  grid.inv_translation_x;
        auto ly = grid.inv_rotation_row1_x * sx + grid.inv_rotation_row1_y * sy + grid.inv_rotation_row1_z * sz +
                  grid.inv_translation_y;
        auto lz = grid.inv_rotation_row2_x * sx + grid.inv_rotation_row2_y * sy + grid.inv_rotation_row2_z * sz +
                  grid.inv_translation_z;

        auto clamped_x = lx.clamp(grid.bounds_lower_x, grid.bounds_upper_x);
        auto clamped_y = ly.clamp(grid.bounds_lower_y, grid.bounds_upper_y);
        auto clamped_z = lz.clamp(grid.bounds_lower_z, grid.bounds_upper_z);

        auto dx = lx - clamped_x;
        auto dy = ly - clamped_y;
        auto dz = lz - clamped_z;
        auto diff_sq = dx * dx + dy * dy + dz * dz;

        auto inv_spacing = DataT::fill(1.0F) / grid.spacing;
        auto fx_cont = (clamped_x - grid.bounds_lower_x) * inv_spacing;
        auto fy_cont = (clamped_y - grid.bounds_lower_y) * inv_spacing;
        auto fz_cont = (clamped_z - grid.bounds_lower_z) * inv_spacing;

        // pRRTC's own spacing is derived from the X extent alone (grid["spacing"] = (upper.x -
        // lower.x) / numX in sdf_check.py, mirrored above) and then reused for all three axes -
        // for a non-cubic voxel grid (Y/Z extent not an exact multiple of that X-derived
        // spacing), fy_cont/fz_cont at the upper bound legitimately land short of (or past)
        // dim-1, not just barely above it. So the continuous coordinate itself must be clamped
        // to [0, dim-1] before flooring - clamping only the floored *index* (as an earlier
        // version of this function did) leaves the fractional part free to exceed 1.0, which is
        // wrong: pRRTC's own branch ("if (i >= numX-1) { i = numX-2; fx = 1.0; }") always forces
        // exactly fx=1.0 in this case, and this clamp-then-floor ordering reproduces that
        // losslessly without a per-lane select. Verified against sdf_check.py's sdf_lookup() on
        // workpiece.bin, including this exact non-cubic-voxel boundary case.
        auto fx_c = fx_cont.clamp(0.0F, static_cast<float>(grid.num_x) - 1.0F);
        auto fy_c = fy_cont.clamp(0.0F, static_cast<float>(grid.num_y) - 1.0F);
        auto fz_c = fz_cont.clamp(0.0F, static_cast<float>(grid.num_z) - 1.0F);

        auto ix = fx_c.floor().clamp(0.0F, static_cast<float>(grid.num_x) - 2.0F);
        auto iy = fy_c.floor().clamp(0.0F, static_cast<float>(grid.num_y) - 2.0F);
        auto iz = fz_c.floor().clamp(0.0F, static_cast<float>(grid.num_z) - 2.0F);

        auto fx = fx_c - ix;
        auto fy = fy_c - iy;
        auto fz = fz_c - iz;

        IndexT ix_i = ix.template to<IndexT>();
        IndexT iy_i = iy.template to<IndexT>();
        IndexT iz_i = iz.template to<IndexT>();

        // X fastest, Z slowest - matches pRRTC's sdf_flat_idx.
        IndexT dim_x = IndexT::fill(static_cast<int>(grid.num_x));
        IndexT dim_xy = IndexT::fill(static_cast<int>(grid.num_x * grid.num_y));
        IndexT base = ix_i + iy_i * dim_x + iz_i * dim_xy;

        const float *data = grid.data->data();
        auto s000 = DataT::gather(data, base);
        auto s100 = DataT::gather(data, base + IndexT::fill(1));
        auto s010 = DataT::gather(data, base + dim_x);
        auto s110 = DataT::gather(data, base + dim_x + IndexT::fill(1));
        auto s001 = DataT::gather(data, base + dim_xy);
        auto s101 = DataT::gather(data, base + dim_xy + IndexT::fill(1));
        auto s011 = DataT::gather(data, base + dim_xy + dim_x);
        auto s111 = DataT::gather(data, base + dim_xy + dim_x + IndexT::fill(1));

        auto one = DataT::fill(1.0F);
        auto c00 = s000 * (one - fx) + s100 * fx;
        auto c10 = s010 * (one - fx) + s110 * fx;
        auto c01 = s001 * (one - fx) + s101 * fx;
        auto c11 = s011 * (one - fx) + s111 * fx;
        auto c0 = c00 * (one - fy) + c10 * fy;
        auto c1 = c01 * (one - fy) + c11 * fy;
        auto dist = c0 * (one - fz) + c1 * fz;

        // Out-of-bounds clamp-distance penalty, same x100-scaled convention as pRRTC's own.
        // diff_sq is exactly 0 for the overwhelmingly common in-bounds case, but DataT::sqrt()
        // is v * rsqrt(v) (vector/isa/avx.hh) - at v=0 that's 0 * inf = NaN, poisoning every
        // in-bounds query. Nudging away from exact zero costs sqrt(1e-12)*100 ~= 1e-4 (cm-scale
        // units), far below anything that could flip a collision verdict, and avoids relying on
        // a per-lane branch/select just to dodge one library edge case.
        dist = dist + (diff_sq + DataT::fill(1e-12F)).sqrt() * DataT::fill(100.0F);

        // Same comparison pRRTC's env-check makes: dist - radius_scaled < offset_scaled means
        // collision. Return positive-means-safe / negative-means-collision, matching every other
        // shape check in this file (sphere_heightfield, sphere_cuboid, ...), so the caller's
        // existing `not (...).test_zero()` convention just works.
        return dist - sr * DataT::fill(100.0F) - grid.offset_scaled;
    }
}  // namespace vamp::collision
