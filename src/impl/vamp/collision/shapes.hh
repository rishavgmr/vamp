#pragma once

#include <cstdint>
#include <string>
#include <memory>

#include <vamp/vector.hh>
#include <vamp/collision/math.hh>

#include <Eigen/Geometry>

namespace vamp::collision
{
    template <typename DataT>
    struct Shape
    {
        std::string name;
        DataT min_distance;

        Shape() = default;

        template <typename OtherDataT>
        explicit Shape(const Shape<OtherDataT> &other) : name(other.name), min_distance(other.min_distance)
        {
        }
    };

    // A cuboid, or rectangular prism, shape oriented in an arbitrary frame. The
    // cube is defined by its center (x, y, z), the three axes defining its frame
    // (axis_1_x, axis_1_y, axis_1_z, etc.), and the half-lengths along each axis
    // (axis_1_r, etc.)
    template <typename DataT>
    struct Cuboid : public Shape<DataT>
    {
        DataT x;
        DataT y;
        DataT z;

        DataT axis_1_x;
        DataT axis_1_y;
        DataT axis_1_z;
        DataT axis_2_x;
        DataT axis_2_y;
        DataT axis_2_z;
        DataT axis_3_x;
        DataT axis_3_y;
        DataT axis_3_z;

        DataT axis_1_r;
        DataT axis_2_r;
        DataT axis_3_r;

        inline constexpr auto compute_min_distance() -> DataT
        {
            auto d_1 = dot_3(-x, -y, -z, axis_1_x, axis_1_y, axis_1_z);
            auto d_2 = dot_3(-x, -y, -z, axis_2_x, axis_2_y, axis_2_z);
            auto d_3 = dot_3(-x, -y, -z, axis_3_x, axis_3_y, axis_3_z);

            auto v_1 = clamp(d_1, -axis_1_r, axis_1_r);
            auto v_2 = clamp(d_2, -axis_2_r, axis_2_r);
            auto v_3 = clamp(d_3, -axis_3_r, axis_3_r);

            auto x_n = x + axis_1_x * v_1 + axis_2_x * v_2 + axis_3_x * v_3;
            auto y_n = y + axis_1_y * v_1 + axis_2_y * v_2 + axis_3_y * v_3;
            auto z_n = z + axis_1_z * v_1 + axis_2_z * v_2 + axis_3_z * v_3;

            return sqrt(x_n * x_n + y_n * y_n + z_n * z_n);
        }

        Cuboid() = default;

        explicit Cuboid(
            DataT x,
            DataT y,
            DataT z,  //
            DataT axis_1_x,
            DataT axis_1_y,
            DataT axis_1_z,  //
            DataT axis_2_x,
            DataT axis_2_y,
            DataT axis_2_z,  //
            DataT axis_3_x,
            DataT axis_3_y,
            DataT axis_3_z,  //
            DataT axis_1_r,
            DataT axis_2_r,
            DataT axis_3_r)
          : Shape<DataT>()
          , x(x)
          , y(y)
          , z(z)
          , axis_1_x(axis_1_x)
          , axis_1_y(axis_1_y)
          , axis_1_z(axis_1_z)
          , axis_2_x(axis_2_x)
          , axis_2_y(axis_2_y)
          , axis_2_z(axis_2_z)
          , axis_3_x(axis_3_x)
          , axis_3_y(axis_3_y)
          , axis_3_z(axis_3_z)
          , axis_1_r(axis_1_r)
          , axis_2_r(axis_2_r)
          , axis_3_r(axis_3_r)
        {
            Shape<DataT>::min_distance = compute_min_distance();
        }

        template <typename OtherDataT>
        explicit Cuboid(const Cuboid<OtherDataT> &other)
          : Shape<DataT>(other)
          , x(other.x)
          , y(other.y)
          , z(other.z)
          , axis_1_x(other.axis_1_x)
          , axis_1_y(other.axis_1_y)
          , axis_1_z(other.axis_1_z)
          , axis_2_x(other.axis_2_x)
          , axis_2_y(other.axis_2_y)
          , axis_2_z(other.axis_2_z)
          , axis_3_x(other.axis_3_x)
          , axis_3_y(other.axis_3_y)
          , axis_3_z(other.axis_3_z)
          , axis_1_r(other.axis_1_r)
          , axis_2_r(other.axis_2_r)
          , axis_3_r(other.axis_3_r)
        {
        }
    };

    // A cylinder shape oriented in an arbitrary frame. The cylinder is defined by
    // two endpoints, which we store as the first point (x1, y1, z1) and the vector
    // from the first point to the second point (xv, yv, zv), and its radius (r)
    template <typename DataT>
    struct Cylinder : public Shape<DataT>
    {
        DataT x1;
        DataT y1;
        DataT z1;

        DataT xv;
        DataT yv;
        DataT zv;

        DataT r;

        // This is the reciprocal of the dot product of the vector between the
        // cylinder's endpoints, i.e. rdv = 1 / ([xv yv zv] . [xv yv zv]) We store it
        // for convenience to make some computations faster
        DataT rdv;

        inline constexpr auto x2() -> DataT
        {
            return x1 + xv;
        }

        inline constexpr auto y2() -> DataT
        {
            return y1 + yv;
        }

        inline constexpr auto z2() -> DataT
        {
            return z1 + zv;
        }

        inline constexpr auto compute_min_distance() -> DataT
        {
            auto dot = clamp(dot_3(-x1, -y1, -z1, xv, yv, zv) * rdv, 0.F, 1.F);

            auto xp = x1 + xv * dot;
            auto yp = y1 + yv * dot;
            auto zp = z1 + zv * dot;

            auto xo = -xp;
            auto yo = -yp;
            auto zo = -zp;

            auto ol = sqrt(dot_3(xo, yo, zo, xo, yo, zo));
            xo = xo / ol;
            yo = yo / ol;
            zo = zo / ol;

            auto ro = clamp(ol, 0.F, r);

            auto xn = xp + ro * xo;
            auto yn = yp + ro * yo;
            auto zn = zp + ro * zo;

            return sqrt(xn * xn + yn * yn + zn * zn);
        }

        Cylinder() = default;

        explicit Cylinder(
            DataT x1,
            DataT y1,
            DataT z1,  //
            DataT xv,
            DataT yv,
            DataT zv,  //
            DataT r,
            DataT rdv)
          : Shape<DataT>(), x1(x1), y1(y1), z1(z1), xv(xv), yv(yv), zv(zv), r(r), rdv(rdv)
        {
            Shape<DataT>::min_distance = compute_min_distance();
        }

        template <typename OtherDataT>
        explicit Cylinder(const Cylinder<OtherDataT> &other)
          : Shape<DataT>(other)
          , x1(other.x1)
          , y1(other.y1)
          , z1(other.z1)
          , xv(other.xv)
          , yv(other.yv)
          , zv(other.zv)
          , r(other.r)
          , rdv(other.rdv)
        {
        }
    };

    template <typename DataT>
    using Capsule = Cylinder<DataT>;

    // A sphere shape, represented as its center (x, y, z) and radius (r)
    template <typename DataT>
    struct Sphere : public Shape<DataT>
    {
        DataT x;
        DataT y;
        DataT z;
        DataT r;

        Sphere() = default;

        explicit Sphere(DataT x, DataT y, DataT z, DataT r) : Shape<DataT>(), x(x), y(y), z(z), r(r)
        {
            Shape<DataT>::min_distance = sqrt(x * x + y * y + z * z) - r;
        }

        template <typename OtherDataT>
        explicit Sphere(const Sphere<OtherDataT> &other)
          : Shape<DataT>(other), x(other.x), y(other.y), z(other.z), r(other.r)
        {
        }
    };

    // A heighfield. Defined by height pixel buffer.
    template <typename DataT>
    struct HeightField : public Shape<DataT>
    {
        DataT x;  // offset
        DataT y;
        DataT z;

        DataT xs;  // scaling factor
        DataT ys;
        DataT zs;

        std::size_t xd;  // image size
        std::size_t yd;

        std::size_t xd2;  // image size half
        std::size_t yd2;

        std::vector<float> data;  // flattened row-major data

        HeightField() = default;

        explicit HeightField(
            DataT x,
            DataT y,
            DataT z,
            DataT xs,
            DataT ys,
            DataT zs,
            std::size_t xd,
            std::size_t yd,
            const std::vector<float> &data)
          : Shape<DataT>()
          , x(x)
          , y(y)
          , z(z)
          , xs(xs)
          , ys(ys)
          , zs(zs)
          , xd(xd)
          , yd(yd)
          , xd2(xd / 2)
          , yd2(yd / 2)
          , data(data)
        {
            Shape<DataT>::min_distance = 0.0F;
        }

        template <typename OtherDataT>
        explicit HeightField(const HeightField<OtherDataT> &other)
          : Shape<DataT>(other)
          , x(other.x)
          , y(other.y)
          , z(other.z)
          , xs(other.xs)
          , ys(other.ys)
          , zs(other.zs)
          , xd(other.xd)
          , yd(other.yd)
          , xd2(other.xd2)
          , yd2(other.yd2)
          , data(other.data)
        {
        }
    };

    // RSW-2740: a dense signed-distance-field grid, for collision checking against voxelized
    // environment meshes (e.g. workpieces) without a triangle-mesh representation. Ported from
    // pRRTC's own ppln::collision::SDFGrid (src/collision/sdf_environment.hh in the pRRTC fork) -
    // same fields, same x100-scaled distance-value convention (grid values and offset_scaled are
    // in that scaled unit; bounds/spacing stay in plain meters), same world-to-local rigid
    // transform convention. `data` is always stored as float here (unlike pRRTC's int16-or-float
    // dual dtype) - converted once at load time, since there's no SIMD benefit to keeping the
    // narrower storage on the CPU/SIMD side the way there is saving GPU memory bandwidth.
    template <typename DataT>
    struct SDFGrid : public Shape<DataT>
    {
        DataT bounds_lower_x, bounds_lower_y, bounds_lower_z;
        DataT bounds_upper_x, bounds_upper_y, bounds_upper_z;
        DataT spacing;
        std::size_t num_x, num_y, num_z;
        DataT offset_scaled;

        // World-to-local rigid transform - identity (row0=(1,0,0), row1=(0,1,0), row2=(0,0,1),
        // inv_translation=(0,0,0)) for a grid placed at the world origin with no rotation, a
        // pure no-op in that case. See pRRTC's own SDFGrid comment for the full reasoning.
        DataT inv_rotation_row0_x, inv_rotation_row0_y, inv_rotation_row0_z;
        DataT inv_rotation_row1_x, inv_rotation_row1_y, inv_rotation_row1_z;
        DataT inv_rotation_row2_x, inv_rotation_row2_y, inv_rotation_row2_z;
        DataT inv_translation_x, inv_translation_y, inv_translation_z;

        std::vector<float> data;  // flattened, X fastest / Z slowest (matches pRRTC's sdf_flat_idx)

        // RSW-2740: bit `i` set means cricket's per-link environment check for link_index `i`
        // (see ccfk_template.hh's per-link block) skips this grid entirely, mirroring pRRTC's
        // own fanucm710_link_env_mask (uploadSDFEnvironment's link_env_mask param) - both driven
        // by the same scene_config_m710.json collisionMask entries. Not part of either
        // constructor below (set via plain member assignment after construction, same as
        // `.name` already is in the benchmark's load_environment()) since it's populated from a
        // separate data source (collisionMask) than the grid's own geometry. Default 0 = this
        // grid is checked against every link, matching prior (unmasked) behavior.
        std::uint32_t masked_link_bitmask = 0;

        SDFGrid() = default;

        explicit SDFGrid(
            DataT bounds_lower_x,
            DataT bounds_lower_y,
            DataT bounds_lower_z,
            DataT bounds_upper_x,
            DataT bounds_upper_y,
            DataT bounds_upper_z,
            DataT spacing,
            std::size_t num_x,
            std::size_t num_y,
            std::size_t num_z,
            DataT offset_scaled,
            DataT inv_rotation_row0_x,
            DataT inv_rotation_row0_y,
            DataT inv_rotation_row0_z,
            DataT inv_rotation_row1_x,
            DataT inv_rotation_row1_y,
            DataT inv_rotation_row1_z,
            DataT inv_rotation_row2_x,
            DataT inv_rotation_row2_y,
            DataT inv_rotation_row2_z,
            DataT inv_translation_x,
            DataT inv_translation_y,
            DataT inv_translation_z,
            const std::vector<float> &data)
          : Shape<DataT>()
          , bounds_lower_x(bounds_lower_x)
          , bounds_lower_y(bounds_lower_y)
          , bounds_lower_z(bounds_lower_z)
          , bounds_upper_x(bounds_upper_x)
          , bounds_upper_y(bounds_upper_y)
          , bounds_upper_z(bounds_upper_z)
          , spacing(spacing)
          , num_x(num_x)
          , num_y(num_y)
          , num_z(num_z)
          , offset_scaled(offset_scaled)
          , inv_rotation_row0_x(inv_rotation_row0_x)
          , inv_rotation_row0_y(inv_rotation_row0_y)
          , inv_rotation_row0_z(inv_rotation_row0_z)
          , inv_rotation_row1_x(inv_rotation_row1_x)
          , inv_rotation_row1_y(inv_rotation_row1_y)
          , inv_rotation_row1_z(inv_rotation_row1_z)
          , inv_rotation_row2_x(inv_rotation_row2_x)
          , inv_rotation_row2_y(inv_rotation_row2_y)
          , inv_rotation_row2_z(inv_rotation_row2_z)
          , inv_translation_x(inv_translation_x)
          , inv_translation_y(inv_translation_y)
          , inv_translation_z(inv_translation_z)
          , data(data)
        {
            Shape<DataT>::min_distance = 0.0F;
        }

        template <typename OtherDataT>
        explicit SDFGrid(const SDFGrid<OtherDataT> &other)
          : Shape<DataT>(other)
          , bounds_lower_x(other.bounds_lower_x)
          , bounds_lower_y(other.bounds_lower_y)
          , bounds_lower_z(other.bounds_lower_z)
          , bounds_upper_x(other.bounds_upper_x)
          , bounds_upper_y(other.bounds_upper_y)
          , bounds_upper_z(other.bounds_upper_z)
          , spacing(other.spacing)
          , num_x(other.num_x)
          , num_y(other.num_y)
          , num_z(other.num_z)
          , offset_scaled(other.offset_scaled)
          , inv_rotation_row0_x(other.inv_rotation_row0_x)
          , inv_rotation_row0_y(other.inv_rotation_row0_y)
          , inv_rotation_row0_z(other.inv_rotation_row0_z)
          , inv_rotation_row1_x(other.inv_rotation_row1_x)
          , inv_rotation_row1_y(other.inv_rotation_row1_y)
          , inv_rotation_row1_z(other.inv_rotation_row1_z)
          , inv_rotation_row2_x(other.inv_rotation_row2_x)
          , inv_rotation_row2_y(other.inv_rotation_row2_y)
          , inv_rotation_row2_z(other.inv_rotation_row2_z)
          , inv_translation_x(other.inv_translation_x)
          , inv_translation_y(other.inv_translation_y)
          , inv_translation_z(other.inv_translation_z)
          , data(other.data)
          , masked_link_bitmask(other.masked_link_bitmask)
        {
        }
    };
}  // namespace vamp::collision
