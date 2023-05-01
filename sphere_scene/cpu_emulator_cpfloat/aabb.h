#ifndef AABB_BFP_H
#define AABB_BFP_H

#include "camera.h"

using namespace custom_precision_fp;

class aabb
{
public:
    aabb() {}
    aabb(const point3 &a, const point3 &b)
    {
        minimum = a;
        maximum = b;
    }

    point3 min() const { return minimum; }
    point3 max() const { return maximum; }

    bool hit(const ray &r, fp_custom t_min, fp_custom t_max) const
    {
        for (int a = 0; a < 3; a++)
        {
            fp_custom t0 = custom_precision_fp::min((minimum[a] - r.origin()[a]) / r.direction()[a], (maximum[a] - r.origin()[a]) / r.direction()[a]);
            fp_custom t1 = custom_precision_fp::min((minimum[a] - r.origin()[a]) / r.direction()[a], (maximum[a] - r.origin()[a]) / r.direction()[a]);

            t_min = custom_precision_fp::max(t0, t_min);
            t_max = custom_precision_fp::max(t1, t_max);

            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max) const
    {
        fp_custom t_min = fp_orig_to_custom(__t_min);
        fp_custom t_max = fp_orig_to_custom(__t_max);

        for (int a = 0; a < 3; a++)
        {
            fp_custom t0 = custom_precision_fp::min((minimum[a] - r.origin()[a]) / r.direction()[a], (maximum[a] - r.origin()[a]) / r.direction()[a]);
            fp_custom t1 = custom_precision_fp::min((minimum[a] - r.origin()[a]) / r.direction()[a], (maximum[a] - r.origin()[a]) / r.direction()[a]);

            t_min = custom_precision_fp::max(t0, t_min);
            t_max = custom_precision_fp::max(t1, t_max);

            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    point3 minimum;
    point3 maximum;
};

inline aabb surrounding_box(aabb box0, aabb box1)
{
    point3 small(min(box0.min().x(), box1.min().x()), min(box0.min().y(), box1.min().y()), min(box0.min().z(), box1.min().z()));
    point3 big(max(box0.max().x(), box1.max().x()), max(box0.max().y(), box1.max().y()), max(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}

#endif