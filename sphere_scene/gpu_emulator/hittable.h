#ifndef HITTABLE_BFP_H
#define HITTABLE_BFP_H

#include <memory>

#include "aabb.h"

using namespace floating_point;

using std::make_shared;
using std::shared_ptr;

class material;

struct hit_record
{
    point3 p;
    vec3 normal;
    shared_ptr<material> mat_ptr;
    fp t;
    bool front_face;

    inline void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < b_0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    virtual bool hit(const ray &r, fp t_min, fp t_max, hit_record &rec) const = 0;
    virtual bool hit(const ray &r, __fpo __t_min, __fpo __t_max, hit_record &rec) const = 0;
    virtual bool bounding_box(fp time0, fp time1, aabb &output_box) const = 0;
    virtual bool bounding_box(__fpo __time0, __fpo __time1, aabb &output_box) const = 0;
};

#endif