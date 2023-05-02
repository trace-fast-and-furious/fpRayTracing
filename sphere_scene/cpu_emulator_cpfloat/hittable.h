#ifndef HITTABLE_H
#define HITTABLE_H
#include <memory>

#include "aabb.h"

using namespace custom_precision_fp;

using std::make_shared;
using std::shared_ptr;

class material;

struct hit_record
{
    point3 p;
    vec3 normal;
    shared_ptr<material> mat_ptr;
    fp_custom t;
    bool front_face;

    inline void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    virtual bool hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const = 0;
    virtual bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const = 0;
    virtual bool bounding_box(fp_custom time0, fp_custom time1, aabb &output_box) const = 0;
    virtual bool bounding_box(fp_orig __time0, fp_orig __time1, aabb &output_box) const = 0;
};

#endif