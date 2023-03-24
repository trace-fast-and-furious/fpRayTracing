#ifndef SPHEREH
#define SPHEREH

#include "material.h"

using namespace bfloat16;

class sphere : public hittable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(__float2bfloat16(r)), mat_ptr(m){};
    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;

    vec3 center;
    __nv_bfloat16 radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    __nv_bfloat16 a = dot(r.direction(), r.direction());
    __nv_bfloat16 b = dot(oc, r.direction());
    __nv_bfloat16 c = dot(oc, oc) - radius * radius;
    __nv_bfloat16 discriminant = b * b - a * c;
    if (discriminant > 0) {
        __nv_bfloat16 temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


#endif

