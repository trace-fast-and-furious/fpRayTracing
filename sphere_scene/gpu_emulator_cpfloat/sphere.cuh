#ifndef SPHERE_BFP_H
#define SPHERE_BFP_H

#include "material.cuh"

using namespace custom_precision_fp;

class sphere : public hittable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(fp_orig_to_custom(r)), mat_ptr(m){};
    __device__ virtual bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const override;
    __device__ virtual bool hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const override;

    vec3 center;
    fp_custom radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const
{
    /* before: fp_custom */
    vec3 oc = r.origin() - center;
    fp_custom t_min = fp_orig_to_custom(__t_min);
    fp_custom t_max = fp_orig_to_custom(__t_max);
    fp_custom a = dot(r.direction(), r.direction());
    fp_custom b = dot(oc, r.direction());
    fp_custom c = dot(oc, oc) - radius * radius;
    fp_custom discriminant = b * b - a * c;
    if (discriminant > 0)
    {
        fp_custom temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

__device__ bool sphere::hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const
{
    /* before: fp_custom */
    vec3 oc = r.origin() - center;
    fp_custom a = dot(r.direction(), r.direction());
    fp_custom b = dot(oc, r.direction());
    fp_custom c = dot(oc, oc) - radius * radius;
    fp_custom discriminant = b * b - a * c;
    if (discriminant > 0)
    {
        fp_custom temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
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
