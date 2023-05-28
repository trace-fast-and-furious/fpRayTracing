#ifndef HITTABLE_BFP_H
#define HITTABLE_BFP_H

#include "camera.cuh"

using namespace custom_precision_fp;

using std::make_shared;
using std::shared_ptr;

class material;

struct hit_record
{
    fp_custom t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hittable
{
public:
    __device__ virtual bool hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const = 0;
    __device__ virtual bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const = 0;
};

#endif