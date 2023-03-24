#ifndef HITABLEH
#define HITABLEH

#include "camera.h"

using namespace bfloat16;

class material;

struct hit_record
{
    __nv_bfloat16 t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hittable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif

