#pragma once
#ifndef RAYH
#define RAYH
#include "vec3.h"

using namespace bfloat16;

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
        __device__ vec3 origin() const       { return A; }
        __device__ vec3 direction() const    { return B; }
        __device__ vec3 point_at_parameter(float t) const { return A + t * B; }
        __device__ vec3 point_at_parameter(__nv_bfloat16 t) const { return A + t * B; }

        vec3 A;
        vec3 B;
};

#endif