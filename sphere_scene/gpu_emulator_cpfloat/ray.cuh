#pragma once
#ifndef RAY_H
#define RAY_H

#include "vec3.cuh"

using namespace custom_precision_fp;

class ray
{
public:
    __device__ __host__ ray() {}
    __device__ __host__ ray(const vec3 &a, const vec3 &b)
    {
        A = a;
        B = b;
    }
    __device__ __host__ vec3 origin() const { return A; }
    __device__ __host__ vec3 direction() const { return B; }
    __device__ __host__ vec3 point_at_parameter(fp_orig t) const { return A + t * B; }
    __device__ __host__ vec3 point_at_parameter(fp_custom t) const { return A + t * B; }

    vec3 A;
    vec3 B;
};

#endif