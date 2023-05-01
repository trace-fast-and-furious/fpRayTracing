#pragma once
#ifndef RAY_H
#define RAY_H

#include "color.h"

using namespace custom_precision_fp;

class ray
{
public:
    ray() {}
    ray(const point3 &origin, const point3 &direction, fp_custom time) : orig(origin), dir(direction), tm(time) {}
    ray(const point3 &origin, const point3 &direction, fp_orig time) : orig(origin), dir(direction), tm(fp_orig_to_custom(time)) {}

    point3 origin() const { return orig; }
    vec3 direction() const { return dir; }
    fp_custom time() const { return tm; }
    point3 at(fp_custom t) const { return orig + t * dir; }

    /* for mixed-precision */
    // point3_float origin_f() const { return vec3_fp_custom_to_fp_customo(orig); }
    // vec3_float direction_f() const { return vec3_fp_custom_to_fp_customo(dir); }
    // point3___fp_customo at(__fp_customo t) const
    // {
    //     return origin_f() + t * direction_f();
    // }

public:
    point3 orig;
    vec3 dir;
    fp_custom tm;
};

#endif