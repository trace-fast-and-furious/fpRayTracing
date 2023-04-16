#pragma once
#ifndef RAY_BFP_H
#define RAY_BFP_H

#include "color.h"

using namespace floating_point;

class ray
{
public:
    ray() {}
    ray(const point3 &origin, const point3 &direction, fp time) : orig(origin), dir(direction), tm(time) {}

    point3 origin() const { return orig; }
    vec3 direction() const { return dir; }
    fp time() const { return tm; }
    __fpo time_f() const { return fp_to_fpo(tm); }
    point3 at(fp t) const { return orig + t * dir; }

    /* for mixed-precision */
    // point3_float origin_f() const { return vec3_fp_to_fpo(orig); }
    // vec3_float direction_f() const { return vec3_fp_to_fpo(dir); }
    // point3___fpo at(__fpo t) const
    // {
    //     return origin_f() + t * direction_f();
    // }

public:
    point3 orig;
    vec3 dir;
    fp tm;
};

#endif