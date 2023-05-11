#ifndef UTILITY_BFP_H
#define UTILITY_BFP_H

#include "fp.h"

using namespace floating_point;

inline fp degrees_to_radians(fp degrees)
{
    return degrees * (b_pi / b_180);
}

inline __fpo random_fpo()
{
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline __fpo random_fpo(__fpo min, __fpo max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_fpo();
}

inline fp random_num()
{
    return fpo_to_fp(random_fpo());
}

inline fp random_num(fp min, fp max)
{
    __fpo f_min = fp_to_fpo(min);
    __fpo f_max = fp_to_fpo(max);

    return fpo_to_fp(random_fpo(f_min, f_max));
}

inline __fpo clamp(__fpo x, __fpo min, __fpo max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

inline fp clamp(fp x, fp min, fp max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

#endif