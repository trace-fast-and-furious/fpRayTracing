#ifndef UTILITY_H
#define UTILITY_H

#include "cpfloat.h"

using namespace custom_precision_fp;

inline fp_custom degrees_to_radians(fp_custom degrees)
{
    return degrees * (cp_PI / 180);
}

inline fp_orig random_orig()
{
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline fp_orig random_orig(fp_orig min, fp_orig max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_orig();
}

inline fp_custom random_num()
{
    return fp_orig_to_custom(random_orig());
}

inline fp_custom random_num(fp_custom min, fp_custom max)
{
    return min + (max - min) * random_num();
}

inline fp_orig random_num(fp_orig min, fp_orig max)
{
    return random_orig(min, max);
}

inline fp_orig clamp(fp_orig x, fp_orig min, fp_orig max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

inline fp_orig clamp(fp_custom x, fp_orig min, fp_orig max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x.val[0];
}

#endif