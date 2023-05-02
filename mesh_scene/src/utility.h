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
// /*
//  * ===================================================
//  *
//  *       Filename:  utility.h
//  *    Description:  Ray Tracing In One Weekend (RTIOW): Final Code
//  *        Created:  2022/07/13
//  *
//  * ===================================================
//  */


// // Preprocessors
// #pragma once

// #include <cmath>
// #include <limits>
// #include <memory>
// #include <cstdlib>


// // Usings

// using std::shared_ptr;
// using std::make_shared;
// using std::sqrt;


// // Constants

// const double infinity = std::numeric_limits<double>::infinity();
// const double pi = 3.1415926535897932385;


// // Utility Functions

// inline double degrees_to_radians(double degrees) {
//        	return degrees * pi / 180.0;
// }

// inline double random_double() {
// 	// Returns a random real in [0,1).
// 	return rand() / (RAND_MAX + 1.0);
// }

// inline double random_double(double min, double max) {
// 	// Returns a random real in [min,max).
// 	return min + (max - min) * random_double();
// }

// inline double clamp(double x, double min, double max) {
//     	if (x < min) return min;
// 	if (x > max) return max;
//     	return x;
// }


// //#include "ray.h"
// //#include "vec3.h"
