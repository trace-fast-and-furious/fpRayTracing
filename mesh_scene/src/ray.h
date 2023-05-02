/*
 * ===================================================
 *
 *       Filename:  ray.h
 *    Description:  Ray Tracing In One Weekend (RTIOW): Final Code 
 *        Created:  2022/07/13
 * 
 * ===================================================
 */


// Preprocessors
#pragma once
#ifndef RAY_H
#define RAY_H

#include "vec3.h"

using namespace custom_precision_fp;

// Classes
class Ray 
{
public:
    Ray() {}
	Ray(const Point3 &origin, const Point3 &direction, fp_custom time) : orig(origin), dir(direction), tm(time) {}
	Ray(const Point3 &origin, const Point3 &direction, fp_orig time) : orig(origin), dir(direction), tm(fp_orig_to_custom(time)) {}

	Point3 origin() const { return orig; }
	Vec3 direction() const { return dir; }
	fp_custom time() const { return tm; }
	Point3 at(double t) const {
		return orig + t * dir;
	}

public:
	Point3 orig;
	Vec3 dir;
	fp_custom tm;
};

#endif
