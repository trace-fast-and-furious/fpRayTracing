/*
 * ===================================================
 *
 *       Filename:  hittable.h
 *    Description:  Ray Tracing In One Weekend (RTIOW): Final Code 
 *        Created:  2022/07/13
 * 
 * ===================================================
 */


// Preprocessors
#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "camera.h"

using namespace custom_precision_fp;

class Material;

// Classes
struct HitRecord 
{
	Point3 p;
	Vec3 normal;
	shared_ptr<Material> mat_ptr;
	fp_custom t;
	bool is_front_face;

	inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
		is_front_face = dot(r.direction(), outward_normal) < 0;
		normal = is_front_face ? outward_normal : -outward_normal;
	}
};


class Hittable 
{
	public:
		virtual bool hit(const Ray &r, fp_orig __t_min, fp_orig __t_max, HitRecord &rec) const = 0;
		virtual bool hit(const Ray &r, fp_custom t_min, fp_custom t_max, HitRecord &rec) const = 0;
};

#endif