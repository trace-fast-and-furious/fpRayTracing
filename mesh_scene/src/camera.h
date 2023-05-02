/*
 * ===================================================
 *
 *       Filename:  camera.h
 *    Description:  Ray Tracing In One Weekend (RTIOW): Final Code
 *        Created:  2022/07/13
 *
 * ===================================================
 */

// Preprocessors
#pragma once
#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

using namespace custom_precision_fp;

// Classes

class Camera
{
public:
	Camera(
		Point3 lookfrom,
		Point3 lookat,
		Vec3 vup,
		fp_orig __vfov, // vertical field-of-view in degrees
		fp_orig __aspect_ratio,
		fp_orig __aperture,
		fp_orig __focus_dist,
		fp_orig __time0 = 0,
		fp_orig __time1 = 0

	)
	{
		fp_custom vfov = fp_orig_to_custom(__vfov);
		fp_custom aspect_ratio = fp_orig_to_custom(__aspect_ratio);
		fp_custom aperture = fp_orig_to_custom(__aperture);
		fp_custom focus_dist = fp_orig_to_custom(__focus_dist);
		fp_custom _time0 = fp_orig_to_custom(__time0);
		fp_custom _time1 = fp_orig_to_custom(__time1);

		fp_custom theta = degrees_to_radians(vfov);
		fp_custom h = tan(theta / 2);
		fp_custom viewport_height = 2 * h;
		fp_custom viewport_width = aspect_ratio * viewport_height;

		Vec3 w = unit_vector(lookfrom - lookat);
		Vec3 u = unit_vector(cross(vup, w));
		Vec3 v = cross(w, u);

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;
		lens_radius = aperture / 2;
		time0 = _time0;
		time1 = _time1;
	}

	Camera(
		Point3 lookfrom,
		Point3 lookat,
		Vec3 vup,
		fp_custom vfov, // vertical field-of-view in degrees
		fp_custom aspect_ratio,
		fp_custom aperture,
		fp_custom focus_dist,
		fp_custom _time0 = {{0}},
		fp_custom _time1 = {{0}})
	{
		fp_custom theta = degrees_to_radians(vfov);
		fp_custom h = tan(theta / 2);
		fp_custom viewport_height = 2 * h;
		fp_custom viewport_width = aspect_ratio * viewport_height;

		Vec3 w = unit_vector(lookfrom - lookat);
		Vec3 u = unit_vector(cross(vup, w));
		Vec3 v = cross(w, u);

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
		lens_radius = aperture / 2;
		time0 = _time0;
		time1 = _time1;
	}

	Ray get_ray(fp_custom s, fp_custom t) const
	{

		Vec3 rd = lens_radius * random_in_unit_disk(true);
		Vec3 offset = u * rd.x() + v * rd.y();

		return Ray(
			origin + offset,
			lower_left_corner + s * horizontal + t * vertical - origin - offset,
			random_num(time0, time1));
	}

	Ray get_ray(fp_orig __s, fp_orig __t) const
	{
		fp_custom s = fp_orig_to_custom(__s);
		fp_custom t = fp_orig_to_custom(__t);

		Vec3 rd = lens_radius * random_in_unit_disk(true);
		Vec3 offset = u * rd.x() + v * rd.y();

		return Ray(
			origin + offset,
			lower_left_corner + s * horizontal + t * vertical - origin - offset,
			random_num(time0, time1));
	}

private:
	Point3 origin;
	Point3 lower_left_corner;
	Vec3 horizontal;
	Vec3 vertical;
	Vec3 u, v, w;
	fp_custom lens_radius;
	fp_custom time0, time1; // shutter open/close times
};

#endif
