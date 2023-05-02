/*
 * ===================================================
 *
 *       Filename:  material.h
 *    Description:  Ray Tracing In One Weekend (RTIOW): Final Code
 *        Created:  2022/07/13
 *
 * ===================================================
 */

// Preprocessors
#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

using namespace custom_precision_fp;

struct HitRecord;

class Material
{
public:
	virtual bool scatter(
		const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const = 0;
};

class Lambertian : public Material
{
public:
	Lambertian(const Color &a) : albedo(a) {}

	virtual bool scatter(
		const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const override
	{
		Vec3 scatter_direction = rec.normal + random_unit_vector(true);

		// Catch degenerate scatter direction
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;
		scattered = Ray(rec.p, scatter_direction, r_in.time());
		attenuation = albedo;

		return true;
	}

public:
	Color albedo;
};

class Metal : public Material
{
public:
	Metal(const Color &a, fp_custom f) : albedo(a), fuzz(f < 1.0f ? f : fp_orig_to_custom(1)) {}
	Metal(const Color &a, fp_orig __f) : albedo(a), fuzz(__f < 1.0f ? fp_orig_to_custom(__f) : fp_orig_to_custom(1)) {}

	virtual bool scatter(
		const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const override
	{
		Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(true), r_in.time());
		attenuation = albedo;

		return (dot(scattered.direction(), rec.normal) > 0);
	}

public:
	Color albedo;
	fp_custom fuzz;
};

class Dielectric : public Material
{
public:
	Dielectric(fp_custom index_of_refraction) : ir(index_of_refraction) {}
	Dielectric(fp_orig __index_of_refraction) : ir(fp_orig_to_custom(__index_of_refraction)) {}

	virtual bool scatter(
		const Ray &r_in, const HitRecord &rec, Color &attenuation, Ray &scattered) const override
	{
		attenuation = Color(1, 1, 1);
		fp_custom refraction_ratio = rec.is_front_face ? (1 / ir) : ir;
		Vec3 unit_direction = unit_vector(r_in.direction());
		fp_custom cos_theta = min(dot(-unit_direction, rec.normal), 1);
		fp_custom sin_theta = sqrt(1 - cos_theta * cos_theta);
		bool cannot_refract = refraction_ratio * sin_theta > 1;
		Vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_num())
		{
			direction = reflect(unit_direction, rec.normal);
		}
		else
		{
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}
		scattered = Ray(rec.p, direction, r_in.time());

		return true;
	}

public:
	fp_custom ir; // Index of Refraction

private:
	static fp_custom reflectance(fp_custom cosine, fp_custom ref_idx)
	{
		// Use Schlick's approximation for reflectance.
		fp_custom r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}

	static fp_orig reflectance(fp_orig cosine, fp_orig ref_idx)
	{
		// Use Schlick's approximation for reflectance.
		fp_orig r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};

#endif