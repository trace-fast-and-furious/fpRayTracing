#ifndef MATERIAL_BFP_H
#define MATERIAL_BFP_H

#include "hittable_list.cuh"

struct hit_record;

using namespace custom_precision_fp;

__device__ __host__ fp_custom schlick(fp_orig cosine_f, fp_orig ref_idx_f)
{
    fp_custom cosine = fp_orig_to_custom(cosine_f);
    fp_custom ref_idx = fp_orig_to_custom(ref_idx_f);
    fp_custom r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ __host__ fp_custom schlick(fp_custom cosine, fp_custom ref_idx)
{
    fp_custom r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ __host__ bool refract(const vec3 &v, const vec3 &n, fp_orig ni_over_nt_f, vec3 &refracted)
{
    fp_custom ni_over_nt = fp_orig_to_custom(ni_over_nt_f);
    vec3 uv = unit_vector(v);
    fp_custom dt = dot(uv, n);
    fp_custom discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ __host__ bool refract(const vec3 &v, const vec3 &n, fp_custom ni_over_nt, vec3 &refracted)
{
    vec3 uv = unit_vector(v);
    fp_custom dt = dot(uv, n);
    fp_custom discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ __host__ vec3 random_in_unit_sphere(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ __host__ vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

class material
{
public:
    __device__ __host__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material
{
public:
    __device__ __host__ lambertian(const vec3 &a) : albedo(a) {}
    __device__ __host__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *local_rand_state) const override
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

public:
    vec3 albedo;
};

class metal : public material
{
public:
    __device__ __host__ metal(const vec3 &a, fp_orig __f) : albedo(a), fuzz(__f < 1.0f ? fp_orig_to_custom(__f) : fp_orig_to_custom(1)) {}

    __device__ __host__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

public:
    vec3 albedo;
    fp_custom fuzz;
};

class dielectric : public material
{
public:
    __device__ __host__ dielectric(fp_orig ri) : ref_idx(fp_orig_to_custom(ri)) {}

    __device__ __host__ virtual bool scatter(const ray &r_in,
                         const hit_record &rec,
                         vec3 &attenuation,
                         ray &scattered,
                         curandState *local_rand_state) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        vec3 refracted;

        fp_custom ni_over_nt;
        fp_custom reflect_prob;
        fp_custom cosine;
        attenuation = vec3(1.0, 1.0, 1.0);

        if (dot(r_in.direction(), rec.normal) > 0.0f)
        {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = fp_orig_to_custom(1);
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

public:
    fp_custom ref_idx;
};

#endif