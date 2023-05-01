#ifndef MATERIAL_BFP_H
#define MATERIAL_BFP_H

#include "moving_sphere.h"

using namespace floating_point;

class material
{
public:
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const = 0;
};

class lambertian : public material
{
public:
    lambertian(const color &a) : albedo(a) {}

    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        vec3 rand_uv = random_unit_vector(true);
        vec3 scatter_direction = rec.normal + rand_uv;
        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        if (DEBUG)
        {
            cout << "<LAMBERTIAN>" << endl;
            cout << "    * Random uV = (" << rand_uv << ")" << endl;
            cout << "    * Scatter D = (" << scatter_direction << ")" << endl;
        }

        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = albedo;
        return true;
    }

public:
    color albedo;
};

class metal : public material
{
public:
    metal(const color &a, fp f) : albedo(a), fuzz(f < 1.0f ? f : b_1) {}
    metal(const color &a, __fpo __f) : albedo(a), fuzz(__f < 1.0f ? fpo_to_fp(__f) : b_1) {}

    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(true), r_in.time());
        attenuation = albedo;

        fp ret = dot(scattered.direction(), rec.normal);

        if (DEBUG)
        {
            cout << "<METAL>" << endl;
            cout << "    * In D = (" << unit_vector(r_in.direction()) << ")" << endl;
            cout << "    * Normal D = (" << rec.normal << ")" << endl;
            cout << "    * Reflected D = (" << reflected << ")" << endl;
            cout << "    * Reflect: " << fp_to_fpo(ret) << " > 0";
            if (ret > b_0)
                cout << " reflect O" << endl;
            else
                cout << "reflect X" << endl;
        }

        return (ret > b_0);
    }

public:
    color albedo;
    fp fuzz;
};

class dielectric : public material
{
public:
    dielectric(fp index_of_refraction) : ir(index_of_refraction) {}
    dielectric(__fpo __index_of_refraction) : ir(fpo_to_fp(__index_of_refraction)) {}

    virtual bool scatter(
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        attenuation = color(b_1, b_1, b_1);
        fp refraction_ratio = rec.front_face ? (b_1 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());

        fp cos_theta = min(dot(-unit_direction, rec.normal), b_1);
        fp sin_theta = sqrt(b_1 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > b_1;
        vec3 direction;

        if (DEBUG)
        {
            cout << "<DIELECTRIC>" << endl;
        }

        fp rand_f = random_num();
        fp ref = reflectance(cos_theta, refraction_ratio);

        if (cannot_refract || ref > rand_f)
        {
            if (DEBUG)
            {
                cout << "  *REFLECT*" << endl;
            }
            direction = reflect(unit_direction, rec.normal);
        }
        else
        {
            if (DEBUG)
            {
                cout << "  *REFRACT*" << endl;
            }
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = ray(rec.p, direction, r_in.time());

        if (DEBUG)
        {
            cout << "  - reflectance: " << fp_to_fpo(ref) << " > " << fp_to_fpo(rand_f) << endl;
            cout << "  - ir: " << fp_to_fpo(ir) << endl;
            cout << "  - normal: (" << rec.normal << ")" << endl;
            cout << "  - refraction_ratio: " << fp_to_fpo(refraction_ratio) << endl;
            cout << "  - unit_direction: (" << unit_direction << ")" << endl;
            cout << "  - cos_theta: " << fp_to_fpo(cos_theta) << endl;
            cout << "  - sin_theta: " << fp_to_fpo(sin_theta) << endl;
            cout << "  - cannot_refract: ";
            if (cannot_refract)
                cout << "T" << endl;
            else
                cout << "F" << endl;
            // vec3_float ray_orig = fp_to_fpo(rec.p);
            cout << "  - ray_origin: (" << rec.p << ")" << endl;
            cout << "  - ray_direction: (" << direction << ")" << endl;
        }

        return true;
    }

public:
    fp ir;

private:
    static fp reflectance(fp cosine, fp ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        fp r0 = (b_1 - ref_idx) / (b_1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (b_1 - r0) * pow((b_1 - cosine), 5);
    }

    static __fpo reflectance(__fpo cosine, __fpo ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif