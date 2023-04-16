#ifndef MATERIAL_BFP_H
#define MATERIAL_BFP_H

#include "moving_sphere.h"

using namespace custom_precision_fp;

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
    metal(const color &a, fp_custom f) : albedo(a), fuzz(f < 1.0f ? f : fp_orig_to_custom(1)) {}
    metal(const color &a, fp_orig __f) : albedo(a), fuzz(__f < 1.0f ? fp_orig_to_custom(__f) : fp_orig_to_custom(1)) {}

    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(true), r_in.time());
        attenuation = albedo;

        fp_custom ret = dot(scattered.direction(), rec.normal);

        if (DEBUG)
        {
            cout << "<METAL>" << endl;
            cout << "    * In D = (" << unit_vector(r_in.direction()) << ")" << endl;
            cout << "    * Normal D = (" << rec.normal << ")" << endl;
            cout << "    * Reflected D = (" << reflected << ")" << endl;
            cout << "    * Reflect: " << val(ret) << " > 0";
            if (ret > 0)
                cout << " reflect O" << endl;
            else
                cout << "reflect X" << endl;
        }

        return (ret > 0);
    }

public:
    color albedo;
    fp_custom fuzz;
};

class dielectric : public material
{
public:
    dielectric(fp_custom index_of_refraction) : ir(index_of_refraction) {}
    dielectric(fp_orig __index_of_refraction) : ir(fp_orig_to_custom(__index_of_refraction)) {}

    virtual bool scatter(
        const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override
    {
        attenuation = color(1, 1, 1);
        fp_custom refraction_ratio = rec.front_face ? (1 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());

        fp_custom cos_theta = min(dot(-unit_direction, rec.normal), 1);
        fp_custom sin_theta = sqrt(1 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1;
        vec3 direction;

        fp_custom rand_f = random_num();
        fp_custom ref = reflectance(cos_theta, refraction_ratio);

        if (cannot_refract || ref > rand_f)
        {
            direction = reflect(unit_direction, rec.normal);
        }
        else
        {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = ray(rec.p, direction, r_in.time());

        if (DEBUG)
        {
            cout << "  - reflectance: " << val(ref) << " > " << val(rand_f) << endl;
            cout << "  - ir: " << val(ir) << endl;
            cout << "  - normal: (" << rec.normal << ")" << endl;
            cout << "  - refraction_ratio: " << val(refraction_ratio) << endl;
            cout << "  - unit_direction: (" << unit_direction << ")" << endl;
            cout << "  - cos_theta: " << val(cos_theta) << endl;
            cout << "  - sin_theta: " << val(sin_theta) << endl;
            cout << "  - cannot_refract: ";
            if (cannot_refract)
                cout << "T" << endl;
            else
                cout << "F" << endl;
            // vec3_float ray_orig = fp_orig_to_custom(rec.p);
            cout << "  - ray_origin: (" << rec.p << ")" << endl;
            cout << "  - ray_direction: (" << direction << ")" << endl;
        }

        return true;
    }

public:
    fp_custom ir;

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
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

#endif