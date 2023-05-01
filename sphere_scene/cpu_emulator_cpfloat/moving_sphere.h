#ifndef MOVING_SPHERE_BFP_H
#define MOVING_SPHERE_BFP_H

#include "sphere.h"

using namespace custom_precision_fp;

class moving_sphere : public hittable
{
public:
    moving_sphere() {}
    moving_sphere(
        int idx, point3 cen0, point3 cen1, fp_custom time0, fp_custom time1, fp_custom r, shared_ptr<material> m)
        : sphereIdx(idx), center0(cen0), center1(cen1), time0(time0), time1(time1), radius(r), mat_ptr(m){};
    moving_sphere(
        int idx, point3 cen0, point3 cen1, fp_orig __time0, fp_orig __time1, fp_custom r, shared_ptr<material> m)
        : sphereIdx(idx), center0(cen0), center1(cen1), time0(fp_orig_to_custom(__time0)), time1(fp_orig_to_custom(__time1)), radius(r), mat_ptr(m){};

    virtual bool hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const override;
    virtual bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const override;
    virtual bool bounding_box(fp_custom time0, fp_custom time1, aabb &output_box) const override;
    virtual bool bounding_box(fp_orig __time0, fp_orig __time1, aabb &output_box) const override;

    point3 center(fp_custom time) const;
    point3 center(fp_orig __time) const;

public:
    point3 center0, center1;
    fp_custom time0, time1;
    fp_custom radius;
    shared_ptr<material> mat_ptr;
    int sphereIdx;
};

point3 moving_sphere::center(fp_custom time) const
{
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

point3 moving_sphere::center(fp_orig __time) const
{
    fp_custom time = fp_orig_to_custom(__time);
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

bool moving_sphere::hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center(r.time());
    fp_custom a = r.direction().length_squared();
    fp_custom half_b = dot(oc, r.direction());
    fp_custom c = oc.length_squared() - radius * radius;

    fp_custom discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;
    fp_custom sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    fp_custom root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root)
    {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

bool moving_sphere::hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center(r.time());
    fp_custom a = r.direction().length_squared();
    fp_custom half_b = dot(oc, r.direction());
    fp_custom c = oc.length_squared() - radius * radius;

    fp_custom discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;
    fp_custom sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    fp_custom root = (-half_b - sqrtd) / a;
    if (root < __t_min || __t_max < root)
    {
        root = (-half_b + sqrtd) / a;
        if (root < __t_min || __t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

bool moving_sphere::bounding_box(fp_custom _time0, fp_custom _time1, aabb &output_box) const
{
    aabb box0(
        center(_time0) - vec3(radius, radius, radius),
        center(_time0) + vec3(radius, radius, radius));
    aabb box1(
        center(_time1) - vec3(radius, radius, radius),
        center(_time1) + vec3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}

bool moving_sphere::bounding_box(fp_orig __time0, fp_orig __time1, aabb &output_box) const
{
    aabb box0(
        center(__time0) - vec3(radius, radius, radius),
        center(__time0) + vec3(radius, radius, radius));
    aabb box1(
        center(__time1) - vec3(radius, radius, radius),
        center(__time1) + vec3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}

#endif
