#ifndef MOVING_SPHERE_BFP_H
#define MOVING_SPHERE_BFP_H

#include "sphere.h"

using namespace floating_point;

class moving_sphere : public hittable
{
public:
    moving_sphere() {}
    moving_sphere(
        int idx, point3 cen0, point3 cen1, fp time0, fp time1, fp r, shared_ptr<material> m)
        : sphereIdx(idx), center0(cen0), center1(cen1), time0(time0), time1(time1), radius(r), mat_ptr(m){};
    moving_sphere(
        int idx, point3 cen0, point3 cen1, __fpo __time0, __fpo __time1, fp r, shared_ptr<material> m)
        : sphereIdx(idx), center0(cen0), center1(cen1), time0(fpo_to_fp(__time0)), time1(fpo_to_fp(__time1)), radius(r), mat_ptr(m){};

    virtual bool hit(const ray &r, fp t_min, fp t_max, hit_record &rec) const override;
    virtual bool hit(const ray &r, __fpo __t_min, __fpo __t_max, hit_record &rec) const override;
    virtual bool bounding_box(fp time0, fp time1, aabb &output_box) const override;
    virtual bool bounding_box(__fpo __time0, __fpo __time1, aabb &output_box) const override;

    point3 center(fp time) const;
    point3 center(__fpo __time) const;

public:
    point3 center0, center1;
    fp time0, time1;
    fp radius;
    shared_ptr<material> mat_ptr;
    int sphereIdx;
};

point3 moving_sphere::center(fp time) const
{
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

point3 moving_sphere::center(__fpo __time) const
{
    fp time = fpo_to_fp(__time);
    return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

bool moving_sphere::hit(const ray &r, fp t_min, fp t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center(r.time());
    fp a = r.direction().length_squared();
    fp half_b = dot(oc, r.direction());
    fp c = oc.length_squared() - radius * radius;

    fp discriminant = half_b * half_b - a * c;
    if (discriminant < b_0)
        return false;
    fp sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    fp root = (-half_b - sqrtd) / a;
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

bool moving_sphere::hit(const ray &r, __fpo __t_min, __fpo __t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center(r.time());
    fp a = r.direction().length_squared();
    fp half_b = dot(oc, r.direction());
    fp c = oc.length_squared() - radius * radius;

    fp discriminant = half_b * half_b - a * c;
    if (discriminant < b_0)
        return false;
    fp sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    fp root = (-half_b - sqrtd) / a;
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

bool moving_sphere::bounding_box(fp _time0, fp _time1, aabb &output_box) const
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

bool moving_sphere::bounding_box(__fpo __time0, __fpo __time1, aabb &output_box) const
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
