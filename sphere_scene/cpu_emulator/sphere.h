#ifndef SPHERE_BFP_H
#define SPHERE_BFP_H

#include "hittable_list.h"

using namespace floating_point;

class sphere : public hittable
{
public:
    sphere() {}
    sphere(int idx, point3 cen, fp r, shared_ptr<material> m) : sphereIdx(idx), center(cen), radius(r), mat_ptr(m){};
    sphere(int idx, point3 cen, __fpo __r, shared_ptr<material> m) : sphereIdx(idx), center(cen), radius(fpo_to_fp(__r)), mat_ptr(m){};

    virtual bool hit(const ray &r, fp t_min, fp t_max, hit_record &rec) const override;
    virtual bool hit(const ray &r, __fpo __t_min, __fpo __t_max, hit_record &rec) const override;
    virtual bool bounding_box(fp time0, fp time1, aabb &output_box) const override;
    virtual bool bounding_box(__fpo __time0, __fpo __time1, aabb &output_box) const override;

    // point3__fpo center_f() const { return vec3_fp_to_fpo(center); }
    __fpo radius_f() const { return fp_to_fpo(radius); }

public:
    point3 center;
    fp radius;
    shared_ptr<material> mat_ptr;
    int sphereIdx;
};

bool sphere::hit(const ray &r, fp t_min, fp t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    fp a = r.direction().length_squared();
    fp half_b = dot(oc, r.direction());
    fp c = oc.length_squared() - radius * radius;
    fp discriminant = half_b * half_b - a * c;
    fp sqrtd = sqrt(discriminant);
    fp root = (-half_b - sqrtd) / a;

    point3 pm = center - vec3(radius, radius, radius);
    point3 pM = center + vec3(radius, radius, radius);

    if (DEBUG)
    {
        cout << "    <SPHERE C=(" << center << "), r=" << fp_to_fpo(radius) << " HIT TEST> " << endl;
        cout << "    <SPHERE C=(" << center << "), r=" << fp_to_fpo(radius) << " HIT TEST> " << endl;
        cout << "      - oc: (" << oc << ")" << endl;
        cout << "      - a: " << fp_to_fpo(a) << endl;
        cout << "      - half_b: " << fp_to_fpo(half_b) << endl;
        cout << "      - c: " << fp_to_fpo(c) << endl;
        cout << "      - discriminant: " << fp_to_fpo(discriminant) << " vs " << fp_to_fpo(b_0_001) << endl;
        cout << "      - sqrtd: " << fp_to_fpo(sqrtd) << endl;
        cout << "      - root: " << fp_to_fpo(root) << endl;
    }

    if (discriminant < b_0) // if the ray doesn't hit the sphere
    {
        if (DEBUG)
        {
            cout << "No hit: D < 0" << endl;
        }
        return false;
    }
    if (root < t_min || t_max < root) // If the ray hits the sphere,
    {
        root = (-half_b + sqrtd) / a; // Find the nearest root that lies in the acceptable range.
        if (root < t_min || t_max < root)
        {
            if (DEBUG)
            {
                cout << "No hit: root t is out of range" << endl;
            }
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    if (DEBUG)
    {
        cout << "    <HIT RECORD> " << endl;
        cout << "      - Time: " << fp_to_fpo(rec.t) << endl;
        cout << "      - Hit P: " << rec.p << endl;
        cout << "      - Normal: (" << outward_normal << ")" << endl;
        cout << "      - Object I/O: ";
        if (rec.front_face)
            cout << "O" << endl;
        else
            cout << "I" << endl;
    }
    return true;
}

bool sphere::hit(const ray &r, __fpo __t_min, __fpo __t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    fp a = r.direction().length_squared();
    fp half_b = dot(oc, r.direction());
    fp c = oc.length_squared() - radius * radius;
    fp discriminant = half_b * half_b - a * c;
    fp sqrtd = sqrt(discriminant);
    fp root = (-half_b - sqrtd) / a;

    point3 pm = center - vec3(radius, radius, radius);
    point3 pM = center + vec3(radius, radius, radius);

    if (discriminant < b_0) // if the ray doesn't hit the sphere
        return false;
    if (root < __t_min || __t_max < root) // If the ray hits the sphere,
    {
        root = (-half_b + sqrtd) / a; // Find the nearest root that lies in the acceptable range.
        if (root < __t_min || __t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

bool sphere::bounding_box(fp time0, fp time1, aabb &output_box) const
{
    // if(DEBUG){
    //     cout << "-----------sphere::bounding_box------------" << endl;
    //     cout << "center:" << endl;
    //     print_vec3(center);
    //     cout << "radius: " << fp_to_fpo(radius) << endl;
    // }

    output_box = aabb(
        center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));

    return true;
}

bool sphere::bounding_box(__fpo __time0, __fpo __time1, aabb &output_box) const
{
    output_box = aabb(
        center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));

    return true;
}

#endif
