#ifndef SPHERE_BFP_H
#define SPHERE_BFP_H

#include "hittable_list.h"

using namespace custom_precision_fp;

class sphere : public hittable
{
public:
    sphere() {}
    sphere(int idx, point3 cen, fp_custom r, shared_ptr<material> m) : sphereIdx(idx), center(cen), radius(r), mat_ptr(m){};
    sphere(int idx, point3 cen, fp_orig __r, shared_ptr<material> m) : sphereIdx(idx), center(cen), radius(fp_orig_to_custom(__r)), mat_ptr(m){};

    virtual bool hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const override;
    virtual bool hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const override;
    virtual bool bounding_box(fp_custom time0, fp_custom time1, aabb &output_box) const override;
    virtual bool bounding_box(fp_orig __time0, fp_orig __time1, aabb &output_box) const override;

    // point3fp_orig center_f() const { return vec3_val(center); }
    fp_orig radius_f() const { return val(radius); }

public:
    point3 center;
    fp_custom radius;
    shared_ptr<material> mat_ptr;
    int sphereIdx;
};

bool sphere::hit(const ray &r, fp_custom t_min, fp_custom t_max, hit_record &rec) const
{
    /* before: fp_custom */
    vec3 oc = r.origin() - center;
    fp_custom a = r.direction().length_squared();
    fp_custom half_b = dot(oc, r.direction());
    fp_custom c = oc.length_squared() - radius * radius;
    fp_custom discriminant = half_b * half_b - a * c;
    fp_custom sqrtd = sqrt(discriminant);
    fp_custom root = (-half_b - sqrtd) / a;

    point3 pm = center - vec3(radius, radius, radius);
    point3 pM = center + vec3(radius, radius, radius);

    if (DEBUG)
    {
        cout << "    <SPHERE C=(" << center << "), r=" << val(radius) << " HIT TEST> " << endl;
        cout << "    <SPHERE C=(" << center << "), r=" << val(radius) << " HIT TEST> " << endl;
        cout << "      - oc: (" << oc << ")" << endl;
        cout << "      - a: " << val(a) << endl;
        cout << "      - half_b: " << val(half_b) << endl;
        cout << "      - c: " << val(c) << endl;
        cout << "      - discriminant: " << val(discriminant) << " vs " << 0.001 << endl;
        cout << "      - sqrtd: " << val(sqrtd) << endl;
        cout << "      - root: " << val(root) << endl;
    }

    if (discriminant < 0) // if the ray doesn't hit the sphere
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
        cout << "      - Time: " << val(rec.t) << endl;
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

bool sphere::hit(const ray &r, fp_orig __t_min, fp_orig __t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    fp_custom a = r.direction().length_squared();
    fp_custom half_b = dot(oc, r.direction());
    fp_custom c = oc.length_squared() - radius * radius;
    fp_custom discriminant = half_b * half_b - a * c;
    fp_custom sqrtd = sqrt(discriminant);
    fp_custom root = (-half_b - sqrtd) / a;

    point3 pm = center - vec3(radius, radius, radius);
    point3 pM = center + vec3(radius, radius, radius);

    if (discriminant < 0) // if the ray doesn't hit the sphere
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

bool sphere::bounding_box(fp_custom time0, fp_custom time1, aabb &output_box) const
{
    // if(DEBUG){
    //     cout << "-----------sphere::bounding_box------------" << endl;
    //     cout << "center:" << endl;
    //     print_vec3(center);
    //     cout << "radius: " << val(radius) << endl;
    // }

    output_box = aabb(
        center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));

    return true;
}

bool sphere::bounding_box(fp_orig __time0, fp_orig __time1, aabb &output_box) const
{
    output_box = aabb(
        center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));

    return true;
}

#endif
