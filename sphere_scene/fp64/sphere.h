/*
 * ===================================================
 *
 *       Filename:  sphere.h
 *    Description:  Ray Tracing In One Weekend (RTIOW): Final Code
 *        Created:  2022/07/13
 *
 * ===================================================
 */


// Preprocessors

#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include <iostream>

#define DEBUG 0

using namespace std;

// Classes

class sphere : public hittable {
public:
    	sphere() {}
        sphere(point3 cen, double r, shared_ptr<material> m)
            : center(cen), radius(r), mat_ptr(m) {};
    
    	virtual bool hit(
			const ray& r, double t_min, double t_max, hit_record& rec) const override;

public:
    	point3 center;
    	double radius;
        shared_ptr<material> mat_ptr;
};


bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;
	auto sqrtd = sqrt(discriminant);

	if(DEBUG)
    {
        double x = (center.x());
        double y = (center.y());
        double z = (center.z());

        cout << "    <SPHERE C=(" << x << "," << y << "," << z << "), r=" << (radius) << " HIT TEST> " << endl;
        cout << "      - oc: (" << oc << ")" << endl;
        cout << "      - a: " << (a) << endl;
        cout << "      - half_b: " << (half_b) << endl;
        cout << "      - c: " << (c) << endl;
        cout << "      - discriminant: " << (discriminant) << endl;
		cout << "      - sqrtd: " << (sqrtd) << endl;

    }

	if (discriminant < 0) return false;


	// Find the nearest root that lies in the acceptable range.
	auto root = (-half_b - sqrtd) / a;
	if (root < t_min || t_max < root) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || t_max < root)
			return false;
	}
    
	rec.t = root;
	rec.p = r.at(rec.t);
	vec3 outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;

    if(DEBUG)
    {
        cout << "    <HIT RECORD> " << endl;
        cout << "      - Time: " << rec.t << endl;
        cout << "      - Normal: (" << outward_normal << ")" << endl;
        cout << "      - Object I/O: ";
        if(rec.front_face)  cout << "O" << endl;
        else    cout << "I" << endl;
    }

    	return true;
}


#endif
