#ifndef CAMERA_BFP_H
#define CAMERA_BFP_H

#include "ray.h"

using namespace custom_precision_fp;

class camera
{
public:
    camera(
        point3 lookfrom,
        point3 lookat,
        vec3 vup,
        fp_custom vfov, // vertical field-of-view in degrees
        fp_custom aspect_ratio,
        fp_custom aperture,
        fp_custom focus_dist,
        fp_custom _time0 = {{0}},
        fp_custom _time1 = {{0}}
    )
    {
        fp_custom theta = degrees_to_radians(vfov);
        fp_custom h = tan(theta / 2);
        fp_custom viewport_height = 2 * h;
        fp_custom viewport_width = aspect_ratio * viewport_height;

        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
        lens_radius = aperture / 2;
        time0 = _time0;
        time1 = _time1;

        if (DEBUG)
        {
            cout << "--------------------- Create camera -----------------" << endl;
            cout << "origin: " << endl;
            print_vec3(origin);
            cout << "horizontal:  " << endl;
            print_vec3(horizontal);
            cout << "vertical:  " << endl;
            print_vec3(vertical);
            cout << "lower_left_corner: " << endl;
            print_vec3(lower_left_corner);
            cout << "lens_radius: " << val(lens_radius) << endl;
        }
        if (DEBUG)
        {
            cout << "--------------------- vup, w, u, v -----------------" << endl;
            cout << "vup: " << endl;
            print_vec3(vup);
            cout << "w: " << endl;
            print_vec3(w);
            cout << "u:  " << endl;
            print_vec3(u);
            cout << "v:  " << endl;
            print_vec3(v);
            cout << "lookfrom:  " << endl;
            print_vec3(lookfrom);
            cout << "lookat:  " << endl;
            print_vec3(lookat);
            cout << "lookfrom - lookat:  " << endl;
            print_vec3(lookfrom - lookat);
        }
    }

    camera(
        point3 lookfrom,
        point3 lookat,
        vec3 vup,
        fp_orig __vfov, // vertical field-of-view in degrees
        fp_orig __aspect_ratio,
        fp_orig __aperture,
        fp_orig __focus_dist,
        fp_orig __time0 = 0.0f,
        fp_orig __time1 = 0.0f

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

        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;
        lens_radius = aperture / 2;
        time0 = _time0;
        time1 = _time1;

        if (DEBUG)
        {
            cout << "--------------------- Create camera -----------------" << endl;
            cout << "origin: " << endl;
            print_vec3(origin);
            cout << "horizontal:  " << endl;
            print_vec3(horizontal);
            cout << "vertical:  " << endl;
            print_vec3(vertical);
            cout << "lower_left_corner: " << endl;
            print_vec3(lower_left_corner);
            cout << "lens_radius: " << val(lens_radius) << endl;
        }
        if (DEBUG)
        {
            cout << "--------------------- w, u, v -----------------" << endl;
            cout << "w: " << endl;
            print_vec3(w);
            cout << "u:  " << endl;
            print_vec3(u);
            cout << "v:  " << endl;
            print_vec3(v);
            cout << "lookfrom:  " << endl;
            print_vec3(lookfrom);
            cout << "lookat:  " << endl;
            print_vec3(lookat);
        }
    }

    ray get_ray(fp_custom s, fp_custom t) const
    {

        vec3 rd = lens_radius * random_in_unit_disk(true);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            random_num(time0, time1));
    }

    ray get_ray(fp_orig __s, fp_orig __t) const
    {
        fp_custom s = fp_orig_to_custom(__s);
        fp_custom t = fp_orig_to_custom(__t);

        vec3 rd = lens_radius * random_in_unit_disk(true);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            random_num(time0, time1));
    }

public:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    fp_custom lens_radius;
    fp_custom time0, time1; // shutter open/close times
};

#endif