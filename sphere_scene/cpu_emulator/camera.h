#ifndef CAMERA_BFP_H
#define CAMERA_BFP_H

#include "ray.h"

using namespace floating_point;

class camera
{
public:
    camera(
        point3 lookfrom,
        point3 lookat,
        vec3 vup,
        fp vfov, // vertical field-of-view in degrees
        fp aspect_ratio,
        fp aperture,
        fp focus_dist,
        fp _time0 = b_0,
        fp _time1 = b_0

    )
    {
        fp theta = degrees_to_radians(vfov);
        fp h = tan(theta / b_2);
        fp viewport_height = b_2 * h;
        fp viewport_width = aspect_ratio * viewport_height;

        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / b_2 - vertical / b_2 - focus_dist * w;
        lens_radius = aperture / b_2;
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
            cout << "lens_radius: " << fp_to_fpo(lens_radius) << endl;
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
        __fpo __vfov, // vertical field-of-view in degrees
        __fpo __aspect_ratio,
        __fpo __aperture,
        __fpo __focus_dist,
        __fpo __time0 = 0.0f,
        __fpo __time1 = 0.0f

    )
    {
        fp vfov = fpo_to_fp(__vfov);
        fp aspect_ratio = fpo_to_fp(__aspect_ratio);
        fp aperture = fpo_to_fp(__aperture);
        fp focus_dist = fpo_to_fp(__focus_dist);
        fp _time0 = fpo_to_fp(__time0);
        fp _time1 = fpo_to_fp(__time1);

        fp theta = degrees_to_radians(vfov);
        fp h = tan(theta / b_2);
        fp viewport_height = b_2 * h;
        fp viewport_width = aspect_ratio * viewport_height;

        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / b_2 - vertical / b_2 - focus_dist * w;
        lens_radius = aperture / b_2;
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
            cout << "lens_radius: " << fp_to_fpo(lens_radius) << endl;
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

    ray get_ray(fp s, fp t) const
    {

        vec3 rd = lens_radius * random_in_unit_disk(true);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            random_num(time0, time1));
    }

    ray get_ray(__fpo __s, __fpo __t) const
    {
        fp s = fpo_to_fp(__s);
        fp t = fpo_to_fp(__t);

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
    fp lens_radius;
    fp time0, time1; // shutter open/close times
};

#endif