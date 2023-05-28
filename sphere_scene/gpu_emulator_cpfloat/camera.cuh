#ifndef CAMERAH
#define CAMERAH

#include "ray.cuh"
#include <curand_kernel.h>

using namespace custom_precision_fp;

__device__ vec3 random_in_unit_disk(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera
{
public:
    __device__ camera(point3 lookfrom,
                      point3 lookat,
                      vec3 vup,
                      fp_orig __vfov, // vertical field-of-view in degrees
                      fp_orig __aspect_ratio,
                      fp_orig __aperture,
                      fp_orig __focus_dist)
    { // vfov is top to bottom in degrees
        fp_custom vfov = fp_orig_to_custom(__vfov);
        fp_custom aspect_ratio = fp_orig_to_custom(__aspect_ratio);
        fp_custom aperture = fp_orig_to_custom(__aperture);
        fp_custom focus_dist = fp_orig_to_custom(__focus_dist);

        fp_custom theta = vfov * fp_orig_to_custom(M_PI) / 180;
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
    }

    __device__ ray get_ray(fp_orig __s, fp_orig __t, curandState *local_rand_state)
    {
        fp_custom s = fp_orig_to_custom(__s);
        fp_custom t = fp_orig_to_custom(__t);
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    fp_custom lens_radius;
};

#endif
