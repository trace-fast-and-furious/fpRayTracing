#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace bfloat16;

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
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov_f, float aspect_f, float aperture_f, float focus_dist_f)
    { // vfov is top to bottom in degrees
        __nv_bfloat16 vfov = __float2bfloat16(vfov_f);
        __nv_bfloat16 aspect = __float2bfloat16(aspect_f);
        __nv_bfloat16 aperture = __float2bfloat16(aperture_f);
        __nv_bfloat16 focus_dist = __float2bfloat16(focus_dist_f);

        lens_radius = aperture / 2.0f;
        __nv_bfloat16 theta = vfov * ((float)M_PI) / 180.0f;
        __nv_bfloat16 half_height = tan(theta / 2.0f);
        __nv_bfloat16 half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        // lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;

        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;

        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    __device__ ray get_ray(float s_f, float t_f, curandState *local_rand_state)
    {
        __nv_bfloat16 s = __float2bfloat16(s);
        __nv_bfloat16 t = __float2bfloat16(t);
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    __nv_bfloat16 lens_radius;
};

#endif
