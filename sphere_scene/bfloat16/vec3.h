#pragma once
#ifndef VEC3H
#define VEC3H

#include "bfloat16.h"

using namespace bfloat16;

class vec3
{
public:
    __device__ vec3() {}
    __device__ vec3(__nv_bfloat16 e0, __nv_bfloat16 e1, __nv_bfloat16 e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __device__ vec3(float e0, float e1, float e2)
    {
        e[0] = __float2bfloat16(e0);
        e[1] = __float2bfloat16(e1);
        e[2] = __float2bfloat16(e2);
    }
    __device__ inline __nv_bfloat16 x() const { return e[0]; }
    __device__ inline __nv_bfloat16 y() const { return e[1]; }
    __device__ inline __nv_bfloat16 z() const { return e[2]; }
    __device__ inline __nv_bfloat16 r() const { return e[0]; }
    __device__ inline __nv_bfloat16 g() const { return e[1]; }
    __device__ inline __nv_bfloat16 b() const { return e[2]; }

    __device__ inline const vec3 &operator+() const { return *this; }
    __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __device__ inline __nv_bfloat16 operator[](int i) const { return e[i]; }
    __device__ inline __nv_bfloat16 &operator[](int i) { return e[i]; };

    __device__ inline vec3 &operator+=(const vec3 &v2);
    __device__ inline vec3 &operator-=(const vec3 &v2);
    __device__ inline vec3 &operator*=(const vec3 &v2);
    __device__ inline vec3 &operator/=(const vec3 &v2);
    __device__ inline vec3 &operator*=(const __nv_bfloat16 t);
    __device__ inline vec3 &operator/=(const __nv_bfloat16 t);

    __device__ inline __nv_bfloat16 length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __device__ inline __nv_bfloat16 squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __device__ inline void make_unit_vector();

    __nv_bfloat16 e[3];
};

// inline std::istream &operator>>(std::istream &is, vec3 &t)
// {
//     is >> __bfloat162float(t.e[0]) >> __bfloat162float(t.e[1]) >> __bfloat162float(t.e[2]);
//     return is;
// }

inline std::ostream &operator<<(std::ostream &os, const vec3 &t)
{
    os << __bfloat162float(t.e[0]) << " " << __bfloat162float(t.e[1]) << " " << __bfloat162float(t.e[2]);
    return os;
}

__device__ inline void vec3::make_unit_vector()
{
    __nv_bfloat16 k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] = e[0] * k;
    e[1] = e[1] * k;
    e[2] = e[2] * k;
}

/* vector 간 연산 */
__device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

/* vector & scalar 연산 */
__device__ inline vec3 operator*(const vec3 &v, __nv_bfloat16 t)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ inline vec3 operator*(float t_f, const vec3 &v)
{
    __nv_bfloat16 t = __float2bfloat16(t_f);
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ inline vec3 operator/(vec3 v, __nv_bfloat16 t)
{
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__device__ inline vec3 operator/(vec3 v, float t_f)
{
    __nv_bfloat16 t = __float2bfloat16(t_f);
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}


__device__ inline __nv_bfloat16 dot(const vec3 &v1, const vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__device__ inline vec3 &vec3::operator+=(const vec3 &v)
{
    e[0] = e[0] + v.e[0];
    e[1] = e[1] + v.e[1];
    e[2] = e[2] + v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator*=(const vec3 &v)
{
    e[0] = e[0] * v.e[0];
    e[1] = e[1] * v.e[1];
    e[2] = e[2] * v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator/=(const vec3 &v)
{
    e[0] = e[0] / v.e[0];
    e[1] = e[1] / v.e[1];
    e[2] = e[2] / v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator-=(const vec3 &v)
{
    e[0] = e[0] - v.e[0];
    e[1] = e[1] - v.e[1];
    e[2] = e[2] - v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator*=(const __nv_bfloat16 t)
{
    e[0] = e[0] * t;
    e[1] = e[1] * t;
    e[2] = e[2] * t;
    return *this;
}

__device__ inline vec3 &vec3::operator/=(const __nv_bfloat16 t)
{
    __nv_bfloat16 k = 1.0 / t;

    e[0] = e[0] * k;
    e[1] = e[1] * k;
    e[2] = e[2] * k;
    return *this;
}

__device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

#endif
