#pragma once
#ifndef VEC3_H
#define VEC3_H

#include "cpfloat.cuh"

using namespace custom_precision_fp;

class vec3
{
public:
    __device__ vec3() : e{{0, 0, 0}} {}
    __device__ vec3(fp_orig e0, fp_orig e1, fp_orig e2) : e{{fp_orig_to_custom(e0), fp_orig_to_custom(e1), fp_orig_to_custom(e2)}} {}
    __device__ vec3(fp_custom e0, fp_custom e1, fp_custom e2) : e{{e0, e1, e2}} {}

    __device__ inline fp_custom x() const { return e.val[0]; }
    __device__ inline fp_custom y() const { return e.val[1]; }
    __device__ inline fp_custom z() const { return e.val[2]; }

    __device__ inline vec3 operator-() const { return vec3(-(e.val[0]), -(e.val[1]), -(e.val[2])); }
    __device__ inline fp_custom operator[](int i) const { return e.val[i]; }
    __device__ inline fp_custom &operator[](int i) { return e.val[i]; }

    __device__ inline vec3 &operator+=(const vec3 &v);
    __device__ inline vec3 &operator-=(const vec3 &v);
    __device__ inline vec3 &operator*=(const vec3 &v);
    __device__ inline vec3 &operator*=(const fp_custom t);
    __device__ inline vec3 &operator*=(const fp_orig t);
    __device__ inline vec3 &operator/=(const vec3 &v);
    __device__ inline vec3 &operator/=(const fp_custom t);
    __device__ inline vec3 &operator/=(const fp_orig t);

    __device__ inline fp_custom length_squared() const { return e.val[0] * e.val[0] + e.val[1] * e.val[1] + e.val[2] * e.val[2]; }
    __device__ inline fp_custom length() const { return sqrt(e.val[0] * e.val[0] + e.val[1] * e.val[1] + e.val[2] * e.val[2]); }

    __device__ inline void make_unit_vector()
    {
        fp_custom k = 1.0 / sqrt(e.val[0] * e.val[0] + e.val[1] * e.val[1] + e.val[2] * e.val[2]);
        e.val[0] = e.val[0] * k;
        e.val[1] = e.val[1] * k;
        e.val[2] = e.val[2] * k;
    }
    __device__ inline bool near_zero() const
    {
        fp_custom s = fp_orig_to_custom(1e-8);
        return (abs(e.val[0]) < s) && (abs(e.val[1]) < s) && (abs(e.val[2]) < s);
    }

public:
    e_custom e;
};

// utility functions
// __device__ inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
// {
//     return out << v.e.val[0].val[0] << ' ' << v.e.val[1].val[0] << ' ' << v.e.val[2].val[0];
// }

__device__ vec3 &vec3::operator+=(const vec3 &v)
{
    e.val[0] = e.val[0] + v.e.val[0];
    e.val[1] = e.val[1] + v.e.val[1];
    e.val[2] = e.val[2] + v.e.val[2];

    return *this;
}

__device__ vec3 &vec3::operator-=(const vec3 &v)
{
    e.val[0] = e.val[0] - v.e.val[0];
    e.val[1] = e.val[1] - v.e.val[1];
    e.val[2] = e.val[2] - v.e.val[2];

    return *this;
}

__device__ inline vec3 &vec3::operator*=(const vec3 &v)
{
    e.val[0] = e.val[0] * v.e.val[0];
    e.val[1] = e.val[1] * v.e.val[1];
    e.val[2] = e.val[2] * v.e.val[2];

    return *this;
}

__device__ inline vec3 &vec3::operator*=(const fp_custom t)
{
    e.val[0] = e.val[0] * t;
    e.val[1] = e.val[1] * t;
    e.val[2] = e.val[2] * t;

    return *this;
}

__device__ inline vec3 &vec3::operator*=(const fp_orig t)
{
    e.val[0] = e.val[0] * t;
    e.val[1] = e.val[1] * t;
    e.val[2] = e.val[2] * t;

    return *this;
}

__device__ inline vec3 &vec3::operator/=(const vec3 &v)
{
    e.val[0] = e.val[0] / v.e.val[0];
    e.val[1] = e.val[1] / v.e.val[1];
    e.val[2] = e.val[2] / v.e.val[2];

    return *this;
}

__device__ inline vec3 &vec3::operator/=(fp_custom t)
{
    fp_custom temp = 1 / t;
    *this = vec3(e.val[0] * temp, e.val[1] * temp, e.val[2] * temp);
    return *this;
}

__device__ inline vec3 &vec3::operator/=(fp_orig t)
{
    fp_orig temp = 1 / t;
    *this = vec3(e.val[0] * temp, e.val[1] * temp, e.val[2] * temp);
    return *this;
}

__device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e.val[0] + v.e.val[0], u.e.val[1] + v.e.val[1], u.e.val[2] + v.e.val[2]);
}

__device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e.val[0] - v.e.val[0], u.e.val[1] - v.e.val[1], u.e.val[2] - v.e.val[2]);
}

__device__ inline vec3 operator*(const vec3 &u, const vec3 &v) { return vec3(u.e.val[0] * v.e.val[0], u.e.val[1] * v.e.val[1], u.e.val[2] * v.e.val[2]); }
__device__ inline vec3 operator*(fp_custom t, const vec3 &v) { return vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }
__device__ inline vec3 operator*(const vec3 &v, fp_custom t) { return t * v; }
__device__ inline vec3 operator*(const vec3 &v, fp_orig t) { return vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }
__device__ inline vec3 operator*(fp_orig t, const vec3 &v) { return vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }

__device__ inline vec3 operator/(vec3 v, fp_custom t) { return (1 / t) * v; }
__device__ inline vec3 operator/(vec3 v, fp_orig t) { return (1 / t) * v; }
__device__ inline vec3 operator/(fp_orig t, vec3 v) { return (1 / t) * v; }

__device__ inline bool isequal(vec3 v1, vec3 v2) { return v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]; }

__device__ inline fp_custom dot(const vec3 &u, const vec3 &v)
{
    return u.e.val[0] * v.e.val[0] + u.e.val[1] * v.e.val[1] + u.e.val[2] * v.e.val[2];
}

__device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e.val[1] * v.e.val[2] - u.e.val[2] * v.e.val[1],
                u.e.val[2] * v.e.val[0] - u.e.val[0] * v.e.val[2],
                u.e.val[0] * v.e.val[1] - u.e.val[1] * v.e.val[0]);
}

__device__ inline vec3 unit_vector(vec3 v)
{

    return v / v.length();
}

using point3 = vec3;
using color = vec3;

#endif