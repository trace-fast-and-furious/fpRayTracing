#pragma once
#ifndef VEC3_H
#define VEC3_H

#include "utility.h"

using namespace custom_precision_fp;

class vec3
{
public:
    vec3() : e{{0, 0, 0}} {}
    vec3(fp_orig e0, fp_orig e1, fp_orig e2) : e{{fp_orig_to_custom(e0), fp_orig_to_custom(e1), fp_orig_to_custom(e2)}} {}
    vec3(fp_custom e0, fp_custom e1, fp_custom e2) : e{{e0, e1, e2}} {}

    fp_custom x() const { return e.val[0]; }
    fp_custom y() const { return e.val[1]; }
    fp_custom z() const { return e.val[2]; }

    vec3 operator-() const { return vec3(-(e.val[0]), -(e.val[1]), -(e.val[2])); }
    fp_custom operator[](int i) const { return e.val[i]; }
    fp_custom &operator[](int i) { return e.val[i]; }

    vec3 &operator+=(const vec3 &v)
    {
        e.val[0] = e.val[0] + v.e.val[0];
        e.val[1] = e.val[1] + v.e.val[1];
        e.val[2] = e.val[2] + v.e.val[2];

        return *this;
    }

    vec3 &operator*=(const fp_custom t)
    {
        e.val[0] = e.val[0] * t;
        e.val[1] = e.val[1] * t;
        e.val[2] = e.val[2] * t;

        return *this;
    }

    vec3 &operator*=(const fp_orig t)
    {
        e.val[0] = e.val[0] * t;
        e.val[1] = e.val[1] * t;
        e.val[2] = e.val[2] * t;

        return *this;
    }

    vec3 &operator/=(fp_custom t)
    {
        fp_custom temp = 1 / t;
        *this = vec3(e.val[0] * temp, e.val[1] * temp, e.val[2] * temp);
        return *this;
    }

    vec3 &operator/=(fp_orig t)
    {
        fp_orig temp = 1 / t;
        *this = vec3(e.val[0] * temp, e.val[1] * temp, e.val[2] * temp);
        return *this;
    }

    fp_custom length() const { return sqrt(length_squared()); }
    fp_custom length_squared() const { return e.val[0] * e.val[0] + e.val[1] * e.val[1] + e.val[2] * e.val[2]; }

    bool near_zero() const
    {
        fp_custom s = fp_orig_to_custom(1e-8);
        return (abs(e.val[0]) < s) && (abs(e.val[1]) < s) && (abs(e.val[2]) < s);
    }

    static vec3 random() { return vec3(random_num(), random_num(), random_num()); }
    static vec3 random(fp_custom min, fp_custom max) { return vec3(random_num(min, max), random_num(min, max), random_num(min, max)); }
    static vec3 random(fp_orig min, fp_orig max) { return vec3(random_num(min, max), random_num(min, max), random_num(min, max)); }

public:
    e_custom e;
};

// utility functions
inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e.val[0].val[0] << ' ' << v.e.val[1].val[0] << ' ' << v.e.val[2].val[0];
}

inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e.val[0] + v.e.val[0], u.e.val[1] + v.e.val[1], u.e.val[2] + v.e.val[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e.val[0] - v.e.val[0], u.e.val[1] - v.e.val[1], u.e.val[2] - v.e.val[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v) { return vec3(u.e.val[0] * v.e.val[0], u.e.val[1] * v.e.val[1], u.e.val[2] * v.e.val[2]); }
inline vec3 operator*(fp_custom t, const vec3 &v) { return vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }
inline vec3 operator*(const vec3 &v, fp_custom t) { return t * v; }
inline vec3 operator*(const vec3 &v, fp_orig t) { return vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }
inline vec3 operator*(fp_orig t, const vec3 &v) { return vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }

inline vec3 operator/(vec3 v, fp_custom t) { return (1 / t) * v; }
inline vec3 operator/(vec3 v, fp_orig t) { return (1 / t) * v; }
inline vec3 operator/(fp_orig t, vec3 v) { return (1 / t) * v; }

inline bool isequal(vec3 v1, vec3 v2)
{
    return v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2];
}

inline fp_custom dot(const vec3 &u, const vec3 &v)
{
    return u.e.val[0] * v.e.val[0] + u.e.val[1] * v.e.val[1] + u.e.val[2] * v.e.val[2];
}

inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e.val[1] * v.e.val[2] - u.e.val[2] * v.e.val[1],
                u.e.val[2] * v.e.val[0] - u.e.val[0] * v.e.val[2],
                u.e.val[0] * v.e.val[1] - u.e.val[1] * v.e.val[0]);
}

inline vec3 unit_vector(vec3 v)
{

    return v / v.length();
}

inline vec3 random_in_unit_sphere(bool t)
{
    while (true)
    {
        vec3 p = vec3::random(-1, 1);
        if (p.length_squared() >= 1)
            continue;

        return p;
    }
}

inline vec3 random_unit_vector(bool t)
{
    return unit_vector(random_in_unit_sphere(true));
}

inline vec3 random_in_hemisphere(const vec3 &normal)
{
    vec3 in_unit_sphere = random_in_unit_sphere(true);

    if (dot(in_unit_sphere, normal) > 0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

inline vec3 refract(const vec3 &uv, const vec3 &n, fp_custom etai_over_etat)
{
    fp_custom cos_theta = min(dot(-uv, n), 1);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(abs(1 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

vec3 random_in_unit_disk(bool t)
{

    while (true)
    {
        vec3 p = vec3(random_num(-1, 1), random_num(-1, 1), 0);
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
}

inline void print_vec3(const vec3 &v)
{
    cout << "\tv[0] : " << v.e.val[0].val[0] << endl;
    cout << "\tv[1] : " << v.e.val[1].val[0] << endl;
    cout << "\tv[2] : " << v.e.val[2].val[0] << endl;
}

using point3 = vec3;
using color = vec3;

#endif