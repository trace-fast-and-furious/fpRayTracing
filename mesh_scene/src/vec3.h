#pragma once
#ifndef VEC3_H
#define VEC3_H

#include "utility.h"

using namespace custom_precision_fp;

class Vec3
{
public:
	Vec3() : e{{0, 0, 0}} {}
	Vec3(fp_orig e0, fp_orig e1, fp_orig e2) : e{{fp_orig_to_custom(e0), fp_orig_to_custom(e1), fp_orig_to_custom(e2)}} {}
	Vec3(fp_custom e0, fp_custom e1, fp_custom e2) : e{{e0, e1, e2}} {}

	fp_custom x() const { return e.val[0]; }
	fp_custom y() const { return e.val[1]; }
	fp_custom z() const { return e.val[2]; }

	Vec3 operator-() const { return Vec3(-(e.val[0]), -(e.val[1]), -(e.val[2])); }
	fp_custom operator[](int i) const { return e.val[i]; }
	fp_custom &operator[](int i) { return e.val[i]; }

	Vec3 &operator+=(const Vec3 &v)
	{
		e.val[0] = e.val[0] + v.e.val[0];
		e.val[1] = e.val[1] + v.e.val[1];
		e.val[2] = e.val[2] + v.e.val[2];

		return *this;
	}

	Vec3 &operator*=(const fp_custom t)
	{
		e.val[0] = e.val[0] * t;
		e.val[1] = e.val[1] * t;
		e.val[2] = e.val[2] * t;

		return *this;
	}

	Vec3 &operator*=(const fp_orig t)
	{
		e.val[0] = e.val[0] * t;
		e.val[1] = e.val[1] * t;
		e.val[2] = e.val[2] * t;

		return *this;
	}

	Vec3 &operator/=(fp_custom t)
	{
		fp_custom temp = 1 / t;
		*this = Vec3(e.val[0] * temp, e.val[1] * temp, e.val[2] * temp);
		return *this;
	}

	Vec3 &operator/=(fp_orig t)
	{
		fp_orig temp = 1 / t;
		*this = Vec3(e.val[0] * temp, e.val[1] * temp, e.val[2] * temp);
		return *this;
	}

	fp_custom length() const { return sqrt(length_squared()); }
	fp_custom length_squared() const { return e.val[0] * e.val[0] + e.val[1] * e.val[1] + e.val[2] * e.val[2]; }

	bool near_zero() const
	{
		fp_custom s = fp_orig_to_custom(1e-8);
		return (abs(e.val[0]) < s) && (abs(e.val[1]) < s) && (abs(e.val[2]) < s);
	}

	static Vec3 random() { return Vec3(random_num(), random_num(), random_num()); }
	static Vec3 random(fp_custom min, fp_custom max) { return Vec3(random_num(min, max), random_num(min, max), random_num(min, max)); }
	static Vec3 random(fp_orig min, fp_orig max) { return Vec3(random_num(min, max), random_num(min, max), random_num(min, max)); }

public:
	e_custom e;
};

// utility functions
inline std::ostream &operator<<(std::ostream &out, const Vec3 &v)
{
	return out << v.e.val[0].val[0] << ' ' << v.e.val[1].val[0] << ' ' << v.e.val[2].val[0];
}

inline Vec3 operator+(const Vec3 &u, const Vec3 &v)
{
	return Vec3(u.e.val[0] + v.e.val[0], u.e.val[1] + v.e.val[1], u.e.val[2] + v.e.val[2]);
}

inline Vec3 operator-(const Vec3 &u, const Vec3 &v)
{
	return Vec3(u.e.val[0] - v.e.val[0], u.e.val[1] - v.e.val[1], u.e.val[2] - v.e.val[2]);
}

inline Vec3 operator*(const Vec3 &u, const Vec3 &v) { return Vec3(u.e.val[0] * v.e.val[0], u.e.val[1] * v.e.val[1], u.e.val[2] * v.e.val[2]); }
inline Vec3 operator*(fp_custom t, const Vec3 &v) { return Vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }
inline Vec3 operator*(const Vec3 &v, fp_custom t) { return t * v; }
inline Vec3 operator*(const Vec3 &v, fp_orig t) { return Vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }
inline Vec3 operator*(fp_orig t, const Vec3 &v) { return Vec3(t * v.e.val[0], t * v.e.val[1], t * v.e.val[2]); }

inline Vec3 operator/(Vec3 v, fp_custom t) { return (1 / t) * v; }
inline Vec3 operator/(Vec3 v, fp_orig t) { return (1 / t) * v; }
inline Vec3 operator/(fp_orig t, Vec3 v) { return (1 / t) * v; }

inline bool isequal(Vec3 v1, Vec3 v2)
{
	return v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2];
}

inline fp_custom dot(const Vec3 &u, const Vec3 &v)
{
	return u.e.val[0] * v.e.val[0] + u.e.val[1] * v.e.val[1] + u.e.val[2] * v.e.val[2];
}

inline Vec3 cross(const Vec3 &u, const Vec3 &v)
{
	return Vec3(u.e.val[1] * v.e.val[2] - u.e.val[2] * v.e.val[1],
				u.e.val[2] * v.e.val[0] - u.e.val[0] * v.e.val[2],
				u.e.val[0] * v.e.val[1] - u.e.val[1] * v.e.val[0]);
}

inline Vec3 unit_vector(Vec3 v)
{

	return v / v.length();
}

inline Vec3 random_in_unit_sphere(bool t)
{
	while (true)
	{
		Vec3 p = Vec3::random(-1, 1);
		if (p.length_squared() >= 1)
			continue;

		return p;
	}
}

inline Vec3 random_unit_vector(bool t)
{
	return unit_vector(random_in_unit_sphere(true));
}

inline Vec3 random_in_hemisphere(const Vec3 &normal)
{
	Vec3 in_unit_sphere = random_in_unit_sphere(true);

	if (dot(in_unit_sphere, normal) > 0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

inline Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
	return v - 2 * dot(v, n) * n;
}

inline Vec3 refract(const Vec3 &uv, const Vec3 &n, fp_custom etai_over_etat)
{
	fp_custom cos_theta = min(dot(-uv, n), 1);
	Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	Vec3 r_out_parallel = -sqrt(abs(1 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

Vec3 random_in_unit_disk(bool t)
{
	while (true)
	{
		Vec3 p = Vec3(random_num(-1, 1), random_num(-1, 1), 0);
		if (p.length_squared() >= 1)
			continue;
		return p;
	}
}

inline void print_Vec3(const Vec3 &v)
{
	cout << "\tv[0] : " << v.e.val[0].val[0] << endl;
	cout << "\tv[1] : " << v.e.val[1].val[0] << endl;
	cout << "\tv[2] : " << v.e.val[2].val[0] << endl;
}

using Point3 = Vec3;
using Color = Vec3;

#endif