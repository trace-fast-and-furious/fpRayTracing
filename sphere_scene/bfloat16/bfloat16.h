#pragma once
#ifndef BFLOAT16H
#define BFLOAT16H

#include <time.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <curand_kernel.h>
#include <cuda_bf16.h>

namespace bfloat16
{
    /* arithmetic functions */
    __device__ __forceinline__ __nv_bfloat16 operator+(const float a, const __nv_bfloat16 b) { return __hadd(__float2bfloat16(a), b); }
    __device__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16 a, const float b) { return __hadd(a, __float2bfloat16(b)); }

    __device__ __forceinline__ __nv_bfloat16 operator-(const float a, const __nv_bfloat16 b) { return __hsub(__float2bfloat16(a), b); }
    __device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16 a, const float b) { return __hsub(a, __float2bfloat16(b)); }

    __device__ __forceinline__ __nv_bfloat16 operator*(const float a, const __nv_bfloat16 b) { return __hmul(__float2bfloat16(a), b); }
    __device__ __forceinline__ __nv_bfloat16 operator*(const __nv_bfloat16 a, const float b) { return __hmul(a, __float2bfloat16(b)); }

    __device__ __forceinline__ __nv_bfloat16 operator/(const float a, const __nv_bfloat16 b) { return __hdiv(__float2bfloat16(a), b); }
    __device__ __forceinline__ __nv_bfloat16 operator/(const __nv_bfloat16 a, const float b) { return __hdiv(a, __float2bfloat16(b)); }

    // __device__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16 a) {return __hneg(a);}

    /* comparison functions */
    __device__ __forceinline__ bool operator>(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hgt(a, b); }
    __device__ __forceinline__ bool operator>(const float a, const __nv_bfloat16 b) { return __hgt(__float2bfloat16(a), b); }
    __device__ __forceinline__ bool operator>(const __nv_bfloat16 a, const float b) { return __hgt(a, __float2bfloat16(b)); }

    __device__ __forceinline__ bool operator>=(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hge(a, b); }
    __device__ __forceinline__ bool operator>=(const float a, const __nv_bfloat16 b) { return __hge(__float2bfloat16(a), b); }
    __device__ __forceinline__ bool operator>=(const __nv_bfloat16 a, const float b) { return __hge(a, __float2bfloat16(b)); }

    __device__ __forceinline__ bool operator<(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hlt(a, b); }
    __device__ __forceinline__ bool operator<(const float a, const __nv_bfloat16 b) { return __hlt(__float2bfloat16(a), b); }
    __device__ __forceinline__ bool operator<(const __nv_bfloat16 a, const float b) { return __hlt(a, __float2bfloat16(b)); }

    __device__ __forceinline__ bool operator<=(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hle(a, b); }
    __device__ __forceinline__ bool operator<=(const float a, const __nv_bfloat16 b) { return __hle(__float2bfloat16(a), b); }
    __device__ __forceinline__ bool operator<=(const __nv_bfloat16 a, const float b) { return __hle(a, __float2bfloat16(b)); }

    __device__ __forceinline__ bool operator==(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __heq(a, b); }
    __device__ __forceinline__ bool operator==(const float a, const __nv_bfloat16 b) { return __heq(__float2bfloat16(a), b); }
    __device__ __forceinline__ bool operator==(const __nv_bfloat16 a, const float b) { return __heq(a, __float2bfloat16(b)); }

    __device__ __forceinline__ bool operator!=(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hne(a, b); }
    __device__ __forceinline__ bool operator!=(const float a, const __nv_bfloat16 b) { return __hne(__float2bfloat16(a), b); }
    __device__ __forceinline__ bool operator!=(const __nv_bfloat16 a, const float b) { return __hne(a, __float2bfloat16(b)); }

    __device__ __forceinline__ __nv_bfloat16 max(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hmax(a, b); }
    __device__ __forceinline__ __nv_bfloat16 min(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hmin(a, b); }

    /* math functions */
    __device__ __forceinline__ __nv_bfloat16 sqrt(const __nv_bfloat16 a) { return hsqrt(a); }
    __device__ __forceinline__ __nv_bfloat16 pow(const __nv_bfloat16 a, float b) { return __float2bfloat16(std::pow(__bfloat162float(a), b)); }
    __device__ __forceinline__ __nv_bfloat16 pow(const __nv_bfloat16 a, __nv_bfloat16 b) { return __float2bfloat16(std::pow(__bfloat162float(a), __bfloat162float(b))); }
    __device__ __forceinline__ __nv_bfloat16 sin(const __nv_bfloat16 a) { return hsin(a); }
    __device__ __forceinline__ __nv_bfloat16 cos(const __nv_bfloat16 a) { return hcos(a); }
    __device__ __forceinline__ __nv_bfloat16 tan(const __nv_bfloat16 a) { return hsin(a) / hcos(a); }

}

#endif