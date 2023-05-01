#include "settings.cuh"

namespace custom_precision_fp
{
    /* conversion functions */
    __device__ __host__ fp_custom fp_orig_to_custom(fp_orig a)
    {
        fp_orig O[1] = {a};
        fp_custom C = {{}};

        cpfloat(C.val, O, 1, fpopts);
        return C;
    }

    __device__ __host__ fp_orig val(fp_custom a) { return a.val[0]; }

    /* arithmetic functions (between 2 fp_custom nums)*/
    __device__ __host__ fp_custom add(fp_custom a, fp_custom b)
    {
        fp_custom res = {{}};
        cpf_add(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom add(fp_orig a_orig, fp_custom b)
    {
        fp_custom a = fp_orig_to_custom(a_orig);
        fp_custom res = {{}};
        cpf_add(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom add(fp_custom a, fp_orig b_orig)
    {
        fp_custom b = fp_orig_to_custom(b_orig);
        fp_custom res = {{}};
        cpf_add(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom sub(fp_custom a, fp_custom b)
    {
        fp_custom res = {{}};
        cpf_sub(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom sub(fp_orig a_orig, fp_custom b)
    {
        fp_custom a = fp_orig_to_custom(a_orig);
        fp_custom res = {{}};
        cpf_sub(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom sub(fp_custom a, fp_orig b_orig)
    {
        fp_custom b = fp_orig_to_custom(b_orig);
        fp_custom res = {{}};
        cpf_sub(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom mul(fp_custom a, fp_custom b)
    {
        fp_custom res = {{}};
        cpf_mul(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom mul(fp_orig a_orig, fp_custom b)
    {
        fp_custom a = fp_orig_to_custom(a_orig);
        fp_custom res = {{}};
        cpf_mul(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom mul(fp_custom a, fp_orig b_orig)
    {
        fp_custom b = fp_orig_to_custom(b_orig);
        fp_custom res = {{}};
        cpf_mul(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom div(fp_custom a, fp_custom b)
    {
        fp_custom res = {{}};
        cpf_div(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom div(fp_orig a_orig, fp_custom b)
    {
        fp_custom a = fp_orig_to_custom(a_orig);
        fp_custom res = {{}};
        cpf_div(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom div(fp_custom a, fp_orig b_orig)
    {
        fp_custom b = fp_orig_to_custom(b_orig);
        fp_custom res = {{}};
        cpf_div(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom min(fp_custom a, fp_custom b)
    {
        fp_custom res = {{}};
        cpf_fmin(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom min(fp_orig a_orig, fp_custom b)
    {
        fp_custom a = fp_orig_to_custom(a_orig);
        fp_custom res = {{}};
        cpf_fmin(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom min(fp_custom a, fp_orig b_orig)
    {
        fp_custom b = fp_orig_to_custom(b_orig);
        fp_custom res = {{}};
        cpf_fmin(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom max(fp_custom a, fp_custom b)
    {
        fp_custom res = {{}};
        cpf_fmax(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom max(fp_orig a_orig, fp_custom b)
    {
        fp_custom a = fp_orig_to_custom(a_orig);
        fp_custom res = {{}};
        cpf_fmax(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom max(fp_custom a, fp_orig b_orig)
    {
        fp_custom b = fp_orig_to_custom(b_orig);
        fp_custom res = {{}};
        cpf_fmax(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom sqrt(fp_custom a)
    {
        fp_custom res = {{}};
        cpf_sqrt(res.val, a.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom pow(fp_custom a, fp_orig b_orig)
    {
        fp_custom res = {{}};
        fp_custom b = fp_orig_to_custom(b_orig);
        cpf_pow(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom pow(fp_custom a, fp_custom b)
    {
        fp_custom res = {{}};
        cpf_pow(res.val, a.val, b.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom abs(fp_custom a)
    {
        fp_custom res = {{}};
        cpf_fabs(res.val, a.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom neg(fp_custom a)
    {
        fp_custom res = {{-a.val[0]}};
        return res;
    }

    __device__ __host__ fp_custom tan(fp_custom a)
    {
        fp_custom res = {{}};
        cpf_tan(res.val, a.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom sin(fp_custom a)
    {
        fp_custom res = {{}};
        cpf_sin(res.val, a.val, 1, fpopts);
        return res;
    }

    __device__ __host__ fp_custom cos(fp_custom a)
    {
        fp_custom res = {{}};
        cpf_cos(res.val, a.val, 1, fpopts);
        return res;
    }

    /* operator overloading */
    __device__ __host__ __forceinline__ fp_custom operator+(fp_custom a, fp_custom b) { return add(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator+(fp_orig a, fp_custom b) { return add(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator+(fp_custom a, fp_orig b) { return add(a, b); }

    __device__ __host__ __forceinline__ fp_custom operator-(fp_custom a, fp_custom b) { return sub(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator-(fp_orig a, fp_custom b) { return sub(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator-(fp_custom a, fp_orig b) { return sub(a, b); }

    __device__ __host__ __forceinline__ fp_custom operator-(fp_custom a) { return neg(a); }

    __device__ __host__ __forceinline__ fp_custom operator*(fp_custom a, fp_custom b) { return mul(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator*(fp_orig a, fp_custom b) { return mul(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator*(fp_custom a, fp_orig b) { return mul(a, b); }

    __device__ __host__ __forceinline__ fp_custom operator/(fp_custom a, fp_custom b) { return div(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator/(fp_orig a, fp_custom b) { return div(a, b); }
    __device__ __host__ __forceinline__ fp_custom operator/(fp_custom a, fp_orig b) { return div(a, b); }

    __device__ __host__ __forceinline__ bool operator>(fp_custom a, fp_custom b) { return a.val[0] > b.val[0]; }
    __device__ __host__ __forceinline__ bool operator>(fp_orig a, fp_custom b) { return a > b.val[0]; }
    __device__ __host__ __forceinline__ bool operator>(fp_custom a, fp_orig b) { return a.val[0] > b; }

    __device__ __host__ __forceinline__ bool operator<(fp_custom a, fp_custom b) { return a.val[0] < b.val[0]; }
    __device__ __host__ __forceinline__ bool operator<(fp_orig a, fp_custom b) { return a < b.val[0]; }
    __device__ __host__ __forceinline__ bool operator<(fp_custom a, fp_orig b) { return a.val[0] < b; }

    __device__ __host__ __forceinline__ bool operator>=(fp_custom a, fp_custom b) { return a.val[0] >= b.val[0]; }
    __device__ __host__ __forceinline__ bool operator>=(fp_orig a, fp_custom b) { return a >= b.val[0]; }
    __device__ __host__ __forceinline__ bool operator>=(fp_custom a, fp_orig b) { return a.val[0] >= b; }

    __device__ __host__ __forceinline__ bool operator<=(fp_custom a, fp_custom b) { return a.val[0] <= b.val[0]; }
    __device__ __host__ __forceinline__ bool operator<=(fp_orig a, fp_custom b) { return a <= b.val[0]; }
    __device__ __host__ __forceinline__ bool operator<=(fp_custom a, fp_orig b) { return a.val[0] <= b; }

    __device__ __host__ __forceinline__ bool operator==(fp_custom a, fp_custom b) { return a.val[0] == b.val[0]; }
    __device__ __host__ __forceinline__ bool operator==(fp_orig a, fp_custom b) { return a == b.val[0]; }
    __device__ __host__ __forceinline__ bool operator==(fp_custom a, fp_orig b) { return a.val[0] == b; }

    __device__ __host__ __forceinline__ bool operator!=(fp_custom a, fp_custom b) { return a.val[0] != b.val[0]; }
}