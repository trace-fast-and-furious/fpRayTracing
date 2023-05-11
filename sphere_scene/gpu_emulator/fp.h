#pragma once
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <string.h>

#include "print.h"

using namespace std;

namespace floating_point
{
    const long long int FP_MANT_EXTRACT_BITS = ((int)std::pow(2, FP_MANT_BITSIZE) - 1);
    const long long int FP_EXP_EXTRACT_BITS = (1 << FP_EXP_BITSIZE) - 1;
    const long long int FP_SIGN_EXTRACT_BITS = (1 << (FP_MANT_BITSIZE + FP_EXP_BITSIZE));

    const long long int __FPO_MANT_BITSIZE = typeid(__fpo) == typeid(float) ? FLOAT_MANT_BITSIZE : DOUBLE_MANT_BITSIZE;
    const long long int __FPO_EXP_BITSIZE = typeid(__fpo) == typeid(float) ? FLOAT_EXP_BITSIZE : DOUBLE_EXP_BITSIZE;
    const long long int __FPO_MANT_EXTRACT_BITS = typeid(__fpo) == typeid(float) ? FLOAT_MANT_EXTRACT_BITS : DOUBLE_MANT_EXTRACT_BITS;
    const long long int __FPO_EXP_EXTRACT_BITS = typeid(__fpo) == typeid(float) ? FLOAT_EXP_EXTRACT_BITS : DOUBLE_EXP_EXTRACT_BITS;
    const long long int __FPO_BIAS = typeid(__fpo) == typeid(float) ? FLOAT_BIAS : DOUBLE_BIAS;
    const long long int LONG_ALL_ONES = 0xffffffffffffffff;
    const long long int INT_ALL_ONES = 0xffffffff;

    /* type conversions */
    float fp_to_fpo(fp b)
    {
        long long t = 0;
        long long sign, exp, mant;

        /* exceptions: 0, inf, NaN*/
        if (b.exp == 0 && b.mant == 0)
        {
            if (b.mant == 0)
                return 0;
            else
                exp = 0;
        }
        else if (b.exp == FP_EXP_EXTRACT_BITS)
        {
            b.exp = __FPO_EXP_EXTRACT_BITS;
            if (b.mant == 0)
                return std::numeric_limits<__fpo>::infinity();
            else
                return std::numeric_limits<__fpo>::quiet_NaN();
        }
        sign = (long long)b.sign;
        if (b.exp) // if not 0
            exp = (long long)b.exp - FP_BIAS + __FPO_BIAS;
        mant = (long long)b.mant;

        if (__FPO_MANT_BITSIZE < FP_MANT_BITSIZE)
            mant >>= std::abs(FP_MANT_BITSIZE - __FPO_MANT_BITSIZE);
        else
            mant <<= std::abs(__FPO_MANT_BITSIZE - FP_MANT_BITSIZE);

        mant &= __FPO_MANT_EXTRACT_BITS; // remove implicit 1

        t ^= sign;
        t <<= __FPO_EXP_BITSIZE;
        t ^= exp;
        t <<= __FPO_MANT_BITSIZE;
        t ^= mant;

        __fpo *res = reinterpret_cast<__fpo *>(&t);
        return *res;
    }

    fp fpo_to_fp(__fpo f)
    {
        fp res = {0, 0, 0};

        // if (TEST_BIT)
        // {
        //     cout << f << " " << typeid(__fpo).name() << " " << FP_EXP_BITSIZE << "-" << FP_MANT_BITSIZE << endl;
        // }
        if (f == 0)
            return res;
        else if (std::isinf(f))
            return {0, (unsigned int)FP_EXP_EXTRACT_BITS, 0};
        else if (std::isnan(f))
            return {0, (unsigned int)FP_EXP_EXTRACT_BITS, 1};
        else
        {
            if (typeid(__fpo) == typeid(float))
            {
                // cast to integer for bitwise operations
                unsigned int *t = reinterpret_cast<unsigned int *>(&f);

                res.mant = *t & FLOAT_MANT_EXTRACT_BITS;
                *t >>= FLOAT_MANT_BITSIZE;
                res.exp = (*t & FLOAT_EXP_EXTRACT_BITS) - FLOAT_BIAS + FP_BIAS;
                *t >>= FLOAT_EXP_BITSIZE;
                res.sign = *t;
            }

            if (typeid(__fpo) == typeid(double))
            {
                // cast to integer for bitwise operations
                unsigned long long *t = reinterpret_cast<unsigned long long *>(&f);

                res.mant = *t & DOUBLE_MANT_EXTRACT_BITS;
                *t >>= DOUBLE_MANT_BITSIZE;
                res.exp = (*t & DOUBLE_EXP_EXTRACT_BITS) - DOUBLE_BIAS + FP_BIAS;
                *t >>= DOUBLE_EXP_BITSIZE;
                res.sign = *t;
            }

            // set FP's mantissa from float's extracted mantissa bits
            if (__FPO_MANT_BITSIZE > FP_MANT_BITSIZE)
                res.mant >>= std::abs(__FPO_MANT_BITSIZE - FP_MANT_BITSIZE);
            else
                res.mant <<= std::abs(__FPO_MANT_BITSIZE - FP_MANT_BITSIZE);

            // add implicit 1 when needed
            res.mant ^= (1 << FP_MANT_BITSIZE);
            // __fpo exponent 값이 fp가 담기에 너무 큰 경우
            if (res.exp > ((1 << FP_EXP_BITSIZE) - 1))
                return {1, (1 << FP_EXP_BITSIZE) - 1, (1 << FP_MANT_BITSIZE) - 1};

            return res;
        }
    }

    fp int_to_fp(int i) { return fpo_to_fp((float)i); }

    int fp_to_int(fp b) { return int(fp_to_fpo(b)); }

    /* frequently used fps */
    fp b_pi = fpo_to_fp(3.1415926535897932385);
    fp b_infinity = fpo_to_fp(std::numeric_limits<__fpo>::infinity());
    fp b_0 = {0, 0, 0};
    fp b_0_001 = fpo_to_fp(0.001);
    fp b_0_1 = fpo_to_fp(0.1);
    fp b_0_15 = fpo_to_fp(0.15);
    fp b_0_2 = fpo_to_fp(0.2);
    fp b_0_3 = fpo_to_fp(0.3);
    fp b_0_4 = fpo_to_fp(0.4);
    fp b_0_5 = fpo_to_fp(0.5);
    fp b_0_6 = fpo_to_fp(0.6);
    fp b_0_7 = fpo_to_fp(0.7);
    fp b_0_8 = fpo_to_fp(0.8);
    fp b_0_9 = fpo_to_fp(0.9);
    fp b_0_95 = fpo_to_fp(0.95);
    fp b_0_999 = fpo_to_fp(0.999);
    fp b_1 = int_to_fp(1);
    fp b_1_neg = int_to_fp(-1);
    fp b_1_5 = fpo_to_fp(1.5);
    fp b_2 = int_to_fp(2);
    fp b_3 = int_to_fp(3);
    fp b_4 = int_to_fp(4);
    fp b_9 = int_to_fp(9);
    fp b_10 = int_to_fp(10);
    fp b_11 = int_to_fp(11);
    fp b_13 = int_to_fp(13);
    fp b_16 = int_to_fp(16);
    fp b_20 = int_to_fp(20);
    fp b_121 = int_to_fp(121);
    fp b_180 = int_to_fp(180);
    fp b_256 = int_to_fp(256);
    fp b_1000 = int_to_fp(1000);
    fp b_1000_neg = int_to_fp(-1000);

    /* arithmetic operations for 2 numbers */
    fp add(fp a, fp b)
    {
        // std::cout << "add " << fp_to_fpo(a) << " " << fp_to_fpo(b) << endl;

        fp res = {0, 0, 0};

        // if both numbers are 0, skip process
        if ((a.mant == 0) && (a.exp == 0) && (b.mant == 0) && (b.exp == 0))
            return b_0;

        /* save mantissas in long long so that there is no data loss during
         * shifting */
        int temp_shift_num = 61 - FP_MANT_BITSIZE;
        long long res_mant_temp = 0;
        long long a_mant_temp = (long long)a.mant << temp_shift_num;
        long long b_mant_temp = (long long)b.mant << temp_shift_num;
        long long implicit_1 = (long long)1 << 61;

        // std::cout << "conversion" << endl;
        // printBit_ulong(a_mant_temp, true);
        // printBit_ulong(b_mant_temp, true);

        /* decide exponent */
        if (a.exp >= b.exp)
        {
            res.exp = a.exp;
            b_mant_temp >>= (a.exp - b.exp);
        }
        else
        {
            res.exp = b.exp;
            a_mant_temp >>= (b.exp - a.exp);
        }

        // std::cout << "\nstep 0: decide exponent" << endl;
        // printBit_ulong(a_mant_temp, true);
        // printBit_ulong(b_mant_temp, true);
        // printBit_ulong(res.exp, true);

        // if addition result is 0, skip process
        if ((a.sign != b.sign) && (a_mant_temp == b_mant_temp))
            return b_0;

        /* cal mantissa */
        // 1. conversion to 2's complement for negative mantissas
        if (a.sign)
        {
            a_mant_temp = ~a_mant_temp + 1;
        }
        if (b.sign)
        {
            b_mant_temp = ~b_mant_temp + 1;
        }

        // std::cout << "\nstep 1: conversion to 2's complement for negative mantissas"
        //           << endl;
        // printBit_ulong(a_mant_temp, true);
        // printBit_ulong(b_mant_temp, true);

        // 2. add mantissas
        res_mant_temp = a_mant_temp + b_mant_temp;

        // std::cout << "\nstep 2: add mantissas" << endl;
        // printBit_ulong(res_mant_temp, true);

        // 3. convert to signed magnitude if negative
        if (res_mant_temp & 0x8000000000000000)
        {
            res_mant_temp = ~res_mant_temp + 1;
            res.sign = 1;
        }

        // std::cout << "\nstep 3: convert to signed magnitude if negative" << endl;
        // printBit_ulong(res_mant_temp, true);

        // 4. normalization(implicit 1)
        // if (res.sign && (a.sign ^ b.sign)) // add implicit 1 for negative number addition
        //     res_mant_temp |= implicit_1;

        bool has_implicit_1 = (bool)(res_mant_temp >> 61);
        if (res.exp && !has_implicit_1)
        {
            while (!(res_mant_temp & implicit_1))
            { // if there is no implicit 1
                res_mant_temp <<= 1;
                res.exp -= 1;
            }
        }

        // std::cout << "\nstep 5: normalization(implicit 1)" << endl;
        // printBit_ulong(res_mant_temp, true);

        int carry = (int)(res_mant_temp >> 62);
        do
        {
            // 5. normalization(carry)
            while (carry)
            { // if there is carry
                res_mant_temp >>= 1;
                res.exp += 1;
                carry >>= 1;
            }

            // std::cout << "\nstep 4: normalization(carry)" << endl;
            // printBit_ulong(res_mant_temp, true);

            // 6. rounding to nearest even
            long long t = (long long)1 << temp_shift_num;
            unsigned short last_bit =
                (unsigned short)((res_mant_temp & t) >> temp_shift_num);
            t >>= 1;
            unsigned short ground_bit =
                (unsigned short)((res_mant_temp & t) >> temp_shift_num - 1);
            t >>= 1;
            unsigned short round_bit =
                (unsigned short)((res_mant_temp & t) >> temp_shift_num - 2);
            t -= 1;
            unsigned short sticky_bits = (unsigned short)((bool)(res_mant_temp & t));

            long long lsb = (long long)1 << temp_shift_num;
            if (ground_bit)
            {
                if (round_bit == 0 && sticky_bits == 0)
                { // round to even
                    if (last_bit)
                    {
                        res_mant_temp += lsb;
                    }
                }
                else
                {
                    res_mant_temp += lsb; // round up
                }
            }

            // std::cout << "\nstep 6: rounding" << endl;
            // printBit_ulong(res_mant_temp, true);
            // std::cout << last_bit << ground_bit << round_bit << sticky_bits << endl;

            carry = (int)(res_mant_temp >> 62);
        } while (carry);

        // 7. store result
        res.mant = (int)(res_mant_temp >> temp_shift_num);

        // std::cout << "done" << endl;
        return res;
    }

    fp sub(fp a, fp b)
    {
        if (b.exp == 0 && b.mant == 0)
            return a;
        b.sign ^= 1;
        return add(a, b);
    }

    fp mult(fp a, fp b)
    {
        fp res = {(unsigned short)(a.sign ^ b.sign), a.exp + b.exp - FP_BIAS, 0};
        unsigned long long res_mant_temp = 0;

        // if a or b is 0
        if (a.mant == 0 && a.exp == 0)
            return b_0;
        if (b.mant == 0 && b.exp == 0)
            return b_0;

        // underflow: if number is too small return 0
        if ((int)res.exp < 0)
            return b_0;

        // std::cout << "\nstep 0: conversion to temps"
        //           << endl;
        // printBit_ulong((unsigned long long)a.mant, true);
        // printBit_ulong((unsigned long long)b.mant, true);
        // printBit_ulong((unsigned long long)res.exp, true);

        // 1. multiply
        res_mant_temp = (unsigned long long)a.mant * (unsigned long long)b.mant;

        // std::cout << "\nstep 1: multiply" << endl;
        // printBit_ulong(res_mant_temp, true);
        // printBit_ulong((unsigned long long)res.exp, true);

        // 3. normalization(implicit 1)
        if (res.exp)
        {
            unsigned long long implicit_1 = (unsigned long long)1 << (FP_MANT_BITSIZE + FP_MANT_BITSIZE);
            bool has_implicit_1 = (bool)(res_mant_temp >> (FP_MANT_BITSIZE + FP_MANT_BITSIZE));
            // long long left_shift_cnt = 0;

            if (!has_implicit_1)
            { // exponent==0이면 implicit 1을 추가하지 않기 때문
                while (res.exp && !(res_mant_temp & implicit_1))
                { // if there is no implicit 1
                    res_mant_temp <<= 1;
                    // left_shift_cnt++;
                    res.exp -= 1;
                }
            }

            if (res.exp && (a.exp == 0 ^ b.exp == 0))
                res.exp += 1;

            // res.exp = (unsigned int)(std::max((long long)0, (long long)res.exp - left_shift_cnt));
        }

        // std::cout << "\nstep 3: normalization(implicit 1)" << endl;
        // printBit_ulong(res_mant_temp, true);
        // printBit_ulong((unsigned long long)res.exp, true);

        int carry = (int)(res_mant_temp >> (FP_MANT_BITSIZE + FP_MANT_BITSIZE + 1));
        do
        {
            // 4. normalization(Carry)
            while (carry)
            { // if there is carry
                res_mant_temp >>= 1;
                res.exp += 1;
                carry >>= 1;
            }

            // std::cout << "\nstep 4: normalization(carry)" << endl;
            // printBit_ulong(res_mant_temp, true);
            // printBit_ulong((unsigned long long)res.exp, true);

            // 5. rounding to nearest even
            int t = (int)std::pow(2, FP_MANT_BITSIZE);
            unsigned short last_bit =
                (unsigned short)((res_mant_temp & t) >> FP_MANT_BITSIZE);
            t >>= 1;
            unsigned short ground_bit =
                (unsigned short)((res_mant_temp & t) >> FP_MANT_BITSIZE - 1);
            t >>= 1;
            unsigned short round_bit =
                (unsigned short)((res_mant_temp & t) >> FP_MANT_BITSIZE - 2);
            t -= 1;
            unsigned short sticky_bits = (unsigned short)((bool)(res_mant_temp & t));

            int lsb = (int)std::pow(2, FP_MANT_BITSIZE);
            if (ground_bit)
            {
                if (round_bit == 0 && sticky_bits == 0)
                { // round to even
                    if (last_bit)
                    {
                        res_mant_temp += lsb;
                    }
                }
                else
                {
                    res_mant_temp += lsb; // round up
                }
            }

            // std::cout << "\nstep 5: rounding" << endl;
            // printBit_ulong(res_mant_temp, true);
            // std::cout << last_bit << ground_bit << round_bit << sticky_bits << endl;

            carry = (int)(res_mant_temp >> (FP_MANT_BITSIZE + FP_MANT_BITSIZE + 1)); // update carry
        } while (carry);

        res.mant =
            (int)(res_mant_temp >> FP_MANT_BITSIZE); // save result in res.mant

        return res;
    }

    fp div(fp a, fp b)
    {

        fp res = {(unsigned short)(a.sign ^ b.sign), a.exp - b.exp + FP_BIAS, 0};

        // if a is 0
        if (a.sign == 0 && a.exp == 0 && a.mant == 0)
            return b_0;
        if (b.sign == 0 && b.exp == 0 && b.mant == 0)
        {
            return b_infinity;
            // throw std::invalid_argument("EXCEPTION: division with 0");
            // exit(0);
        }

        // underflow: if number is too small return 0
        if ((int)res.exp < 0)
            return b_0;

        // 0. conversion to temps & shifting
        unsigned long long a_temp = (unsigned long long)a.mant;
        unsigned long long b_temp = (unsigned long long)b.mant;

        // std::cout << "\nstep 0: conversion to temps"
        //           << endl;
        // printBit_ulong(a_temp, true);
        // printBit_ulong(b_temp, true);
        // printBit_ulong((unsigned long long)res.exp, true);

        unsigned long long msb = (unsigned long long)1 << 63;
        long long shift_cnt = -(64 - FP_SIGNIFICAND_BITSIZE);

        while (!(a_temp & msb))
        {
            shift_cnt++;
            a_temp <<= 1;
        }
        res.exp = std::max((long long)0, res.exp - shift_cnt);

        // // unsigned long long a_temp = (unsigned long long)a.mant << (64 - FP_SIGNIFICAND_BITSIZE);

        // cout << "shift cnt: " << shift_cnt << endl;
        // printBit_ulong(a_temp, true);
        // printBit_ulong(b_temp, true);
        // printBit_ulong((unsigned long long)res.exp, true);

        // 1. divide mantissas
        unsigned long long res_mant_temp = (unsigned long long)a_temp / b_temp;

        // std::cout << "\nstep 1: division" << endl;
        // printBit_ulong(res_mant_temp, true);
        // printBit_ulong((unsigned long long)res.exp, true);

        // 2. normalization(implicit 1)
        if (res.exp)
        {
            unsigned long long implicit_1 = (unsigned long long)1 << (64 - FP_SIGNIFICAND_BITSIZE);
            bool has_implicit_1 = (bool)(res_mant_temp >> (64 - FP_SIGNIFICAND_BITSIZE));

            if (!has_implicit_1)
            {
                while (!(res_mant_temp & implicit_1))
                {
                    res_mant_temp <<= 1;
                    res.exp -= 1;
                }
            }

            if (a.exp == 0 ^ b.exp == 0)
                res.exp += 1;
        }
        else
        {
            res_mant_temp >>= 1;
            // res_mant_temp += (unsigned long long)1 << (64 - 1 - FP_MANT_BITSIZE - FP_MANT_BITSIZE);
        }

        // std::cout << "\nstep 2: normalization(implicit 1)" << endl;
        // printBit_ulong(res_mant_temp, true);
        // printBit_ulong((unsigned long long)res.exp, true);

        int carry = (int)(res_mant_temp >> (64 - FP_SIGNIFICAND_BITSIZE + 1));
        do
        {
            // 3. normalization(carry)
            while (carry)
            { // if there is carry
                res_mant_temp >>= 1;
                res.exp += 1;
                carry >>= 1;
            }

            // std::cout << "\nstep 4: normalization(carry)" << endl;
            // printBit_ulong(res_mant_temp, true);
            // printBit_ulong((unsigned long long)res.exp, true);

            // 4. rounding to nearest even
            int lsb_zero_cnt = 64 - 1 - FP_MANT_BITSIZE - FP_MANT_BITSIZE;
            unsigned long long t = (unsigned long long)1 << lsb_zero_cnt;
            unsigned short last_bit =
                (unsigned short)((res_mant_temp & t) >> lsb_zero_cnt);
            t >>= 1;
            unsigned short ground_bit =
                (unsigned short)((res_mant_temp & t) >> (lsb_zero_cnt - 1));
            t >>= 1;
            unsigned short round_bit =
                (unsigned short)((res_mant_temp & t) >> (lsb_zero_cnt - 2));
            t -= 1;
            unsigned short sticky_bits = (unsigned short)((bool)(res_mant_temp & t));

            unsigned long long lsb = (unsigned long long)1 << lsb_zero_cnt;
            if (ground_bit)
            {
                if (round_bit == 0 && sticky_bits == 0)
                {
                    if (last_bit)
                    {
                        res_mant_temp += lsb;
                    }
                }
                else
                {
                    res_mant_temp += lsb;
                }
            }

            // std::cout << "\nstep 4: rounding" << endl;
            // printBit_ulong(res_mant_temp, true);
            // printBit_ulong((unsigned long long)res.exp, true);
            // std::cout << last_bit << ground_bit << round_bit << sticky_bits << endl;

            carry = (int)(res_mant_temp >> (64 - FP_SIGNIFICAND_BITSIZE + 1));

        } while (carry);

        int lsb_zero_cnt = 64 - 1 - FP_MANT_BITSIZE - FP_MANT_BITSIZE;
        res.mant = (int)(res_mant_temp >> lsb_zero_cnt); // save result in res.mant

        return res;
    }

    fp sqrt(fp a) { return fpo_to_fp(std::sqrt(fp_to_fpo(a))); }

    fp sqrt(float a) { return fpo_to_fp(std::sqrt(a)); }

    fp pow(fp base, fp n)
    {
        return fpo_to_fp(std::pow(fp_to_fpo(base), fp_to_fpo(n)));
    }

    fp pow(fp base, float n)
    {
        return fpo_to_fp(std::pow(fp_to_fpo(base), n));
    }

    fp abs(fp a)
    {
        a.sign = 0;
        return a;
    }

    fp tan(fp a) { return fpo_to_fp(std::tan(fp_to_fpo(a))); }

    fp sin(fp a) { return fpo_to_fp(std::sin(fp_to_fpo(a))); }

    fp cos(fp a) { return fpo_to_fp(std::cos(fp_to_fpo(a))); }

    bool compare(fp a, fp b)
    {
        return fp_to_fpo(a) > fp_to_fpo(b);
    }

    bool isequal(fp a, fp b)
    {
        return fp_to_fpo(a) == fp_to_fpo(b);
    }

    inline fp operator+(fp a, fp b) { return add(a, b); }

    inline fp operator-(fp a, fp b) { return sub(a, b); }

    inline fp operator-(fp a)
    {
        if (a.exp == 0 && a.mant == 0)
            return a;
        a.sign ^= 1;
        return a;
    }

    inline fp operator*(fp a, fp b) { return mult(a, b); }

    inline fp operator/(fp a, fp b) { return div(a, b); }

    inline fp min(fp a, fp b)
    {
        float a_f = fp_to_fpo(a);
        float b_f = fp_to_fpo(b);
        float min = std::fmin(a_f, b_f);
        return fpo_to_fp(min);
        // /* align exponents */
        // if (a.exp >= b.exp)
        // {
        //     b.mant >>= (a.exp - b.exp);
        // }
        // else
        // {
        //     a.mant >>= (b.exp - a.exp);
        // }

        // /* a > b */
        // if (a.sign ^ b.sign)
        // { // 둘 중 하나만 음수인 경우
        //     if (a.sign)
        //         return a; // a가 음수인 경우
        //     else
        //         return b; // b가 음수인 경우
        // }
        // else if (a.sign && b.sign)
        // { // 둘 다 음수인 경우
        //     if (a.mant < b.mant)
        //         return b;
        //     else
        //         return a;
        // }
        // else
        { // 둘 다 양수인 경우
            if (a.mant > b.mant)
                return b;
            else
                return a;
        }
    }

    inline fp max(fp a, fp b)
    {
        float a_f = fp_to_fpo(a);
        float b_f = fp_to_fpo(b);
        float max = std::fmax(a_f, b_f);
        return fpo_to_fp(max);
    }

    inline bool operator>(fp a, fp b) { return compare(a, b); }

    inline bool operator>=(fp a, fp b)
    {
        return compare(a, b) || isequal(a, b);
    }

    inline bool operator<(fp a, fp b) { return compare(b, a); }

    inline bool operator<=(fp a, fp b)
    {
        return compare(b, a) || isequal(a, b);
    }

    inline bool operator==(fp a, fp b) { return isequal(a, b); }

    inline bool operator!=(fp a, fp b)
    {
        return (a.sign != b.sign) || (a.exp != b.exp) || (a.mant != b.mant);
    }

    //--------------------------------------
    inline bool operator>(__fpo a, fp b) { return a > fp_to_fpo(b); }

    inline bool operator>=(__fpo a, fp b) { return a >= fp_to_fpo(b); }

    inline bool operator<(__fpo a, fp b) { return a < fp_to_fpo(b); }

    inline bool operator<=(__fpo a, fp b) { return a <= fp_to_fpo(b); }

    inline bool operator==(__fpo a, fp b) { return a == fp_to_fpo(b); }

    inline bool operator!=(__fpo a, fp b) { return a != fp_to_fpo(b); }

    //--------------------------------------
    inline bool operator>(fp a, __fpo b) { return fp_to_fpo(a) > b; }

    inline bool operator>=(fp a, __fpo b) { return fp_to_fpo(a) >= b; }

    inline bool operator<(fp a, __fpo b) { return fp_to_fpo(a) < b; }

    inline bool operator<=(fp a, __fpo b) { return fp_to_fpo(a) <= b; }

    inline bool operator==(fp a, __fpo b) { return fp_to_fpo(a) == b; }

    inline bool operator!=(fp a, __fpo b) { return fp_to_fpo(a) != b; }
} // namespace fp
