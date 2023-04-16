#pragma once
#include <vector>

#define FLOAT_LEN_BITSIZE 32
#define FLOAT_EXP_BITSIZE 8
#define FLOAT_MANT_BITSIZE 23
#define FLOAT_BIAS 127
#define FLOAT_MANT_EXTRACT_BITS 0x007FFFFF
#define FLOAT_EXP_EXTRACT_BITS 0x000000FF

#define DOUBLE_EXP_BITSIZE 11
#define DOUBLE_MANT_BITSIZE 52
#define DOUBLE_BIAS 1023
#define DOUBLE_MANT_EXTRACT_BITS 0x000FFFFFFFFFFFFF
#define DOUBLE_EXP_EXTRACT_BITS 0x00000000000007FF

#define MAX_BITSIZE 100
#define DEBUG 0
#define TEST_BIT 0
#define SIMPLE_TEST 1
#define NO_RANDOM 0

/* set bit configuration */
#define FP_MANT_BITSIZE 23
#define FP_SIGNIFICAND_BITSIZE 24
#define FP_EXP_BITSIZE 8
#define FP_BIAS (unsigned int)(std::pow(2, FP_EXP_BITSIZE - 1) - 1)

typedef struct
{
    unsigned short sign;
    unsigned int exp;
    long long int mant;
} fp;

/* set original data type*/
typedef double __fpo; // short for 'floating point original'
