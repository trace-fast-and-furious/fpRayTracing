#include <stdio.h>
#include <iostream>
#include <memory>
#include <limits>
#include <string.h>
#include <algorithm>
#include <vector>
#include <string>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <bits/stdc++.h>

#include "./lib/cpfloat_binary64.cuh"

using std::cout;
using std::endl;
using std::string;
using std::to_string;

#define DEBUG 0
#define DATE 0

/* set data structures and names */
typedef double fp_orig;

struct fp_custom
{
    double val[1];
};

struct e_custom
{
    fp_custom val[3];
};

/* precision settings */
#define CP_EXP_BITSIZE 8
#define CP_MANT_BITSIZE 23

// __device__ optstruct *fpopts = init_optstruct();
__device__ optstruct *fpopts;                  
