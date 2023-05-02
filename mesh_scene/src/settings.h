#include <stdio.h>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <algorithm>
#include <vector>
#include <string>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <cassert>


#include "./lib/cpfloat_binary64.h"

using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
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
#define CP_EXP_BITSIZE 11
#define CP_MANT_BITSIZE 52

optstruct *fpopts = init_optstruct();
