#pragma once
#include <string.h>
#include <iostream>

#include "fp_struct.h"

void print_fp(fp b);
void printBit_fp(fp b, bool nextLine);
void printBit_fp_exp(unsigned int e, bool nextLine);
void printBit_fp_mant(int m, bool nextLine);
void printBit_float_mant(int m, bool newLine);
void printBit_uint(unsigned int num, int len);
void printBit_float(float f);
void printBit_sint(int num, bool newLine);
void printBit_ulong(long long num, bool newLine);

using namespace std;

/* print structure */
void print_fp(fp b)
{
    cout << "---------fp component -------------" << endl;
    cout << "sign: " << b.sign << endl;
    cout << "exp: " << b.exp << endl;
    cout << "mant: " << b.mant << endl;
}

/* print bit representations of fp struct */
void printBit_fp(fp b, bool nextLine)
{
    printBit_uint(b.sign, 1);
    cout << " ";
    printBit_uint(b.exp, FP_EXP_BITSIZE);
    cout << " ";
    printBit_uint(b.mant, FP_MANT_BITSIZE);

    if (nextLine)
    {
        cout << endl;
    }
}

void printBit_fp_exp(unsigned int e, bool nextLine)
{
    cout << e << " => ";
    printBit_uint(e, FP_EXP_BITSIZE);

    if (nextLine)
    {
        cout << endl;
    }
}

void printBit_fp_mant(int m, bool nextLine)
{
    int temp = m;
    vector<string> out;
    for (int i = 0; i < FP_MANT_BITSIZE + 1; i++, temp >>= 1)
    {
        if (i % 4 == 0)
        {
            out.insert(out.begin(), " ");
        }
        if (temp & 1)
        {
            out.insert(out.begin(), "1");
        }
        else
        {
            out.insert(out.begin(), "0");
        }
    }

    for (const auto &bit : out)
    {
        cout << bit;
    }

    if (nextLine)
    {
        cout << "\n";
    }
}

void printBit_float_mant(int m, bool newLine)
{
    int temp = m;
    char out[MAX_BITSIZE] = "";
    for (int i = 0; i < 24; i++, temp >>= 1)
    {
        if (i % 4 == 0)
        {
            strcat(out, " ");
        }
        if (temp & 1)
        {
            strcat(out, "1");
        }
        else
        {
            strcat(out, "0");
        }
    }

    for (int i = 29; i >= 0; i--)
    {
        cout << out[i];
    }
    printf(" (0x%.8x)", m);

    if (newLine)
    {
        cout << "\n";
    }
}

/* print bit representations of basic data types */
void printBit_uint(unsigned int num, int len)
{
    char out[MAX_BITSIZE] = "";

    for (int i = 0; i < len; i++, num >>= 1)
    {
        if (num & 1)
        {
            strcat(out, "1");
        }
        else
        {
            strcat(out, "0");
        }
    }

    for (int i = len - 1; i >= 0; i--)
    {
        cout << out[i];
    }
}

void printBit_float(float f)
{
    cout << f << "   =>   ";
    // cast to integer for bitwise operations
    unsigned int *temp = reinterpret_cast<unsigned int *>(&f);
    unsigned int orginal_val = *temp;

    char out[MAX_BITSIZE] = "";
    for (int i = 0; i < 32; i++, *temp >>= 1)
    {
        if (i == 23 || i == 31)
        {
            strcat(out, " ");
        }
        if (*temp & 1)
        {
            strcat(out, "1");
        }
        else
        {
            strcat(out, "0");
        }
    }

    for (int i = 35; i >= 0; i--)
    {
        cout << out[i];
    }
    printf(" (0x%.8x)\n", orginal_val);
}

void printBit_double(double f)
{
    cout << f << "   =>   ";
    // cast to integer for bitwise operations
    unsigned long long *temp = reinterpret_cast<unsigned long long *>(&f);
    unsigned long long orginal_val = *temp;


    char out[MAX_BITSIZE] = "";
    for (int i = 0; i < 64; i++, *temp >>= 1)
    {
        if (i == 52 || i == 63)
        {
            strcat(out, " ");
        }
        if (*temp & 1)
        {
            strcat(out, "1");
        }
        else
        {
            strcat(out, "0");
        }
    }

    for (int i = 79; i >= 0; i--)
    {
        cout << out[i];
    }
    printf(" (0x%.8x)\n", orginal_val);
}

void printBit_sint(int num, bool newLine)
{ // 4bits grouped together
    int temp = num;
    char out[MAX_BITSIZE] = "";
    for (int i = 0; i < 32; i++, temp >>= 1)
    {
        if (i % 4 == 0)
        {
            strcat(out, " ");
        }
        if (temp & 1)
        {
            strcat(out, "1");
        }
        else
        {
            strcat(out, "0");
        }
    }

    for (int i = 39; i >= 0; i--)
    {
        cout << out[i];
    }
    printf(" (0x%x)", num);

    if (newLine)
    {
        cout << endl;
    }
}

void printBit_ulong(long long num, bool newLine)
{ // 4bits grouped together
    long long temp = num;
    char out[MAX_BITSIZE] = "";
    for (int i = 0; i < 64; i++, temp >>= 1)
    {
        if (i % 4 == 0)
        {
            strcat(out, " ");
        }
        if (temp & 1)
        {
            strcat(out, "1");
        }
        else
        {
            strcat(out, "0");
        }
    }

    for (int i = 79; i >= 0; i--)
    {
        std::cout << out[i];
    }

    printf("\t(%llx)", num);

    if (newLine)
    {
        std::cout << endl;
    }
}
