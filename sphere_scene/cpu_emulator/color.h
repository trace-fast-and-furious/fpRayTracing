#pragma once
#ifndef COLOR_BFP_H
#define COLOR_BFP_H

#include "vec3.h"

using namespace floating_point;

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel)
{
    fp r = pixel_color.x();
    fp g = pixel_color.y();
    fp b = pixel_color.z();

    fp scale = fpo_to_fp(1.0 / samples_per_pixel);
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    out << fp_to_int(b_256 * clamp(r, b_0, b_0_999)) << ' '
        << fp_to_int(b_256 * clamp(g, b_0, b_0_999)) << ' '
        << fp_to_int(b_256 * clamp(b, b_0, b_0_999)) << '\n';
}

void print_color(color c){
    cout << "======== COLOR=======" << endl;
    cout << "r: " << fp_to_fpo(c[0]) << "\t=>\t" << c[0].sign << " " << c[0].exp << " " << c[0].mant << endl;
    cout << "g: " << fp_to_fpo(c[1]) << "\t=>\t" << c[1].sign << " " << c[1].exp << " " << c[1].mant << endl;
    cout << "b: " << fp_to_fpo(c[2]) << "\t=>\t" << c[2].sign << " " << c[2].exp << " " << c[2].mant << endl;
}

#endif
