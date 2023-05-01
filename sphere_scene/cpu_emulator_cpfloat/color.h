#pragma once
#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

using namespace custom_precision_fp;

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel)
{
    fp_custom r = pixel_color.x();
    fp_custom g = pixel_color.y();
    fp_custom b = pixel_color.z();

    fp_custom scale = fp_orig_to_custom(1.0 / samples_per_pixel);
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    out << 256 * clamp(r, 0, 0.999) << ' '
        << 256 * clamp(g, 0, 0.999) << ' '
        << 256 * clamp(b, 0, 0.999) << '\n';
}

// void print_color(color c){
//     cout << "======== COLOR=======" << endl;
//     cout << "r: " << fp_to_fpo(c[0]) << "\t=>\t" << c[0].sign << " " << c[0].exp << " " << c[0].mant << endl;
//     cout << "g: " << fp_to_fpo(c[1]) << "\t=>\t" << c[1].sign << " " << c[1].exp << " " << c[1].mant << endl;
//     cout << "b: " << fp_to_fpo(c[2]) << "\t=>\t" << c[2].sign << " " << c[2].exp << " " << c[2].mant << endl;
// }

#endif
