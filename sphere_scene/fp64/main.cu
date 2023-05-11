/*
 * ===================================================
 *
 *       Filename:  main.cu
 *    Description:  Ray Tracing In One Weekend (RTIOW): Final Code 
 *        Created:  2022/07/13
 * 
 * ===================================================
 */


// Preprocessors

#include "moving_sphere.h"
#include "material.h"
#include "utility.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#include "mkPpm.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"

#include <iostream>

using namespace std;

#define DEBUG 0
static int tmp_count;


unsigned char *array_img;


// Functions

// 1. random_scene(): Implements the 3D World.


hittable_list random_scene() {
	hittable_list world;

/*
	auto met = make_shared<metal>(color(0.7, 0.6, 0.5), 0.1);
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, met));
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, met));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, met));
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, met));
*/

   


/*
	auto ground_material = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

	auto material1 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(0, 2, 0), 1, material1));
*/

	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

	int n = 5;
	for (int a = -n; a < n; a++) {
		for (int b = -n; b < n; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				shared_ptr<material> sphere_material;

				if (choose_mat < 0.8) {
			
					// diffuse
						auto albedo = color::random() * color::random();
						sphere_material = make_shared<lambertian>(albedo);
						//auto center2 = center + vec3(0, random_double(0,.5), 0);
						//world.add(make_shared<moving_sphere>(center, center2, 0.0, 1.0, 0.2, sphere_material));
						world.add(make_shared<sphere>(center, 0.2, sphere_material));
				} else if (choose_mat < 0.95) {
						// metal
						auto albedo = color::random(0.5, 1);
						auto fuzz = random_double(0, 0.5);
						sphere_material = make_shared<metal>(albedo, fuzz);
						world.add(make_shared<sphere>(center, 0.2, sphere_material));
				} else {
						// glass
						sphere_material = make_shared<dielectric>(1.5);
						world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}
	
	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	

	return world;
}



// ray_color: calculates color of the current ray intersection point.
color ray_color(const ray& r, const hittable& world, int depth) {
    
	hit_record rec;
 
     	// Limit the number of child ray.
       	if (depth <= 0)
	       	return color(0, 0, 0);  // If the ray hits objects more than 'depth' times, consider that no light approaches the current point.

    	// If the ray hits an object: Hittable Object
    	if (world.hit(r, 0.001, infinity, rec)) {
		ray scattered;
		color attenuation;
		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {

			if(DEBUG)
			{
				tmp_count++;
				cout << "  - Ray #" << tmp_count << ": O=(" << scattered.origin() << "), D=(" << scattered.direction() << ")" << endl; 
				cout << "    <Attenuation>: " << attenuation << endl;

			}

			return attenuation * ray_color(scattered, world, depth-1);
		}
		return color(0,0,0);
	}

    	// If the ray hits no object: Background
    	vec3 unit_direction = unit_vector(r.direction());
    	auto t = 0.5 * (unit_direction.y() + 1.0);
    	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}



int main() {

	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	ckCpu->clockReset();


    	// Image

	double aspect_ratio = 16.0 / 9.0;
	int image_width = 500;
	int samples_per_pixel = 3;    
	const int max_depth = 50;


    	// World

	hittable_list world = random_scene();


    	// Camera

	point3 lookfrom(13,2,3);
    	point3 lookat(0,0,0);
    	vec3 vup(0,1,0);
    	double dist_to_focus = 10.0;
    	double aperture = 0.1;
	int image_height = static_cast<int>(image_width / aspect_ratio);
    	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);


	// Rendered Image Array
	array_img = (unsigned char *)malloc(sizeof(unsigned char) * image_width * image_height * 3);


	// Measure the rendering time.
	ckCpu->clockResume();


	// Render

	double r, g, b;

	for (int j = 0; j < image_height; ++j) {
	   	for (int i = 0; i < image_width; ++i) {
			int idx = (j * image_width + i) * 3;
		  	color pixel_color(0, 0, 0);

			for (int s = 0; s < samples_per_pixel; ++s) {
				//double u = (i + random_double()) / (image_width - 1);
				//double v = ((image_height-j-1) + random_double()) / (image_height - 1);

				double u = (double)i / (image_width - 1);
				double v = (double)(image_height-j-1) / (image_height - 1);

				cout << u << endl;
				cout << v << endl;

				ray cur_ray = cam.get_ray(u, v);

				if(DEBUG) 
				{
					tmp_count = 0;
					cout << "Pixel (" << j << "," << i << ")" << endl;
					cout << "[Sample " << s << "]" <<  endl;
					cout << "  - Ray #" << tmp_count << ": O=(" << cur_ray.origin() << "), D=(" << cur_ray.direction() << ")" << endl; 
				}


				pixel_color += ray_color(cur_ray, world, max_depth);

				if(DEBUG) 
				{
					cout << "=> Color = (" << pixel_color << ")" << endl << endl; 
				}

				r = pixel_color.x();
				g = pixel_color.y();
				b = pixel_color.z();

				// Antialiasing
				double scale = 1.0 / samples_per_pixel;
				r = sqrt(scale * r);
				g = sqrt(scale * g);
				b = sqrt(scale * b);
			}
            
	    		array_img[idx] = (256 * clamp(r, 0.0, 0.999));
	    		array_img[idx+1] = (256 * clamp(g, 0.0, 0.999));
	    		array_img[idx+2] = (256 * clamp(b, 0.0, 0.999));

				if(DEBUG)
				{
					unsigned int tmp_c = 255;
					if(i == 0) {
						if(j % 2 == 0)
						{
							array_img[idx] = tmp_c;
							array_img[idx+1] = tmp_c;
							array_img[idx+2] = tmp_c;
						}
						if(j % 5 == 0)
						{
							array_img[idx] = 0;
							array_img[idx+1] = 255;
							array_img[idx+2] = 0;
						}
					}
				}
    		}
    	}

    	ckCpu->clockPause();
    	ckCpu->clockPrint();

    	ppmSave("fp64_A0p001.ppm", array_img, image_width, image_height);

    	return 0;
}
