/*
 * ===================================================
 *
 *       Filename:  main.cu
 *    Description:  Ray Tracing In One Weekend (RTIOW): ~BVH 
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
#include "bvh.h"

#include "mkPpm.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"

#include <iostream>

#define MAX_SIZE 500

unsigned char *array;

// Functions


// 1. random_scene: Implements the 3D World.
hittable_list random_scene(int n) {
    	hittable_list world;
	int count = 0;
    
	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	//auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.0));

    	world.add(make_shared<sphere>(++count, point3(0,-1000,0), 1000, ground_material));

    	for (int a = -n; a < n; a++) {
		for (int b = -n; b < n; b++) {

			// Generate constant scene primitives.
			float choose_mat = (a * 11 + b)/121;

			// Generate random scene primitives.
			//auto choose_mat = random_float();
	    		//point3 center(a + 0.9*random_float(), 0.2, b + 0.9*random_float());
	    		point3 center(a, 0.2, b);

	    		if ((center - point3(4, 0.2, 0)).length() > 0.9) {
			    	shared_ptr<material> sphere_material;

				if (choose_mat < 0.8) {
    		
					// diffuse
		    			auto albedo = color::random() * color::random();
		    			sphere_material = make_shared<lambertian>(albedo);
					auto center2 = center + vec3(0, random_float(0,.5), 0);
		    			world.add(make_shared<sphere>(
					   	++count, center, 0.2, sphere_material));
				} else if (choose_mat < 0.95) {
		    			// metal
		    			auto albedo = color::random(0.5, 1);
		    			auto fuzz = random_float(0, 0.5);
		    			sphere_material = make_shared<metal>(albedo, fuzz);
		    			world.add(make_shared<sphere>(++count, center, 0.2, sphere_material));
				} else {
		    			// glass
		    			sphere_material = make_shared<dielectric>(1.5);
		    			world.add(make_shared<sphere>(++count, center, 0.2, sphere_material));
				}
	    		}
		}
    	}

	auto material1 = make_shared<dielectric>(0.5);
	  world.add(make_shared<sphere>(++count, point3(0, 1, 0), 1.0, material1));

    	//world.add(make_shared<sphere>(++count, point3(3, 1, -0.5), 1.0, material1));

    	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    	world.add(make_shared<sphere>(++count, point3(-4, 1, 0), 1.0, material2));

    	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    	world.add(make_shared<sphere>(++count, point3(4, 1, 0), 1.0, material3));
    //	world.add(make_shared<sphere>(++count, point3(3, 1, 2), 1.0, material3));

		auto material4 = make_shared<metal>(color(0.5, 0.7, 0.5), 0.1);
	//	world.add(make_shared<sphere>(++count, point3(3, 1, -0.5), 1.0, material4));


		auto material5 = make_shared<lambertian>(color(1.0, 0.0, 0.6));
	//	world.add(make_shared<sphere>(++count, point3(5, 0.5, 0), 0.5, material5));
		
		auto material6 = make_shared<dielectric>(0.5);
	//	world.add(make_shared<sphere>(++count, point3(5, 0.3, 1.3), 0.3, material6));

//	return world;
	
	// Constructing BVH
	hittable_list world_bvh;
	world_bvh.add(make_shared<bvh_node>(world, 0, 1));
	printf("\n\n================================== BVH CONSTURCTION COMPLETED ==================================\n\n\n");


	return world_bvh;
}



// 2. ray_color: calculates color of the current ray intersection point.
color ray_color(const ray& r, const hittable& world, int depth) {
    
	hit_record rec;



		// RT18: CHECK THE BACKGROUND COLOR
		//return vec3(0.5, 0.7, 1.0);


 
     	// Limit the number of child ray.
       	if (depth <= 0)
	       	return color(0, 0, 0);  // If the ray hits objects more than 'depth' times, consider that no light approaches the current point.

    	// If the ray hits an object: Hittable Object
    	if (world.hit(r, 0.001, infinity, rec)) {
		
		ray scattered;
		color attenuation;
		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
	    		return attenuation * ray_color(scattered, world, depth-1);
		return color(0,0,0);
	}

    	// If the ray hits no object: Background
    	vec3 unit_direction = unit_vector(r.direction());
    	auto t = 0.5 * (unit_direction.y() + 1.0);
    	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
    	//return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.7, 0.7, 1.0);

}



// 3. main
int main() {

	// Measure the execution time.
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	ckCpu->clockReset();


    	// Image
	auto aspect_ratio = 16.0 / 9.0;
    	int image_width = 100;  //400
       	int samples_per_pixel = 5;    
	const int max_depth = 5;

	// Objects
	int n = 0;
	int object_num = (n+n)*(n+n)+4;

	
	ckCpu->clockResume();
    	// World
	hittable_list world = random_scene(n);

    	ckCpu->clockPause();
    	ckCpu->clockPrint("Creat World");


    	// Camera
	point3 lookfrom(13,2,3);
    	point3 lookat(0,0,0);
    	vec3 vup(0,1,0);
    	auto dist_to_focus = 20.0;
    	auto aperture = 0.1;
	int image_height = static_cast<int>(image_width / aspect_ratio);
    	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);


	// Rendered Image Array
	array = (unsigned char *)malloc(sizeof(unsigned char) * image_width * image_height * 3);


	ckCpu->clockReset();
	ckCpu->clockResume();


	// Render
	float r, g, b;

	// RT18
	//PRINT PIXEL VALUES OF THE OUTPUT IMAGE: printf("------------------- IMAGE -------------------\n");

	for (int j = 0; j < image_height; ++j) {
	   	for (int i = 0; i < image_width; ++i) {
	       		int idx = (j * image_width + i) * 3;
		  	color pixel_color(0, 0, 0);

				for (int s = 0; s < samples_per_pixel; ++s) {
			      	auto u = (i + random_float()) / (image_width - 1);
				auto v = ((image_height-j-1) + random_float()) / (image_height - 1);

				ray cur_ray = cam.get_ray(u, v);


				// RT17: FOR DEBUGGING
				/*
				printf("(RENDER) Pixel (%lf, %lf): Ray Direction = (%lf, %lf, %lf)\n\n", 
				u, v, 
				(cur_ray.direction()).e[0], (cur_ray.direction()).e[1], (cur_ray.direction()).e[2]);
				*/

				pixel_color += ray_color(cur_ray, world, max_depth);

				r = pixel_color.x();
				g = pixel_color.y();
				b = pixel_color.z();

				// Antialiasing
				float scale = 1.0 / samples_per_pixel;
				r = sqrt(scale * r);
				g = sqrt(scale * g);
				b = sqrt(scale * b);
				
				
//				printf("[%dx%d s:%d] %lf %lf %lf\n", j, i, s, pixel_color[0], pixel_color[1], pixel_color[2]);
				//printf("[%d] %f %f %f\n", s, pixel_color[0], pixel_color[1], pixel_color[2]);
			}
            
	    		array[idx] = (256 * clamp(r, 0.0, 0.999));
	    		array[idx+1] = (256 * clamp(g, 0.0, 0.999));
	    		array[idx+2] = (256 * clamp(b, 0.0, 0.999));

				// RT18 - PRINT PIXEL VALUES OF THE OUTPUT IMAGE:
				//printf("  R:%d, G:%d, B:%d\n", array[idx], array[idx+1], array[idx+2]);

    		}
    	}
		// RT18 - PRINT PIXEL VALUES OF THE OUTPUT IMAGE: 
		//printf("---------------------------------------------\n");


    	ckCpu->clockPause();
    	ckCpu->clockPrint("Rendering");

    	ppmSave("img.ppm", array, image_width, image_height, object_num, samples_per_pixel);

    	return 0;
}
