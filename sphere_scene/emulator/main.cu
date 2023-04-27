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
#include "bvh.h"

#include "mkPpm.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"

#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <bits/stdc++.h>

#define MAX_SIZE 500
static int tmp_count;

unsigned char *image_array;

// 1. random_scene: Implements the 3D World.
hittable_list random_scene()
{
	hittable_list world;
	int count = 0;

	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(++count, point3(0.0, -1000.0, 0.0), 1000, ground_material));

	if (NO_RANDOM)
	{
		auto material3 = make_shared<metal>(color(b_0_7, b_0_6, b_0_5), b_0);
		world.add(make_shared<sphere>(++count, point3(b_0, b_1000_neg, b_0), b_1000, material3));
		world.add(make_shared<sphere>(++count, point3(b_0, b_1, b_0), b_1, material3));
		world.add(make_shared<sphere>(++count, point3(-b_4, b_1, b_0), b_1, material3));
		world.add(make_shared<sphere>(++count, point3(b_4, b_1, b_0), b_1, material3));
	}
	else
	{
		auto material1 = make_shared<dielectric>(b_1_5);
		world.add(make_shared<sphere>(++count, point3(0.0, 1.0, 0.0), 1.0, material1));

		auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
		world.add(make_shared<sphere>(++count, point3(-4.0, 1.0, 0.0), 1.0, material2));

		auto material3 = make_shared<metal>(color(0.7, 0.6, 0.6), 0.0);
		world.add(make_shared<sphere>(++count, point3(4, 1, 0), 1, material3));
	}

	// Constructing BVH
	hittable_list world_bvh;
	world_bvh.add(make_shared<bvh_node>(world, 0, 1));

	return world_bvh;
}

// 2. ray_color: calculates color of the current ray intersection point.
color ray_color(const ray &r, const hittable &world, int depth)
{
	hit_record rec;

	if (depth <= 0)
		return color(0, 0, 0);

	if (world.hit(r, b_0_1, b_infinity, rec)) // hit: 충돌지점을 결정(child ray 의 origin)
	{										  // if ray hits object
		ray scattered;
		color attenuation;
		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) // scatter: child ray 방향성을 결정
		{
			if (DEBUG)
			{
				tmp_count++;
				cout << "  - Ray #" << tmp_count << ": O=(" << scattered.origin() << "), D=(" << scattered.direction() << ")" << endl;
				cout << "    <Attenuation>: " << attenuation << endl;
			}

			return attenuation * ray_color(scattered, world, depth - 1);
		}
		return color(b_0, b_0, b_0);
	}

	// if ray doesn;t hit any object: background
	vec3 unit_direction = unit_vector(r.direction());
	fp t = b_0_5 * (unit_direction.y() + b_1);
	return (b_1 - t) * color(1, 1, 1) + t * color(0.5, 0.7, 1);
}

// 3. main
int main()
{
	if (DEBUG)
	{
		cout << "EXPONENT: " << FP_EXP_BITSIZE << ", MANTISSA: " << FP_MANT_BITSIZE << endl;
	}
	// Measure the execution time.
	mkClockMeasure *ckCpu = new mkClockMeasure("TOTAL TIME");
	mkClockMeasure *ckWolrdBVH = new mkClockMeasure("CONSTRUCT WORLD & BVH");
	mkClockMeasure *ckRendering = new mkClockMeasure("RENDERING");
	ckCpu->clockReset();
	ckWolrdBVH->clockReset();
	ckRendering->clockReset();

	ckCpu->clockResume();
	// Image
	auto aspect_ratio = 16.0 / 9.0;
	int image_width = 400; // 400
	int image_height = static_cast<int>(image_width / aspect_ratio);
	int samples_per_pixel = 1;
	const int max_depth = 50;
	float scale = 1.0 / samples_per_pixel;

	fp _image_height = int_to_fp(image_height);
	fp _image_width = int_to_fp(image_width);
	fp _samples_per_pixel = int_to_fp(samples_per_pixel);

	// World
	ckWolrdBVH->clockResume();

	if (DEBUG)
	{
		cout << "CREATE WORLD" << endl;
	}
	hittable_list world = random_scene();

	if (DEBUG)
	{
		cout << "WORLD CREATED" << endl;
	}
	ckWolrdBVH->clockPause();

	// Camera
	point3 lookfrom(b_13, b_2, b_3);
	point3 lookat(b_0, b_0, b_0);
	vec3 vup(b_0, b_1, b_0);
	fp dist_to_focus = b_10;
	fp aperture = b_0_1;

	camera cam(lookfrom, lookat, vup, b_20, fpo_to_fp(aspect_ratio), aperture, dist_to_focus, b_0, b_1);

	if (DEBUG)
	{
		cout << "CAMERA CREATED" << endl;
	}
	// Rendered Image image_array
	image_array = (unsigned char *)malloc(sizeof(unsigned char) * image_width * image_height * 3);

	// Render
	ckRendering->clockResume();
	__fpo r, g, b;
	for (int j = 0; j < image_height; ++j)
	{
		for (int i = 0; i < image_width; ++i)
		{

			fp _i = int_to_fp(i);
			fp _j = int_to_fp(j);

			int idx = (j * image_width + i) * 3;
			color pixel_color(b_0, b_0, b_0);

			/* float version */
			for (int s = 0; s < samples_per_pixel; ++s)
			{
				__fpo u = (i + random_fpo()) / (image_width - 1);
				__fpo v = ((image_height - j - 1) + random_fpo()) / (image_height - 1);

				ray cur_ray = cam.get_ray(u, v);

				if (DEBUG)
				{
					tmp_count = 0;
					cout << "Pixel (" << j << "," << i << ")" << endl;
					cout << "[Sample " << s << "]" << endl;
					cout << "  - Ray #" << tmp_count << ": O=(" << cur_ray.origin() << "), D=(" << cur_ray.direction() << ")" << endl;
				}

				pixel_color += ray_color(cur_ray, world, max_depth);

				if (DEBUG)
				{
					cout << "=> Color = (" << pixel_color << ")" << endl
						 << endl;
				}
			}

			r = fp_to_fpo(pixel_color.x());
			g = fp_to_fpo(pixel_color.y());
			b = fp_to_fpo(pixel_color.z());

			r = std::sqrt(scale * r);
			g = std::sqrt(scale * g);
			b = std::sqrt(scale * b);

			image_array[idx] = (256 * clamp(r, 0.0, 0.999));
			image_array[idx + 1] = (256 * clamp(g, 0.0, 0.999));
			image_array[idx + 2] = (256 * clamp(b, 0.0, 0.999));
		}
	}
	ckRendering->clockPause();
	ckCpu->clockPause();

	/* Print clocks */
	ckCpu->clockPrint();
	ckWolrdBVH->clockPrint();
	ckRendering->clockPrint();

	/* Save image */
	if (DEBUG || SIMPLE_TEST)
	{
		ppmSave("img.ppm", image_array, image_width, image_height);
		return 0;
	}

	time_t t = time(NULL);
	tm *tPtr = localtime(&t);
	int timeStamp = (((tPtr->tm_year) + 1900) % 100) * 10000 + ((tPtr->tm_mon) + 1) * 100 + (tPtr->tm_mday);

	// creating directory
	string fp_size = to_string(FP_EXP_BITSIZE) + "_" + to_string(FP_MANT_BITSIZE);
	string directory = "../images/emulator/" + string(typeid(__fpo).name()) + "/";
	if (mkdir(directory.c_str(), 0777) == -1)
		cerr << "Error :  " << strerror(errno) << endl;

	string img_path = directory + "/" + to_string(timeStamp) + "_" + fp_size + "_" + to_string(image_width) + "_" + to_string(samples_per_pixel) + "_" + to_string(max_depth) + "img.ppm";
	ppmSave(img_path.c_str(), image_array, image_width, image_height);

	return 0;
}
