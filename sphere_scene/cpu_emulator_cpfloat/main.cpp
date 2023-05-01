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

	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(++count, point3(0.0, 1.0, 0.0), 1.0, material1));

	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(++count, point3(-4.0, 1.0, 0.0), 1.0, material2));

	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.6), 0.0);
	world.add(make_shared<sphere>(++count, point3(4, 1, 0), 1, material3));

	// auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0);
	// world.add(make_shared<sphere>(++count, point3(0, -1000, 0), 1000, material3));
	// world.add(make_shared<sphere>(++count, point3(0, 1, 0), 1, material3));
	// world.add(make_shared<sphere>(++count, point3(-4, 1, 0), 1, material3));
	// world.add(make_shared<sphere>(++count, point3(4, 1, 0), 1, material3));

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

	if (world.hit(r, 0.1, std::numeric_limits<fp_orig>::infinity(), rec)) // hit: 충돌지점을 결정(child ray 의 origin)
	{																	  // if ray hits object
		ray scattered;
		color attenuation;
		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) // scatter: child ray 방향성을 결정
		{
			if (DEBUG)
			{
				tmp_count++;
				std::cout << "  - Ray #" << tmp_count << ": O=(" << scattered.origin() << "), D=(" << scattered.direction() << ")" << endl;
				std::cout << "    <Attenuation>: " << attenuation << endl;
			}

			return attenuation * ray_color(scattered, world, depth - 1);
		}
		return color(0, 0, 0);
	}

	// if ray doesn;t hit any object: background
	vec3 unit_direction = unit_vector(r.direction());
	fp_custom t = 0.5 * (unit_direction.y() + 1);
	return (1 - t) * color(1, 1, 1) + t * color(0.5, 0.7, 1);
}

// 3. main
int main()
{

	// Allocate the data structure for target formats and rounding parameters.
	fpopts = init_optstruct();

	// Set up the parameters for binary16 target format.
	fpopts->precision = CP_MANT_BITSIZE + 1;	   // Bits in the significand + 1.
	fpopts->emax = pow(CP_EXP_BITSIZE - 1, 2) - 1; // The maximum exponent value(=bias)
	fpopts->subnormal = CPFLOAT_SUBN_USE;		   // Support for subnormals is on.
	fpopts->round = CPFLOAT_RND_NE;				   // Round toward +infinity.
	fpopts->flip = CPFLOAT_NO_SOFTERR;			   // Bit flips are off.
	fpopts->p = 0;								   // Bit flip probability (not used).
	fpopts->explim = CPFLOAT_EXPRANGE_TARG;		   // Limited exponent in target format.

	// Validate the parameters in fpopts.
	int retval = cpfloat_validate_optstruct(fpopts);
	printf("The validation function returned %d.\n", retval);
	cout << "EXPONENT: " << CP_EXP_BITSIZE << ", MANTISSA: " << CP_MANT_BITSIZE << endl;

	std::cout.precision(std::numeric_limits<fp_orig>::max_digits10 - 1);

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
	int samples_per_pixel = 1000;
	const int max_depth = 50;
	float scale = 1.0 / samples_per_pixel;

	// World
	ckWolrdBVH->clockResume();

	if (DEBUG)
	{
		std::cout << "CREATE WORLD" << endl;
	}
	hittable_list world = random_scene();

	if (DEBUG)
	{
		std::cout << "WORLD CREATED" << endl;
	}
	ckWolrdBVH->clockPause();

	// Camera
	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	fp_orig dist_to_focus = 10;
	fp_orig aperture = 0.1;

	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0, 1);

	if (DEBUG)
	{
		std::cout << "CAMERA CREATED" << endl;
	}
	// Rendered Image image_array
	image_array = (unsigned char *)malloc(sizeof(unsigned char) * image_width * image_height * 3);

	// Render
	ckRendering->clockResume();
	fp_orig r, g, b;
	for (int j = 0; j < image_height; ++j)
	{
		for (int i = 0; i < image_width; ++i)
		{
			int idx = (j * image_width + i) * 3;
			color pixel_color(0, 0, 0);

			/* float version */
			for (int s = 0; s < samples_per_pixel; ++s)
			{
				fp_orig u = (i + random_orig()) / (image_width - 1);
				fp_orig v = ((image_height - j - 1) + random_orig()) / (image_height - 1);

				ray cur_ray = cam.get_ray(u, v);

				if (DEBUG)
				{
					tmp_count = 0;
					std::cout << "Pixel (" << j << "," << i << ")" << endl;
					std::cout << "[Sample " << s << "]" << endl;
					std::cout << "  - Ray #" << tmp_count << ": O=(" << cur_ray.origin() << "), D=(" << cur_ray.direction() << ")" << endl;
				}

				pixel_color += ray_color(cur_ray, world, max_depth);

				if (DEBUG)
				{
					std::cout << "=> Color = (" << val(pixel_color[0]) << ", " << val(pixel_color[1]) << ", " << val(pixel_color[2]) << ")" << endl
							  << endl;
				}
			}

			r = val(pixel_color.x());
			g = val(pixel_color.y());
			b = val(pixel_color.z());

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
	if (DEBUG)
	{
		ppmSave("img.ppm", image_array, image_width, image_height);
		return 0;
	}

	time_t t = time(NULL);
	tm *tPtr = localtime(&t);
	int timeStamp = (((tPtr->tm_year) + 1900) % 100) * 10000 + ((tPtr->tm_mon) + 1) * 100 + (tPtr->tm_mday);

	// creating directory
	string cp_size = to_string(CP_EXP_BITSIZE) + "_" + to_string(CP_MANT_BITSIZE);
	string directory = "../images/emulator_cpfloat/" + cp_size + "/" + string(typeid(fp_orig).name());
	if (DATE)
		directory += "_" + to_string(timeStamp);
	if (mkdir(directory.c_str(), 0777) == -1)
		std::cerr
			<< "Error :  " << strerror(errno) << endl;

	string img_path = directory + "_" + cp_size + "_" + to_string(image_width) + "_" + to_string(samples_per_pixel) + "_" + to_string(max_depth) + "_img.ppm";
	ppmSave(img_path.c_str(), image_array, image_width, image_height);

	return 0;
}
