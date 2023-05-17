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

std::ofstream file("output.txt");
static int tmp_count;

unsigned char *image_array;

// 1. random_scene: Implements the 3D World.
hittable_list random_scene()
{
	hittable_list world;
	auto ground_material = make_shared<lambertian>(color(0.45, 0.45, 0));
	world.add(make_shared<sphere>(0, point3(0, -1000, 0), 1000, ground_material));

	color albedo;
	shared_ptr<material> sphere_material;
	fp_orig radius;
	point3 center;

	// diffuse
	std::vector<point3> diffuse_center = {
		point3(-2.434016167977825, 0.1553969955537469, -1.57034265352413),
		point3(-2.78140190653503, 0.1606968875974417, -0.9853294855449348),
		point3(-1.922549736965448, 0.1949327074922622, -2.526604185020552),
		point3(-1.12642928189598, 0.1063095838297159, -1.785548041341826),
		point3(-1, 0.173853431455791, -1),
		point3(-1.381997082801536, 0.1893372414167971, 1.31532416054979),
		point3(-1.180125172669068, 0.1814766895957292, 2.615796672459692),
		point3(-0.6251488476525993, 0.1276234671939165, -1.499200620455667),
		point3(-0.3294357500970363, 0.1793470387812704, 1.147691740235314),
		point3(0.1838957495056093, 0.1157807128503919, -1.100005786446855),
		point3(0.1623350390698761, 0.1923069126904011, -0.4654970389325171),
		point3(1.069295212719589, 0.1830011833459139, -2.199946125224232),
		point3(1.569764973921701, 0.1231427952181548, -1.933255127165467),
		point3(1.538633169420064, 0.1003231459762901, 0.3104486593510956),
		point3(1.564442097442225, 0.1184622467495501, 2.548195275571198),
		point3(2.533890309324488, 0.1436496996786445, 0.8627732652705165),
	};
	std::vector<color> diffuse_albedo = {
		point3(0.996, 0.43, 0),
		point3(0, 0.289, 0.176),
		point3(0.296, 0.359, 0.769),
		point3(0, 0.63, 0.87),
		point3(1, 0.85, 0),
		point3(0, 0.4375, 0.469),
		point3(0, 0.176, 0.453),
		point3(0.95, 0.76, 0),
		point3(0, 0.62, 0.418),
		point3(1, 1, 1),
		point3(0.9, 0.3, 0),
		point3(0, 0, 0.996),
		point3(0.465, 0.465, 0.473),
		point3(0, 0.289, 0.176),
		point3(0.48, 0.1, 0.6),
		point3(0.98, 0.668, 0.09),
		point3(0.258, 0.258, 0.242),
	};
	std::vector<fp_orig> diffuse_radius = {
		0.1553969955537468,
		0.1606968875974417,
		0.1949327074922621,
		0.1063095838297159,
		0.173853431455791,
		0.1893372414167971,
		0.1814766895957291,
		0.1276234671939165,
		0.1793470387812704,
		0.1157807128503919,
		0.1923069126904011,
		0.1830011833459139,
		0.1231427952181548,
		0.1003231459762901,
		0.1184622467495501,
		0.1436496996786445};

	// metal
	std::vector<point3> metal_center = {
		point3(-2.281403970206156, 0.1394382926635445, -2.295210698945448),
		point3(-2.528141529159621, 0.1296031617559493, 1.573797040665522),
		point3(-2.197623493568972, 0.1769913835916668, 2.360205759713426),
		point3(-1.96464769099839, 0.1667723760474473, -0.5215542094781995),
		point3(-0.7470465289428829, 0.3, -4),
		point3(-0.7909646153450012, 0.1368663541506976, 0.2647443256806583),
		point3(-0.1293353757355362, 0.173265441134572, 2.590907287411392),
		point3(0.1841895775403828, 0.107823214167729, -2.937084242049605),
		point3(0.0896760571282357, 0.1452575845643878, 0.6186486908700317),
		point3(0.5660189379472287, 0.1877613777760416, 1.67302836640738),
		point3(0.6694300862029197, 0.1831037540454418, 2.881490715965628),
		point3(1.533285926748067, 0.1113280554767698, -0.5756649139337242),
		point3(1.036777872499079, 0.1182555787265301, 1.559640538878739),
		point3(2.078879265207798, 0.1257265482097865, -2.520803124736995),
		point3(2.814439223892987, 0.1548042049631477, 2.267559607047588),
	};
	std::vector<fp_orig> metal_radius = {
		0.1394382926635444,
		0.1296031617559493,
		0.1769913835916668,
		0.1667723760474473,
		0.3,
		0.1368663541506976,
		0.173265441134572,
		0.107823214167729,
		0.1452575845643878,
		0.1877613777760416,
		0.1831037540454418,
		0.1113280554767698,
		0.1182555787265301,
		0.1257265482097864,
		0.1548042049631476};
	std::vector<color> metal_albedo = {
		point3(0.6676113777793944, 0.598775684600696, 0.9558236787561327),
		point3(0.6462583921384066, 0.9863875117152929, 0.7467914933804423),
		point3(0.9038622598163784, 0.6762291735503823, 0.6416573729366064),
		point3(0.9654048974625766, 0.9659175279084593, 0.7188187981955707),
		point3(0.7235167894978076, 0.6537289367988706, 0.8930010488256812),
		point3(0.5761948958970606, 0.6222063677851111, 0.7922442501876503),
		point3(0.5467402385547757, 0.87986742076464, 0.8197291726246476),
		point3(0.786659314064309, 0.9098386398982257, 0.730710236588493),
		point3(0.6521475748158991, 0.8786469160113484, 0.7654039938934147),
		point3(0.9166192708071321, 0.8739014333114028, 0.5177104533649981),
		point3(0.8334401671309024, 0.9917978688608855, 0.9516831699293107),
		point3(0.6681755627505481, 0.7254587789066136, 0.9721590478438884),
		point3(0.8369682480115443, 0.8479918953962624, 0.7069918697234243),
		point3(0.8430624636821449, 0.9386919636745006, 0.6302484711632133),
		point3(0.7490719631314278, 0.9369894838891923, 0.9548214785754681)};
	std::vector<fp_orig> metal_fuzz = {
		0.3841147972270846,
		0.3856788487173617,
		0.4595132367685437,
		0.3604761713650078,
		0.1130533125251532,
		0.3660742577631027,
		0.06745120580308139,
		0.377790417522192,
		0.4961142304819077,
		0.4626882753800601,
		0.2486292594112456,
		0.4238422177731991,
		0.318820076296106,
		0.04687011800706387,
		0.2880997739266604};

	// dialectric
	std::vector<point3> dielectric_center = {
		point3(-2.244798989128322, 0.1218256905209273, 0.4616391547489912),
		point3(-0.655130502069369, 0.1935003986116499, -0.3839994851034134),
		point3(2.298331167828292, 0.100357857439667, -0.2553479784633964),
		point3(2.183193394634872, 0.1916272911708802, 1.775725410040468)};
	std::vector<fp_orig> dielectric_radius = {
		0.1218256905209273,
		0.1935003986116499,
		0.100357857439667,
		0.1916272911708802};

	for (int i = 0; i < diffuse_center.size(); i++)
	{
		sphere_material = make_shared<lambertian>(diffuse_albedo[i]);
		world.add(make_shared<sphere>(0, diffuse_center[i], diffuse_radius[i], sphere_material));
	}
	for (int i = 0; i < metal_center.size(); i++)
	{
		sphere_material = make_shared<metal>(metal_albedo[i], metal_fuzz[i]);
		world.add(make_shared<sphere>(0, metal_center[i], metal_radius[i], sphere_material));
	}
	for (int i = 0; i < dielectric_center.size(); i++)
	{
		sphere_material = make_shared<dielectric>(1.5);
		world.add(make_shared<sphere>(0, dielectric_center[i], dielectric_radius[i], sphere_material));
	}

	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(0, point3(4, 0.7, 0), 0.7, material1));

	// auto material2 = make_shared<lambertian>(color(0.765, 0.008, 0.2));
	auto material2 = make_shared<lambertian>(color(0.765, 0.008, 0.2));
	world.add(make_shared<sphere>(0, point3(-8, 0.7, -3), 0.7, material2));

	auto material3 = make_shared<metal>(color(0.85, 0.85, 0.85), 0.0);
	world.add(make_shared<sphere>(0, point3(-5, 0.7, 0), 0.7, material3));

	return world;
}

// 2. ray_color: calculates color of the current ray intersection point.
color ray_color(const ray &r, const hittable &world, int depth)
{
	hit_record rec;

	if (depth <= 0)
		return color(0, 0, 0);

	if (world.hit(r, 0.001, std::numeric_limits<fp_orig>::infinity(), rec)) // hit: 충돌지점을 결정(child ray 의 origin)
	{																		// if ray hits object
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

	// Set up the parameters for target format.
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
	cout << "image width: " << image_width << ", sample #: " << samples_per_pixel << endl;

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
	point3 lookat(0, 0.3, 0);
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

	file.close();

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
	// string directory = "../images/emulator_cpfloat/test/" + string(typeid(fp_orig).name());
	string directory = "../images/emulator_cpfloat/36_spheres/1920_1000_50/";
	if (DATE)
		directory += "_" + to_string(timeStamp);
	if (mkdir(directory.c_str(), 0777) == -1)
		std::cerr
			<< "Error :  " << strerror(errno) << endl;

	string img_path = directory + string(typeid(fp_orig).name()) + "_" + cp_size + "_" + to_string(image_width) + "_" + to_string(samples_per_pixel) + "_" + to_string(max_depth) + "_img.ppm";
	ppmSave(img_path.c_str(), image_array, image_width, image_height);

	return 0;
}
