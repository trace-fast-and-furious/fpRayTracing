/*
 * ===================================================
 *
 *       Filename:  main.cu
 *    Description:  Ray Tracing In One Weekend (RTIOW): ~BVH 
 *        Created:  2022/07/13
 *  Last Modified: 2023/05/05
 * 
 * ===================================================
 */

// Preprocessors
#include "mesh.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"
#include "mkPpm.h"

using namespace custom_precision_fp;

// Function Prototypes
Color computeRayColor(const Ray &ray, Mesh &mesh, int depth);
void render(int image_height, int image_width, int samples_per_pixel, int depth, unsigned char *image, const Camera &cam, Mesh &mesh);

// Global variables
unsigned char *out_image;
const string MESHNAME = "dragon";
const char IMG_TYPE = '2';  // 1: for SSIM graph, 2: for output image, 3: camera test
fp_orig t_epsilon = 0.001;

int main(void)
{
    // Allocate the data structure for target formats and rounding parameters.
    fpopts = init_optstruct();

    // Set up the parameters for binary16 target format.
    fpopts->precision = CP_MANT_BITSIZE + 1;         // Bits in the significand + 1.
    fpopts->emax = pow((CP_EXP_BITSIZE - 1), 2) - 1; // The maximum exponent value(=bias)
    fpopts->subnormal = CPFLOAT_SUBN_USE;            // Support for subnormals is on.
    fpopts->round = CPFLOAT_RND_NE;                  // Round toward +infinity.
    fpopts->flip = CPFLOAT_NO_SOFTERR;               // Bit flips are off.
    fpopts->p = 0;                                   // Bit flip probability (not used).
    fpopts->explim = CPFLOAT_EXPRANGE_TARG;          // Limited exponent in target format.

    char *clock_name = "CPU CODE";
    mkClockMeasure *ckCpu = new mkClockMeasure(clock_name);
    ckCpu->clockReset();

    // mesh
    Mesh mesh;

    // material
    auto metal = make_shared<Metal>(Color(0.6, 0.6, 0.3), 0);
    auto dielectric = make_shared<Dielectric>(0.3);
    auto lambertian = make_shared<Lambertian>(Color(0.8, 0.6, 0.3));
    auto dielectric2 = make_shared<Dielectric>(0.1);

    // Load meshes  
    string mesh_filename = "../obj/" + MESHNAME + ".obj";  // object file  
	loadObjFile(mesh_filename, mesh);
    mesh.setMaterial(metal);   
    // printMesh(mesh_filename, mesh);

    // Image
	// auto aspect_ratio = 16.0 / 9.0;
    auto aspect_ratio = 1;
    int image_width, image_height, samples_per_pixel, max_depth;

    if(IMG_TYPE == '1'){
        image_width = 128;
        image_height = static_cast<int>(image_width / aspect_ratio);
        samples_per_pixel = 1;    
        max_depth = 50;
    }
    else if(IMG_TYPE == '2')
    {
        image_width = 1;
        image_height = static_cast<int>(image_width / aspect_ratio);
        samples_per_pixel = 10;    
        max_depth = 50;
    }
    else if(IMG_TYPE == '3')  // test
    {
        image_width = 10;
        image_height = static_cast<int>(image_width / aspect_ratio);
        samples_per_pixel = 1;    
        max_depth = 3;
    }

    // Camera
    Vec3 vup(0, 1, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;
    
    int fov;  // field of view
    Point3 lookfrom, lookat;

    if(!(MESHNAME.compare("stanford_bunny"))){
        lookfrom = Point3(0.5, 0.3, 0.5);
        lookat = Point3(-0.05,0.1,0);
        fov = 23;
        t_epsilon = 0.06;
    }
    else if(!(MESHNAME.compare("dragon"))) {
        // lookfrom = Point3(0.24, 0.3, 0.5);
        // lookat = Point3(-0.01,0.1,0);
        // fov = 32;
        // t_epsilon = 0.055;
        lookfrom = Point3(0.25, 0.3, 0.5);
        lookat = Point3(0.0,0.1,0);
        fov = 25;
        t_epsilon = 0.054;
    }
    else if(!(MESHNAME.compare("Armadillo"))) {
        lookfrom = Point3(-80, 30, 80);
        lookat = Point3(0,30,0);
        fov = 120;
        t_epsilon = 0.05;
    }
    else if(!(MESHNAME.compare("HappyBuddha"))) {
        lookfrom = Point3(0.3, 0.3, 0.5);
        lookat = Point3(0,0.1,0);
        fov = 35;
        t_epsilon = 0.05;
    }

    if(CP_EXP_BITSIZE == 11 && CP_MANT_BITSIZE == 52){
        t_epsilon = 0.001;
    }

    Camera cam(lookfrom, lookat, vup, fov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);  // pyramid
    // Camera cam(lookfrom, lookat, vup, 16, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);  // bunny
    // Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);  // dino, bunny2_2
    // Camera cam(lookfrom, lookat, vup, 6, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);  // pig

    // Rendered Image Array
    out_image = (unsigned char *)malloc(sizeof(unsigned char) * image_width * image_height * 3);

    // Measure the rendering time
    ckCpu->clockResume();


    // Render an image
    cout << "Image Type: " << IMG_TYPE << ". ";
    if(IMG_TYPE == '1') cout << "SSIM" << endl;
    else if(IMG_TYPE == '2') cout << "SHOW" << endl;
    else if(IMG_TYPE == '3') cout << "TEST" << endl;

    cout << MESHNAME << ".ppm: " << image_width << "x" << image_height << "_s" << samples_per_pixel << "_d" << max_depth << "_e" << t_epsilon << endl;
    cout << "  - Exponent: " << to_string(CP_EXP_BITSIZE) << ", Mantissa: " << to_string(CP_MANT_BITSIZE) << endl;
    render(image_height, image_width, samples_per_pixel, max_depth, out_image, cam, mesh);

    ckCpu->clockPause();
    ckCpu->clockPrint();

    // Save a PPM image
    // Directories
    string directory1, directory2, directory3;
    
    if(IMG_TYPE == '1') {
        directory1 = "../img1.SSIM/";
        directory2 = directory1 + MESHNAME + "_E" + to_string(CP_EXP_BITSIZE) + "_M" + to_string(CP_MANT_BITSIZE) + "/";
        directory3 = directory2 + to_string(image_width) + "x" + to_string(image_height) + "_s" + to_string(samples_per_pixel) + "_d" + to_string(max_depth) + "/";
    }
    if(IMG_TYPE == '2') {
        directory1 = "../img2.SHOW/";
        directory2 = directory1 + MESHNAME + "_E" + to_string(CP_EXP_BITSIZE) + "_M" + to_string(CP_MANT_BITSIZE) + "/";
        directory3 = directory2 + to_string(image_width) + "x" + to_string(image_height) + "_s" + to_string(samples_per_pixel) + "_d" + to_string(max_depth) + "/";
    }
    if(IMG_TYPE == '3') {
        directory1 = "../img3.TEST/";
        directory2 = directory1 + MESHNAME + "_E" + to_string(CP_EXP_BITSIZE) + "_M" + to_string(CP_MANT_BITSIZE) + "/";
        directory3 = directory2 + to_string(image_width) + "x" + to_string(image_height) + "_s" + to_string(samples_per_pixel) + "_d" + to_string(max_depth) + "/";
    }

    if (mkdir(directory1.c_str(), 0777) == -1)
	{
		//cerr << "Error :  " << strerror(errno) << endl;
		if(errno == EEXIST)
		{
			cout << "1: File exists" << endl;
		}
		else cout << "1: No such file or directory" << endl;

	}
    if (mkdir(directory2.c_str(), 0777) == -1)
	{
		//cerr << "Error :  " << strerror(errno) << endl;
		if(errno == EEXIST)
		{
			cout << "2: File exists" << endl;
		}
		else cout << "2: No such file or directory" << endl;

	}
    if (mkdir(directory3.c_str(), 0777) == -1)
	{
		//cerr << "Error :  " << strerror(errno) << endl;
		if(errno == EEXIST)
		{
			cout << "3: File exists" << endl;
		}
		else cout << "3: No such file or directory" << endl;

	}

	// Filename
    string str_lf = to_string(val(lookfrom.x())) + "_" + to_string(val(lookfrom.y())) + "_" + to_string(val(lookfrom.z()));
    string str_la = to_string(val(lookat.x())) + "_" + to_string(val(lookat.y())) + "_" + to_string(val(lookat.z()));
	// string cam_info = "LF(" + str_lf + ")_LA(" + str_la + ")" + "_fov" + to_string(fov);
	// string img_path = directory2 + cam_info + ".ppm";
	
    
    string img_path = directory3 + MESHNAME + "_w" + to_string(image_width) + "_s" + to_string(samples_per_pixel) + "_d" + to_string(max_depth) + "_e" + to_string(t_epsilon) + ".ppm";
    // string img_path = "../img_final_HP/" + MESHNAME + "_w" + to_string(image_width) + "_s" + to_string(samples_per_pixel) + "_d" + to_string(max_depth) + ".ppm";

    ppmSave(img_path.c_str(), out_image, image_width, image_height, samples_per_pixel, max_depth);

    return 0;
}

// render: renders an output image.
void render(int image_height, int image_width, int samples_per_pixel, int depth, unsigned char *image, const Camera &cam, Mesh &mesh)
{
    // RT18
    // PRINT PIXEL VALUES OF THE OUTPUT IMAGE: printf("------------------- IMAGE -------------------\n");

    // Render
    float r, g, b;
    for (int j = 0; j < image_height; ++j)
    {
        cout << "[" << MESHNAME << "] h: " << j << endl; 
        
        for (int i = 0; i < image_width; ++i)
        {
            // cout << "[Rendering] w: " << i << endl; 

            int idx = (j * image_width + i) * 3;
            Color pixel_color(0, 0, 0);

            for (int s = 0; s < samples_per_pixel; ++s)
            {
                float u = (i + random_orig()) / (image_width - 1);
                float v = ((image_height - j - 1) + random_orig()) / (image_height - 1);

                Ray cur_ray = cam.get_ray(u, v);
                pixel_color += computeRayColor(cur_ray, mesh, depth);

                r = val(pixel_color.x());
                g = val(pixel_color.y());
                b = val(pixel_color.z());

                // Antialiasing
                float scale = 1.0 / samples_per_pixel;
                r = std::sqrt(scale * r);
                g = std::sqrt(scale * g);
                b = std::sqrt(scale * b);
            }
            out_image[idx] = (256 * clamp(r, 0.0, 0.999));
            out_image[idx + 1] = (256 * clamp(g, 0.0, 0.999));
            out_image[idx + 2] = (256 * clamp(b, 0.0, 0.999));

            // RT18 - PRINT PIXEL VALUES OF THE OUTPUT IMAGE:
            //			printf("  R:%d, G:%d, B:%d\n", out_image[idx], out_image[idx+1], out_image[idx+2]);
        }
    }
}

// computeRayColor: calculates color of the current ray intersection point.
Color computeRayColor(const Ray &ray, Mesh &mesh, int depth)
{
    HitRecord rec;
    Color cur_color{1.0, 1.0, 1.0};

    // Limit the number of child ray.
    if (depth <= 0)
    {
        return Color(0, 0, 0); // If the ray hits objects more than 'depth' times, consider that no light approaches the current point.
    }

    // If the ray hits an object
    if (mesh.hit(ray, t_epsilon, rec))  // updated (230507)
    {
        Ray ray_scattered;
        Color attenuation;

        if (rec.mat_ptr->scatter(ray, rec, attenuation, ray_scattered)) // Decide color of the current intersection point
        {
            cur_color = attenuation * computeRayColor(ray_scattered, mesh, depth - 1);
            return cur_color;
        }
        return Color(0, 0, 0);
    }
    else
    {
        // If the ray hits no object: Background
        Vec3 unit_direction = unit_vector(ray.direction());
        fp_custom t;

        if(IMG_TYPE == '1') {  // SSIM
            t = fp_orig_to_custom(1.5);  // no gradation
        }
        else if(IMG_TYPE == '2') {  // SHOW
            t = 0.5 * (unit_direction.y() + 1.0);  // gradation and more realistic shadow
        }

        // return (1.0 - t) * Color(0.3, 0.3, 0.3) + t * Color(0, 0, 0);
        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);       
    }
}