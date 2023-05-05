// Preprocessors
#include "mesh.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"
#include "mkPpm.h"

#define CAM_CHECK 0


using namespace custom_precision_fp;

// Function Prototypes
Color computeRayColor(const Ray &ray, Mesh &mesh, int depth);
void render(int image_height, int image_width, int samples_per_pixel, int depth, unsigned char *image, const Camera &cam, Mesh &mesh);

// Global variables
unsigned char *out_image;
const string MESHNAME = "stanford_bunny";


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
    auto metal = make_shared<Metal>(Color(0.6, 0.6, 0.3), 0.2);
    auto dielectric = make_shared<Dielectric>(0.3);
    auto lambertian = make_shared<Lambertian>(Color(0.8, 0.6, 0.3));
    auto dielectric2 = make_shared<Dielectric>(0.1);

    // filename
    string img_filename = MESHNAME + ".ppm";  // output image file
    char p_img_filename[30];
    strcpy(p_img_filename, img_filename.c_str());
    string mesh_filename = "../obj/" + MESHNAME + ".obj";  // object file

    // Load meshes    
	loadObjFile(mesh_filename, mesh);
    mesh.setMaterial(lambertian);   
    // printMesh(mesh_filename, mesh);

    // Image
	// auto aspect_ratio = 16.0 / 9.0;
    auto aspect_ratio = 1;
	int image_width = 512;
    int image_height = static_cast<int>(image_width / aspect_ratio);
	int samples_per_pixel = 100;    
	int max_depth = 100;

    if(CAM_CHECK)
    {
        image_width = 30;
        image_height = static_cast<int>(image_width / aspect_ratio);
        samples_per_pixel = 1;    
        max_depth = 3;
    }

    // Camera

    // 1) Stanford Bunny
    // Point3 lookfrom(-0.1, 0.7, -0.1);
    // Point3 lookat(0,-0.1,0);


    // pyramid
    	// Point3 lookfrom(13,3,3);
    //    Point3 lookat(0,0,0);

    // cow
    //	Point3 lookfrom(50, 0, 50);
    //	Point3 lookat(0, 0, 0);

    // dino bottom
    // Point3 lookfrom(-100, -100, -100);
    // Point3 lookat(0, 5.5, 0);

    // dino front
    // Point3 lookfrom(-200, 0, 0);
    // Point3 lookat(0, 10, 0);

    // bunny1
    Point3 lookfrom(0.5, 0.3, 0.5);
    Point3 lookat(-0.05,0.1,0);

    // bunny2
    //  Point3 lookfrom(-3, 2, 3);
    //  Point3 lookat(0, 0.65, 0);

    Vec3 vup(0, 1, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;
    
    Camera cam(lookfrom, lookat, vup, 23, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);  // SF bunny
    // Camera cam(lookfrom, lookat, vup, 10, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);  // bunny
    //Camera cam(lookfrom, lookat, vup, 16, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0); // dino
    // Camera cam(lookfrom, lookat, vup, 10, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0); // cow

    // Rendered Image Array
    out_image = (unsigned char *)malloc(sizeof(unsigned char) * image_width * image_height * 3);

    // Measure the rendering time
    ckCpu->clockResume();

    // Render an image
    render(image_height, image_width, samples_per_pixel, max_depth, out_image, cam, mesh);

    ckCpu->clockPause();
    ckCpu->clockPrint();


    // Save a PPM image
    // Directory
    string directory1 = "../img/" + MESHNAME + "/";
    string directory2 = directory1 + to_string(image_width) + "x" + to_string(image_height) + "_s" + to_string(samples_per_pixel) + "_d" + to_string(max_depth) + "/";
	
	// if (mkdir("../img/", 0777) == -1 && mkdir(directory1.c_str(), 0777) == -1 && mkdir(directory2.c_str(), 0777) == -1)
    if (mkdir("../img/", 0777) == -1 && mkdir(directory1.c_str(), 0777) == -1 && mkdir(directory2.c_str(), 0777) == -1)
	{
		//cerr << "Error :  " << strerror(errno) << endl;
		if(errno == EEXIST)
		{
			cout << "File exists" << endl;
		}
		else cout << "No such file or directory" << endl;

	}

	// Filename
    string str_lf = to_string(val(lookfrom.x())) + "_" + to_string(val(lookfrom.y())) + "_" + to_string(val(lookfrom.z()));
    string str_la = to_string(val(lookat.x())) + "_" + to_string(val(lookat.y())) + "_" + to_string(val(lookat.z()));
	string cam_info = "LF(" + str_lf + ")_LA(" + str_la + ")";
	string img_path = directory2 + cam_info + ".ppm";
	//string img_path = "../img/mant_400_s3_d50_n3/e" + to_string(BFP_EXP_BITSIZE) + "m" + to_string(BFP_MANT_BITSIZE) + "_A" + str_a + ".ppm";
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
        cout << "[Rendering] h: " << j << endl; 
        
        for (int i = 0; i < image_width; ++i)
        {
            cout << "[Rendering] w: " << j << endl; 

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
    if (mesh.hit(ray, 0.001, rec))
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
        // fp_custom t = 0.5 * (unit_direction.y() + 1.0);
        fp_custom t = fp_orig_to_custom(1.5);

        // return (1.0 - t) * Color(0.3, 0.3, 0.3) + t * Color(0, 0, 0);
        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);       
    }
}
