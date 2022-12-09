# fpRayTracing 
Floating-point(FP) 연산을 사용하는 기존의 레이 트레이서 코드이다.

## 코드 실행 관련 설명
### 사용자가 조정 가능한 변수들
> #### 이미지 관련 변수들
> * `image_width`: 결과 이미지의 가로 길이
> * `aspect_ratio`: 결과 이미지의 종횡비(가로, 세로 비율)
> * `samples_per_pixel`: 픽셀 당 쏠 레이의 개수 (값을 높일수록 이미지의 계단효과를 줄이고 정확도를 높일 수 있음)
> * `max_depth`: 레이를 추적할 횟수 (값을 낮출수록 이미지의 밝기가 어두워짐)
>
> #### 카메라 관련 변수들
> * point3 lookfrom(13,2,3): 카메라의 원점(위치)
> * point3 lookat(0,0,0): 카메라가 바라보는 지점의 위치
> * vec3 vup(0,1,0): 카메라의 머리가 가리키는 방향
> * auto dist_to_focus = 10.0: 카메라의 초점거리 (초점이 맞는 거리를 결정함)
> * auto aperture = 0.1: 카메라의 조리개값 (높일수록 더 오랜 시간동안 촬영을 함)
  
### 코드 실행 방법
> * CPU: 아래 과정을 거치지 않고 바로 실행하면 된다 *(단, CUDA 문법을 사용하는 라인이 일부 있기 때문에 현재 코드는 CPU에서 실행할 수 없다).*
> * GPU: nvcc 컴파일러를 이용해야 한다.
>> 1. 컴파일: `$ make`(Makefile을 이용하는 경우) 또는 `nvcc -g -G --expt-relaxed-constexpr -o RTTNW.out main.cu` 명령어 입력
>> 2. 실행: `$ ./RTTNW.out` 명령어 입력
>> * 디버깅: `$ cuda-gdb ./RTTNW.out` 명령어 입력

# fpRayTracing 코드 설명
레이 트레이서는 아래와 같이 크게 3가지 단계로 구현된다.
## 1. 3차원 월드(world) 생성하기
> 구현이 간단한 '구(sphere)' 물체만으로 구성된 3차원 공간 모델을 직접 생성한다. 
> 1. 구(sphere) 클래스를 정의한다.
> 2. 구 리스트를 생성한다. 
> 3. 다양한 구 객체들을 생성하여 구 리스트(=3차원 공간)에 추가한다.

1. 구(sphere) 클래스를 정의한다.
```c++
class sphere : public hittable {
public:
	// 생성자
    	sphere() {}
        sphere(point3 cen, double r, shared_ptr<material> m)
            : center(cen), radius(r), mat_ptr(m) {};
	   
    	// 구와의 충돌 여부 확인하는 함수
    	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
	// 구의 AABB를 구하는 함수
	virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

public:
    	point3 center;  // 구의 중심
    	double radius;  // 반지름
        shared_ptr<material> mat_ptr;  // 재질
};
```

2. 구 리스트를 생성한다. 
```c++
int main() {
	hittable_list world = create_world();  
	...
}
```

3. 다양한 구 객체들을 생성하여 구 리스트(=3차원 공간)에 추가한다.
```c++
hittable_list create_world() {
	hittable_list world;  // 월드(구 리스트)
   	int n = 10;  // 오브젝트 개수 결정 인자	
	// 1. 오브젝트 객체들을 생성해서 world에 저장
	// (1) 땅(ground)의 역할을 하는 구(sphere)
	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));


	// (2) 작은 구들 (4n^2개): 랜덤하게 구의 재질(material type)을 선택함
   	for (int a = -n; a < n; a++) {
	for (int b = -n; b < n; b++) {

		double choose_mat = (a * 11 + b)/121;  // 재질을 결정하는 인자
			
		// 구의 특징 결정
		double choose_mat = random_double();  // 재질 (랜덤)
	    	point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());  // 구의 중심

	    	if ((center - point3(4, 0.2, 0)).length() > 0.9) {  // 화면 밖으로 벗어난 구들은 생성하지 않음

			   	shared_ptr<material> sphere_material;
				
				// 랜덤하게 재질 결정
				if (choose_mat < 0.8) {
					// (a) Diffuse (매끈한 물체)
		    			auto albedo = color::random() * color::random();  // 반사율
		    			sphere_material = make_shared<lambertian>(albedo); 
					auto center2 = center + vec3(0, random_double(0,.5), 0);  // 움직이는 구: 움직인 후의 중심 
		    			world.add(make_shared<moving_sphere>(center, center2, 0.0, 1.0, 0.2, sphere_material));
				} else if (choose_mat < 0.95) {
					// (2) Metals (금속)
	    				auto albedo = color::random(0.5, 1);
	    				auto fuzz = random_double(0, 0.5);
					sphere_material = make_shared<metal>(albedo, fuzz);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				} else {
					// (3) Dielectrics (투명한 물체)
					sphere_material = make_shared<dielectric>(1.5);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
    			}
		}
	}

	// (3) 큰 구들 (3개)
	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));
	
	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	//return world;  // (BVH 사용하지 않은) 3차원 공간 
	
	// BVH 생성
	hittable_list world_bvh;
	world_bvh.add(make_shared<bvh_node>(world, 0, 1));

	return world_bvh;
}
```

## 1.2. Bounding Volume Hierarchy (BVH) 생성하기
> 레이 트레이싱의 연산(레이가 물체와 충돌하는지 여부를 확인하는 연산) 개수를 줄이기 위해 사용하는 트리형 가속(acceleration) 자료구조인 BVH를 생성한다.
> 1. BVH 트리를 구성하는 노드의 클래스를 정의한다.
> 2. 물체 리스트를 이용하여 BVH 트리를 생성한다.

1. BVH 트리의 노드 클래스를 정의한다.
```c++
class bvh_node : public hittable {
    	public:
		// 생성자
		bvh_node();
		bvh_node(const hittable_list& list, double time0, double time1)
			: bvh_node(list.objects, 0, list.objects.size(), time0, time1) {}
		bvh_node(
			const std::vector<shared_ptr<hittable>>& src_objects,
		    	size_t start, size_t end, double time0, double time1);
			
		// BVH 노드와의 충돌 여부를 확인하는 함수
		virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
		// BVH 노드의 AABB를 생성하는 함수
		virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

    	public:
		shared_ptr<hittable> left;  // (현재 노드의) 왼쪽 자식 노드
		shared_ptr<hittable> right; // 오른쪽 자식 노드
		aabb box;  // 노드를 감싸는 Axis-aligned bounding box (AABB)
};
```

2. 물체 리스트를 이용하여 BVH 트리를 생성한다.
```c++
hittable_list create_world() {
	...
	
	// BVH Construction
	hittable_list world_bvh;
	world_bvh.add(make_shared<bvh_node>(world, 0, 1));

	return world_bvh;
}
```

## 2. 이미지의 해상도를 결정하고 카메라 생성하기
> 렌더링할 이미지의 해상도(가로, 세로 픽셀 개수)와 가상의 카메라를 생성해서 3차원 공간에 위치시킨다.
> 1. 이미지의 해상도를 결정한다.
> 2. 카메라의 클래스를 정의한다.
> 3. 카메라 객체를 생성한다.

1. 이미지의 해상도를 결정한다.
```c++
int main() 
{
	...
	// 이미지 설정
	auto aspect_ratio = 16.0 / 9.0;  // 이미지 종횡비
    	int image_width = 400;  // 가로 길이
	int image_height = static_cast<int>(image_width / aspect_ratio); // 세로 길이
   	int samples_per_pixel = 100;  // 픽셀 당 쏠 레이의 개수 (샘플 개수)   
	...
}
```

2. 카메라의 클래스를 정의한다.
```c++
class camera {
	public:
		// 생성자
	public:
		camera(
			point3 lookfrom,
			point3 lookat,
			vec3   vup,	
			double vfov, // vertical field-of-view in degrees
			double aspect_ratio,
			double aperture,
			double focus_dist,
			double _time0 = 0,
			double _time1 = 0

		      ) {
			auto theta = degrees_to_radians(vfov);
			auto h = tan(theta/2);
			auto viewport_height = 2.0 * h;
			auto viewport_width = aspect_ratio * viewport_height;		

			auto w = unit_vector(lookfrom - lookat);
			auto u = unit_vector(cross(vup, w));
			auto v = cross(w, u);

			origin = lookfrom;
			horizontal = focus_dist * viewport_width * u;
			vertical = focus_dist * viewport_height * v;
			lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;
			lens_radius = aperture / 2;
			time0 = _time0;
			time1 = _time1;
		}
		// (카메라로부터) 레이를 지정한 픽셀로 쏘는 함수
		ray get_ray(double s, double t) const {
			vec3 rd = lens_radius * random_in_unit_disk();
			vec3 offset = u * rd.x() + v * rd.y();
			// 레이 객체를 생성해서 반환
			return ray(
				origin + offset,  // 레이가 쏘아지는 원점의 위치
				lower_left_corner + s*horizontal + t*vertical - origin - offset,  // 레이를 쏘는 지점의 위치
				random_double(time0, time1)  // 레이를 쏠 시간
			);
		}
}
```

3. 카메라 객체를 생성한다.
```c++
int main() 
{
	...
	// 카메라 정보 설정
	point3 lookfrom(13,2,3);  // 카메라의 위치
	point3 lookat(0,0,0);  // 카메라가 바라보는 지점의 위치
	vec3 vup(0,1,0); // 카메라의 머리 방향
	auto dist_to_focus = 10.0;  // 카메라의 초점거리
	auto aperture = 0.1;  // 카메라의 조리개값

	// 카메라 객체 생성
	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
	...
}
```
## 3. 레이 트레이싱을 사용하여 렌더링하기
> 레이 트레이싱 기법을 사용하여 결과 이미지를 생성한다.
> 1. 카메라로부터 각 픽셀마다 레이를 쏘아서 경로를 추적하여 해당 픽셀의 색깔을 결정한다.

1. 카메라로부터 각 픽셀마다 레이를 쏘아서 경로를 추적하여 해당 픽셀의 색깔을 결정한다.
```c++
int main()
{
	...
	// 각 픽셀마다
	for (int j = 0; j < image_height; ++j) {  
		for (int i = 0; i < image_width; ++i) {
			int idx = (j * image_width + i) * 3;  // 픽셀의 인덱스
			color pixel_color(0, 0, 0);  // 픽셀값(RGB)

			// 픽셀마다 여러 개의 레이(샘플)을 사용하여 매끄러운 이미지를 만듦
			for (int s = 0; s < samples_per_pixel; ++s) {
				// 픽셀 영역 내의 랜덤한 위치로 레이를 쏨
				auto u = (i + random_float()) / (image_width - 1);  
				auto v = ((image_height-j-1) + random_float()) / (image_height - 1);

				ray cur_ray = cam.get_ray(u, v);  // 현재 레이
				pixel_color += ray_color(cur_ray, world, max_depth);  // 레이의 경로를 추적

				r = pixel_color.x();
				g = pixel_color.y();
				b = pixel_color.z();

				// 이미지 매끄럽게 만들기 (Antialiasing)
				float scale = 1.0 / samples_per_pixel;
				r = sqrt(scale * r);
				g = sqrt(scale * g);
				b = sqrt(scale * b);	
			}
		array[idx] = (256 * clamp(r, 0.0, 0.999));
		array[idx+1] = (256 * clamp(g, 0.0, 0.999));
		array[idx+2] = (256 * clamp(b, 0.0, 0.999));
		}
	}
	...
}
```

  ### Adjustable Variables
  * `image_width`: width of the output image
  * `samples_per_pixel`: the number of samples to produce one pixel value
  * `max_depth`: the maximum number of child rays that can be generated per pixel
  * `nodesNum`: the maximum number of nodes in BVH (determines the size of stack array(`d_nodes`) and bvh array(`d_bvh`))
   
  ### How to Run the Code
  * Compile: `$ make` or `nvcc -g -G --expt-relaxed-constexpr -o RTTNW.out main.cu`
  * Execute: `$ ./RTTNW.out`
  * Debug: `$ cuda-gdb ./RTTNW.out`
    
