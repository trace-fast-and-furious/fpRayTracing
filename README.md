# fpRayTracing 
Floating-point(FP) 연산을 사용하는 기존의 레이 트레이서 코드이다.
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
		bvh_node();

		bvh_node(const hittable_list& list, double time0, double time1)
	    		: bvh_node(list.objects, 0, list.objects.size(), time0, time1)
			{}

		bvh_node(
		    		const std::vector<shared_ptr<hittable>>& src_objects,
		    		size_t start, size_t end, double time0, double time1);

		virtual bool hit(
		    		const ray& r, double t_min, double t_max, hit_record& rec) const override;

		virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

    	public:
		shared_ptr<hittable> left;
		shared_ptr<hittable> right;
		aabb box;
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

  ### Adjustable Variables
  * `image_width`: width of the output image
  * `samples_per_pixel`: the number of samples to produce one pixel value
  * `max_depth`: the maximum number of child rays that can be generated per pixel
  * `nodesNum`: the maximum number of nodes in BVH (determines the size of stack array(`d_nodes`) and bvh array(`d_bvh`))
   
  ### How to Run the Code
  * Compile: `$ make` or `nvcc -g -G --expt-relaxed-constexpr -o RTTNW.out main.cu`
  * Execute: `$ ./RTTNW.out`
  * Debug: `$ cuda-gdb ./RTTNW.out`
    
