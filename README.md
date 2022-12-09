# fpRayTracing 
Floating-point(FP) 연산을 사용하는 기존의 레이 트레이서 코드이다.
레이 트레이서는 아래와 같이 크게 3가지 단계로 구현된다.

## 1. 3차원 월드(world) 생성하기
> 구현이 간단한 '구(sphere)' 물체만으로 구성된 3차원 공간 모델을 직접 생성한다. 
> * 구(sphere) 클래스를 정의한다.
> * 다양한 구 객체들을 생성하여 구 리스트(=3차원 공간)을 생성한다.
'''c
hittable_list random_scene() {
	/* [3D World 생성하는 과정]
		- world: 3D 공간의 오브젝트들을 저장하는 리스트 
		- n: 작은 구들의 개수를 결정하는 인자
	*/
    hittable_list world;
	int n = 10;
    
	
	/* 1. 오브젝트 객체들을 생성해서 world에 저장한다.
	 	(1) 1개의 오브젝트: 땅(ground)의 역할을 하는 구(sphere) */
	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));


	/* (2) 4n^2개의 오브젝트들: 재질(material type)이 랜덤한 작은 구들 */
    for (int a = -n; a < n; a++) {
		for (int b = -n; b < n; b++) {

			/* - choose_mat: 재질을 결정하는 변수 (우리는 항상 동일한 결과를 내기 위해 일정한 패턴으로 재질을 결정한다) */
			// Generate constant scene primitives.
			double choose_mat = (a * 11 + b)/121;
			
			/* (교과서 방식: 랜덤하게 재질을 결정) */
			// Generate random scene primitives.
			//auto choose_mat = random_double();

			/* - center: 현재 오브젝트의 원점(origin) ('랜덤'하게 선택한다?! -> 확인 필요!!!) */
	    	point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());


			/* choose_mat과 center 변수값을 토대로 오브젝트의 재질과 움직임을 결정해서 객체를 생성한다. */
	    	if ((center - point3(4, 0.2, 0)).length() > 0.9) {  /* 화면 밖으로 벗어난 구들을 생성하지 않는다. */
			   	shared_ptr<material> sphere_material;  // pointer that points the new material

				if (choose_mat < 0.8) {  // Decide the material type

					// (1) diffuse
		    		auto albedo = color::random() * color::random();
		    		sphere_material = make_shared<lambertian>(albedo);
					/* 현재 위치를 '랜덤'하게 선택한다?! -> 확인 필요!!! */
					auto center2 = center + vec3(0, random_double(0,.5), 0);  // center of the moving sphere at time2(now) 
		    		world.add(make_shared<moving_sphere>(
				   	center, center2, 0.0, 1.0, 0.2, sphere_material));
			
				} else if (choose_mat < 0.95) {

					// (2) metal
	    			auto albedo = color::random(0.5, 1);
	    			auto fuzz = random_double(0, 0.5);
	    			sphere_material = make_shared<metal>(albedo, fuzz);
	    			world.add(make_shared<sphere>(center, 0.2, sphere_material));

				} else {

	    			// (3) glass
	    			sphere_material = make_shared<dielectric>(1.5);
	    			world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
    		}
		}
	}

	/* (3) 3개의 오브젝트들:부피가 큰 구들 */
	auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	return world;
	

	/* [BVH Consturction: PPT p.21~32]
	2. 생성한 오브젝트들로 BVH를 만든다. */
	// Constructing BVH
	hittable_list world_bvh;
	world_bvh.add(make_shared<bvh_node>(world, 0, 1));

	return world_bvh;
}
'''
 
  ### Adjustable Variables
  * `image_width`: width of the output image
  * `samples_per_pixel`: the number of samples to produce one pixel value
  * `max_depth`: the maximum number of child rays that can be generated per pixel
  * `nodesNum`: the maximum number of nodes in BVH (determines the size of stack array(`d_nodes`) and bvh array(`d_bvh`))
   
  ### How to Run the Code
  * Compile: `$ make` or `nvcc -g -G --expt-relaxed-constexpr -o RTTNW.out main.cu`
  * Execute: `$ ./RTTNW.out`
  * Debug: `$ cuda-gdb ./RTTNW.out`
    
