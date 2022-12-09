# fpRayTracing 
Floating-point(FP) 연산을 사용하는 기존의 레이 트레이서 코드이다.
레이 트레이서는 아래와 같이 크게 3가지 단계로 구현된다.

## 1. 3차원 월드(world) 생성하기
> 구현이 간단한 '구(sphere)' 물체만으로 구성된 3차원 공간 모델을 직접 생성한다. * 구(sphere) 클래스를 정의한다.
  * 다양한 구 객체들을 생성하여 구 리스트(=3차원 공간)을 생성한다.
  '''c
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
    
