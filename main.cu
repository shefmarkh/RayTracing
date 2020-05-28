//CUDA version of main

#include <cstdio>
#include <fstream>
#include <iostream>

#include "vec3.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color ray_color(const ray& r) {
  vec3 unit_direction = unit_vector(r.direction());
  double t = 0.5*(unit_direction.y() + 1.0);
  return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}


__global__ void render(color* fb_color, int* max_x, int* max_y, double *aspect_ratio){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= *max_x || j >= *max_y) return;
  int pixelIndex = j* (*max_x) + i;

  double viewport_height = 2.0;
  double viewport_width = *aspect_ratio * viewport_height;
  double focal_length = 1.0;

  point3 origin = point3(0,0,0);
  vec3 horizontal = vec3(viewport_width, 0, 0);
  vec3 vertical = vec3(0, viewport_height, 0);
  vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

  double u = double(i) / (*max_x-1);
  double v = double(j) / (*max_y-1);
  ray theRay(origin, lower_left_corner + u*horizontal + v*vertical - origin);
  fb_color[pixelIndex] = ray_color(theRay);

}

int main(){

  int tx = 8;
  int ty = 8;

  //Allocate memory on CPU
  const double aspect_ratio_cpu = 16.0 / 9.0;
  int nx_cpu = 384;
  int ny_cpu = nx_cpu/aspect_ratio_cpu;

  int num_pixels = nx_cpu * ny_cpu;
  color fb_color_cpu[num_pixels];

  //allocates memory on the GPU, first argument is a pointer to a pointer to that memory
  size_t fb_color_size = num_pixels*sizeof(color);
  color* fb_color_gpu;
  checkCudaErrors(cudaMalloc((void **)&fb_color_gpu, fb_color_size));

  int *nx_gpu;
  checkCudaErrors(cudaMalloc((void**)&nx_gpu, sizeof(int)));
  checkCudaErrors(cudaMemcpy(nx_gpu, &nx_cpu, sizeof(int),cudaMemcpyHostToDevice));

  int *ny_gpu;
  checkCudaErrors(cudaMalloc((void**)&ny_gpu, sizeof(int)));
  checkCudaErrors(cudaMemcpy(ny_gpu, &ny_cpu, sizeof(int),cudaMemcpyHostToDevice));

  double *aspect_ratio_gpu;
  checkCudaErrors(cudaMalloc((void**)&aspect_ratio_gpu, sizeof(double)));
  checkCudaErrors(cudaMemcpy(aspect_ratio_gpu, &aspect_ratio_cpu, sizeof(double),cudaMemcpyHostToDevice));

  dim3 blocks(nx_cpu/tx+1,ny_cpu/ty+1);
  dim3 threads(tx,ty);
  render<<<blocks,threads>>>(fb_color_gpu,nx_gpu,ny_gpu,aspect_ratio_gpu);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(&fb_color_cpu,fb_color_gpu,fb_color_size,cudaMemcpyDeviceToHost));

  std::cout << "P3\n" << nx_cpu << ' ' << ny_cpu << "\n255\n";

  for (int j = ny_cpu-1; j >= 0; --j) {
    for (int i = 0; i < nx_cpu; ++i) {
      int pixelIndex = j* (nx_cpu) + i;
      color pixel = fb_color_cpu[pixelIndex];
      int ir = static_cast<int>(255.999 * pixel.x());
      int ig = static_cast<int>(255.999 * pixel.y());
      int ib = static_cast<int>(255.999 * pixel.z());
      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }

  //Clean memory on GPU
  checkCudaErrors(cudaFree(nx_gpu));
  checkCudaErrors(cudaFree(ny_gpu));
  checkCudaErrors(cudaFree(fb_color_gpu));

  return 1;
}
