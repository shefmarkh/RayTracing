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


__global__ void render(color* fb_color, int* max_x, int* max_y){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= *max_x || j >= *max_y) return;
  int pixelIndex = j* (*max_x) + i;
  double r = (double)i/(*max_x -1);
  double g = (double)j/(*max_y -1);
  double b = 0.25;
  fb_color[pixelIndex] = color(r,g,b);
}

int main(){

  int tx = 8;
  int ty = 8;

  //Allocate memory on CPU
  int *nx_cpu = new int(256);
  int *ny_cpu = new int(256);
  float *fb_cpu = new float;

  //allocates memory on the GPU, first argument is a pointer to a pointer to that memory
  int num_pixels = *nx_cpu * (*ny_cpu);

  //allocates memory on the CPU
  size_t fb_color_size = num_pixels*sizeof(color);
  color* fb_color_cpu = (color*)malloc(fb_color_size);

  //allocates memory on the GPU, first argument is a pointer to a pointer to that memory
  color* fb_color_gpu;
  checkCudaErrors(cudaMalloc((void **)&fb_color_gpu, fb_color_size));

  int *nx_gpu;
  checkCudaErrors(cudaMalloc((void**)&nx_gpu, sizeof(int)));
  checkCudaErrors(cudaMemcpy(nx_gpu, nx_cpu, sizeof(int),cudaMemcpyHostToDevice));

  int *ny_gpu;
  checkCudaErrors(cudaMalloc((void**)&ny_gpu, sizeof(int)));
  checkCudaErrors(cudaMemcpy(ny_gpu, ny_cpu, sizeof(int),cudaMemcpyHostToDevice));

  dim3 blocks(*nx_cpu/tx+1,*ny_cpu/ty+1);
  dim3 threads(tx,ty);
  render<<<blocks,threads>>>(fb_color_gpu,nx_gpu,ny_gpu);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(fb_color_cpu,fb_color_gpu,fb_color_size,cudaMemcpyDeviceToHost));

  std::cout << "P3\n" << *nx_cpu << ' ' << *ny_cpu << "\n255\n";

  for (int j = *ny_cpu-1; j >= 0; --j) {
    for (int i = 0; i < *nx_cpu; ++i) {
      int pixelIndex = j* (*nx_cpu) + i;
      color pixel = fb_color_cpu[pixelIndex];
      int ir = static_cast<int>(255.999 * pixel.x());
      int ig = static_cast<int>(255.999 * pixel.y());
      int ib = static_cast<int>(255.999 * pixel.z());
      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }

  //Clean memory on CPU
  delete nx_cpu;
  delete ny_cpu;
  delete fb_cpu;

  return 1;
}
