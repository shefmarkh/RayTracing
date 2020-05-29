#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
    public:
        ray() {}
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        point3 origin() const  { return orig; }
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        vec3 direction() const { return dir; }

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        point3 at(double t) const {
            return orig + t*dir;
        }

    public:
        point3 orig;
        vec3 dir;
};

#endif