#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

using std::sqrt;

class vec3 {
    public:
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        vec3() { e[0] = 0.0; e[1] = 0.0; e[2] = 0.0; }
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        vec3(double e0, double e1, double e2)  { e[0] = e0; e[1] = e1; e[2] = e2;};

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        double x() const { return e[0]; }
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        double y() const { return e[1]; }
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        double z() const { return e[2]; }

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        double operator[](int i) const { return e[i]; }
        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        double& operator[](int i) { return e[i]; }

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        vec3& operator*=(const double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        vec3& operator/=(const double t) {
            return *this *= 1/t;
        }

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        double length() const {
            return sqrt(length_squared());
        }

        #ifdef __CUDACC__
        __host__ __device__ 
        #endif 
        double length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

    public:
        double e[3];
};

// Type aliases for vec3
typedef vec3 point3;   // 3D point
typedef vec3 color;    // RGB color

// vec3 Utility Functions

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

#ifdef __CUDACC__
__host__ __device__ 
#endif 
inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif