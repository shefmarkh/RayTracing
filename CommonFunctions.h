#ifndef COMMONFUNCTIONS_H
#define COMMONFUNCTIONS_H

#include <vector>

#ifdef __CUDACC__
__device__ 
#endif 
void calcDiscriminant(const point3& center, double radius, const ray& r, double (&discriminant)[3]){
  vec3 oc = r.origin() - center;
  double a = dot(r.direction(), r.direction());
  double b = 2.0 * dot(oc, r.direction());
  double c = dot(oc, oc) - radius*radius;
  discriminant[0] = b*b - 4*a*c;
  discriminant[1] = a;
  discriminant[2] = b;
}

#ifdef __CUDACC__
__device__ 
#endif 
bool hit_sphere(const point3& center, double radius, const ray& r) {  
  const unsigned int arraySize = 3;
  double discriminant[arraySize];
  for (unsigned int count = 0; count < arraySize; count++) discriminant[count] = 0.0;
  calcDiscriminant(center,radius,r,discriminant);
  return (discriminant[0] > 0);
}

#ifdef __CUDACC__
__device__ 
#endif 
double calcHitDisciminantOnSphere(const point3& center, double radius, const ray& r) {
  const unsigned int arraySize = 3;
  double discriminant[arraySize];
  for (unsigned int count = 0; count < arraySize; count++) discriminant[count] = 0.0;
  calcDiscriminant(center,radius,r,discriminant);
  if (discriminant[0] < 0) {
    return -1.0;
  } else {
    return (-discriminant[2] - sqrt(discriminant[0]) ) / (2.0*discriminant[1]);
  }
}

#ifdef __CUDACC__
__device__ 
#endif 
color ray_color(const ray& r) {
  if (hit_sphere(point3(0,0,-1), 0.5, r)) return color(1, 0, 0);
  vec3 unit_direction = unit_vector(r.direction());
  double t = 0.5*(unit_direction.y() + 1.0);
  return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

#endif