#ifndef COMMONFUNCTIONS_H
#define COMMONFUNCTIONS_H

#ifdef __CUDACC__
__device__ 
#endif 
bool hit_sphere(const point3& center, double radius, const ray& r) {
  vec3 oc = r.origin() - center;
  double a = dot(r.direction(), r.direction());
  double b = 2.0 * dot(oc, r.direction());
  double c = dot(oc, oc) - radius*radius;
  double discriminant = b*b - 4*a*c;
  return (discriminant > 0);
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