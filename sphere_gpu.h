#ifndef SPHERE_GPU_H
#define SPHERE_GPU_H

#include "vec3.h"

struct hit_record_gpu {
    point3 p;
    vec3 normal;
    double t;

    bool front_face;

    //set normal based on whether the ray is inside or outside the sphere
    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }

};

class sphere_gpu {
    public:
        sphere_gpu() {}
        sphere_gpu(point3 cen, double r) : center(cen), radius(r) {};

        bool hit(const ray& r, double tmin, double tmax, hit_record_gpu& rec) const;

    private:
      void fillHitRecord(hit_record_gpu& rec, const ray& r, const double& solution ) const;

    public:
        point3 center;
        double radius;
};

void sphere_gpu::fillHitRecord(hit_record_gpu& rec, const ray& r, const double& solution ) const{
  rec.t = solution;
  rec.p = r.at(rec.t);
  vec3 outward_normal = (rec.p - center) / radius;
  rec.set_face_normal(r, outward_normal);
}

bool sphere_gpu::hit(const ray& r, double t_min, double t_max, hit_record_gpu& rec) const {
    vec3 oc = r.origin() - center;
    double a = r.direction().length_squared();
    double half_b = dot(oc, r.direction());
    double c = oc.length_squared() - radius*radius;
    double discriminant = half_b*half_b - a*c;

    if (discriminant > 0) {
        double root = sqrt(discriminant);
        double temp = (-half_b - root)/a;
        if (temp < t_max && temp > t_min) {
            this->fillHitRecord(rec,r,temp);
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp < t_max && temp > t_min) {
            this->fillHitRecord(rec,r,temp);
            return true;
        }
    }
    return false;
}


#endif
