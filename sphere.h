#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere: public hittable {
    public:
        sphere() {}
        sphere(point3 cen, double r) : center(cen), radius(r) {};

        virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;

    private:
      void fillHitRecord(hit_record& rec, const ray& r, const double& solution ) const;

    public:
        point3 center;
        double radius;
};

void sphere::fillHitRecord(hit_record& rec, const ray& r, const double& solution ) const{
  rec.t = solution;
  rec.p = r.at(rec.t);
  rec.normal = (rec.p - center)/radius;
}

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
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
