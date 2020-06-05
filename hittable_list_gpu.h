#ifndef HITTABLE_LIST_GPU_H
#define HITTABLE_LIST_GPU_H

#include "sphere_gpu.h"

#include <memory>
#include <vector>

template <unsigned int ARRAYSIZE> class hittable_list_gpu {
    public:
        hittable_list_gpu() {}
        ~hittable_list_gpu() { 
          for (int counter = 0; counter < ARRAYSIZE; counter++) {
            if (m_objects[counter]) delete m_objects[counter];
          }          
        }
        
        void add(sphere_gpu* object, const unsigned int& index) { if (index < ARRAYSIZE) m_objects[index] = object; }

        bool hit(const ray& r, double t_min, double t_max, hit_record_gpu& rec) const{
          hit_record_gpu temp_rec;
          bool hit_anything = false;
          double closest_so_far = t_max;

          for (unsigned int counter = 0; counter < ARRAYSIZE; counter++){
            if ( m_objects[counter]->hit(r, t_min, closest_so_far, temp_rec)) {
              hit_anything = true;
              closest_so_far = temp_rec.t;
              rec = temp_rec;
            }
          }

          return hit_anything;
        }

    private:               
        sphere_gpu* m_objects[ARRAYSIZE];
};

#endif
