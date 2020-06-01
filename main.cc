#include <iostream>

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "CommonFunctions.h"
#include "sphere.h"
#include "hittable_list.h"

int main() {
    
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 384;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    hittable_list world;
    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    for (int j = image_height-1; j >= 0; --j) {
          for (int i = 0; i < image_width; ++i) {
          auto u = double(i) / (image_width-1);
          auto v = double(j) / (image_height-1);
          ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);
          color pixel_color = ray_color(r, world);
          write_color(std::cout, pixel_color);
        }
    }

}
