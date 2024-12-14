# Ray tracer implemented in Python using drjit
# https://raytracing.github.io/books/RayTracingInOneWeekend.html

import random

from PIL import Image
import matplotlib.pyplot as plt
import drjit as dr
from drjit.cuda import Array3f, TensorXf, PCG32, Int32, Float

import weekend_tracer as wt

@dr.syntax
def main():
    dr.set_log_level(dr.LogLevel.Info)
  
    world = wt.HittableList()

    ground_material = wt.Lambertian(wt.Array3f(0.5, 0.5, 0.5))
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f(0, -1000, 0), 1000, ground_material))
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = dr.scalar.Array3f(a + 0.9 * random.random(), 0.2, b + 0.9 * random.random())

            if dr.norm(center - dr.scalar.Array3f(4, 0.2, 0)) > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    albedo_r = random.random() * random.random()
                    albedo_g = random.random() * random.random()
                    albedo_b = random.random() * random.random()
                    lamb_material = wt.Lambertian(Array3f(albedo_r, albedo_g, albedo_b))
                    world.add(wt.Sphere.make_sphere(center, 0.2, lamb_material))
                elif choose_mat < 0.95:
                    # metal
                    albedo_r = 0.5 * (1 + random.random())
                    albedo_g = 0.5 * (1 + random.random())
                    albedo_b = 0.5 * (1 + random.random())
                    fuzz = 0.5 * random.random()
                    metal_material = wt.Metal(
                        wt.TextureBase.make_constant_texture(Array3f(albedo_r, albedo_g, albedo_b)),
                        Float(fuzz))
                    world.add(wt.Sphere.make_sphere(center, 0.2, metal_material))
                else:
                    # glass
                    glass_material = wt.Dielectric(1.5)
                    world.add(wt.Sphere.make_sphere(center, 0.2, glass_material))

    material1 = wt.Dielectric(1.5)
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f(0, 1, 0), 1.0, material1))

    material2 = wt.Lambertian(Array3f(0.4, 0.2, 0.1))
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f(-4, 1, 0), 1.0, material2))

    material3 = wt.Metal(
        wt.TextureBase.make_constant_texture(Array3f(0.7, 0.6, 0.5)), Float(0.0))
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f(4, 1, 0), 1.0, material3))
    world.update()

    cam = wt.Camera()
    cam.aspect_ratio      = 16.0 / 9.0
    cam.image_width       = 1200
    cam.samples_per_pixel = 2048
    cam.max_depth         = 50

    cam.vfov = 20
    cam.lookfrom = dr.scalar.Array3f(13, 2, 3)
    cam.lookat = dr.scalar.Array3f(0, 0, 0)
    cam.vup = dr.scalar.Array3f(0, 1, 0)

    cam.defocus_angle = 0.6
    cam.focus_dist = 10.0

    # Render
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        img_t = cam.render(world)
        img_t = dr.clip(wt.linear_to_gamma(img_t), 0.0, 1.0).numpy()

    execution_time = 0
    for history in dr.kernel_history():
        execution_time += history['execution_time']
        if 'codegen_time' in history:
            execution_time += history['codegen_time']
        print(history)
    print(f"Elapsed time: {execution_time:.2f} ms. FPS: {1000 / execution_time:.2f}")
    
    # Save the image
    img = Image.fromarray((img_t * 255).astype('uint8'))
    img.save('outputs/demo_1.png')

    # Show the image
    plt.axis('off')
    plt.imshow(img_t)
    plt.show()

if __name__ == '__main__':
    main()