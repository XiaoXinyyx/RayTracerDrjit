# Ray tracer implemented in Python using drjit
# https://raytracing.github.io/books/RayTracingInOneWeekend.html

import time

from PIL import Image
import matplotlib.pyplot as plt
import drjit as dr
from drjit.cuda import Array3f, TensorXf, PCG32, Int32, Float

import weekend_tracer as wt


def main():
    dr.set_log_level(dr.LogLevel.Info)
    #dr.set_flag(dr.JitFlag.Debug, True)

    # Material
    material_ground = wt.Lambertian(Array3f(0.8, 0.8, 0.0))
    material_center = wt.Lambertian(Array3f(0.1, 0.2, 0.5))
    material_left   = wt.Dielectric(1.50)
    material_bubble = wt.Dielectric(1.00 / 1.50)
    material_right  = wt.Metal(
        wt.TextureBase.make_constant_texture(Array3f(0.8, 0.6, 0.2)),
        Float(1.0))

    # World
    world = wt.HittableList()
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f( 0.0, -100.5, -1.0), 100, material_ground))
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f( 0.0,    0.0, -1.2), 0.5, material_center))
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f(-1.0,    0.0, -1.0), 0.5, material_left))
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f(-1.0,    0.0, -1.0), 0.4, material_bubble))
    world.add(wt.Sphere.make_sphere(dr.scalar.Array3f( 1.0,    0.0, -1.0), 0.5, material_right))

    world.update()

    cam = wt.Camera()
    cam.aspect_ratio      = 16.0 / 9.0
    cam.image_width       = 1024
    cam.samples_per_pixel = 1024
    cam.max_depth         = 50

    cam.vfov = 20
    cam.lookfrom = dr.scalar.Array3f(-2, 2, 1)
    cam.lookat = dr.scalar.Array3f(0, 0, -1)
    cam.vup = dr.scalar.Array3f(0, 1, 0)

    # Depth of field
    cam.defocus_angle = 10
    cam.focus_dist = 3.4

    # Render
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        img_t = cam.render(world)
        img_t = dr.clip(wt.linear_to_gamma(img_t), 0.0, 1.0).numpy()
    
    # Show computation graph
    #dot_data = dr.graphviz().view()

    execution_time = 0
    for history in dr.kernel_history():
        # codegen_time is always needed
        execution_time += history['execution_time']
        if 'codegen_time' in history:
            execution_time += history['codegen_time']
        #print(history)
    print(f"Elapsed time: {execution_time:.2f} ms. FPS: {1000 / execution_time:.2f}")

    # Save the image
    img = Image.fromarray((img_t * 255).astype('uint8'))
    img.save('outputs/demo_0.png')

    # Show the image
    plt.axis('off')
    plt.imshow(img_t)
    plt.show()

if __name__ == '__main__':
    main()
