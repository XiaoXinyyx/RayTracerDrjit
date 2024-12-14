# Ray tracer implemented in Python using drjit
# https://raytracing.github.io/books/RayTracingInOneWeekend.html

import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import drjit as dr

from weekend_tracer import Array3f, TensorXf, PCG32, Int32, Float
import weekend_tracer as wt

if __name__ == '__main__':
    #dr.set_log_level(dr.LogLevel.Info)
    #dr.set_flag(dr.JitFlag.Debug, True)

    # Materials
    red     = wt.Lambertian(Array3f(0.65, 0.05, 0.05))
    green   = wt.Lambertian(Array3f(0.12, 0.45, 0.15))
    white   = wt.Lambertian(Array3f(0.73, 0.73, 0.73))
    light   = wt.DiffuseLight(Array3f(15, 15, 15))
    metal   = wt.Metal(wt.TextureBase.make_constant_texture(Array3f(0.8, 0.85, 0.88)),
                       Float(0.0))
    glass   = wt.Dielectric(1.5)

    # Quads
    world = wt.HittableList()
    world.add(wt.Quad.make_quad(Array3f(5.55, 0, 0), Array3f(0, 5.55, 0), Array3f(0, 0, 5.55), green))
    world.add(wt.Quad.make_quad(Array3f(0, 0, 0), Array3f(0, 5.55, 0), Array3f(0, 0, 5.55), red))
    world.add(wt.Quad.make_quad(Array3f(3.43, 5.54, 3.32), Array3f(-1.30, 0, 0), Array3f(0, 0, -1.05), light))
    world.add(wt.Quad.make_quad(Array3f(0, 0, 0), Array3f(5.55, 0, 0), Array3f(0, 0, 5.55), white))
    world.add(wt.Quad.make_quad(Array3f(5.55, 5.55, 5.55), Array3f(-5.55, 0, 0), Array3f(0, 0, -5.55), white))
    world.add(wt.Quad.make_quad(Array3f(0, 0, 5.55), Array3f(5.55, 0, 0), Array3f(0, 5.55, 0), white))
    
    world.add_box(Array3f(0.0), Array3f(1.65, 3.3, 1.65), white,
                  rot=wt.build_quaternion(Array3f(0, 1, 0), dr.deg2rad(15)),
                  translate=Array3f(2.65, 0, 2.95))
    world.add_box(Array3f(0.0), Array3f(1.65, 1.65, 1.65), white,
                  rot=wt.build_quaternion(Array3f(0, 1, 0), dr.deg2rad(-18)),
                  translate=Array3f(1.30, 0, 0.65))
    #world.add(wt.Sphere.make_sphere(Array3f(1.9, 0.9, 1.9), 0.9, glass))
    world.update()

    cam = wt.Camera()
    cam.aspect_ratio      = 1.0
    cam.image_width       = 600 # 600
    cam.samples_per_pixel = 4096
    cam.max_depth         = 50
    cam.background        = wt.ConstantBackground(Array3f(0.0))

    cam.vfov = 40
    cam.lookfrom = dr.scalar.Array3f(2.78, 2.78, -8.00)
    cam.lookat = dr.scalar.Array3f(2.78, 2.78, 0)
    cam.vup = dr.scalar.Array3f(0, 1, 0)

    # Depth of field
    cam.defocus_angle = 0
    #cam.focus_dist = 3.4

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

    # Write to png
    img = Image.fromarray((img_t * 255).astype('uint8'))
    img.save('outputs/output.png')

    # Show the image
    plt.axis('off')
    plt.imshow(img_t)
    plt.show()

