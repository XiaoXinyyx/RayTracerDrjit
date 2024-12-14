# Ray tracer implemented in Python using drjit
# https://raytracing.github.io/books/RayTracingInOneWeekend.html

import time

from PIL import Image
import matplotlib.pyplot as plt
import drjit as dr

from weekend_tracer import Array3f, TensorXf, PCG32, Int32, Float
import weekend_tracer as wt

if __name__ == '__main__':
    dr.set_log_level(dr.LogLevel.Info)
    #dr.set_flag(dr.JitFlag.Debug, True)

    print("Start building the scene...")
    start_time = time.time()

    # Materials
    purple  = wt.Lambertian(Array3f(0.292, 0., 0.051) * 2.5)
    white   = wt.Lambertian(Array3f(0.73, 0.73, 0.73))
    black   = wt.Lambertian(Array3f(0.08, 0.08, 0.08))
    light   = wt.DiffuseLight(Array3f(15, 15, 15))
    ground  = wt.Metal(
        wt.TextureBase.make_checkerboard_texture(Float(14),Array3f(0.2),Array3f(0.9)),
        Float(0.05))
    glass   = wt.Dielectric(1.5)

    # Quads
    world = wt.HittableList()
    world.add(wt.Quad.make_quad(
        Array3f(-5, -0.007994, -5), Array3f(10, 0, 0), Array3f(0, 0, 10), ground))
    
    world.add_mesh('assets/model/VCC.obj', purple,
                   scale=Array3f(1),
                   rot = wt.build_quaternion(Array3f(0, 1, 0), dr.deg2rad(-90))
                   )
    world.add_mesh('assets/model/bishop.obj', black,            # front bishop
                   scale=Array3f(0.33),
                   translate=Array3f(-0.646096, 0, 0.883359),)
    world.add_mesh('assets/model/bishop.obj', black,            # left bishop
                   scale=Array3f(0.33),
                   translate=Array3f(0.231037, 0, -1.29858),)
    world.add_mesh('assets/model/bishop.obj', white,
                   scale=Array3f(0.33),
                   translate=Array3f(2.55011, 0, -2.64959),)
    
    world.add_mesh('assets/model/rook.obj', glass, # front rook
                   scale=Array3f(0.33),
                   translate=Array3f(-0.766458, 0, -0.018708),)
    world.add_mesh('assets/model/rook.obj', black,
                   scale=Array3f(0.33),
                   translate=Array3f(1.60148, 0, -1.19013),)
    
    world.add_mesh('assets/model/king.obj', glass,   # front king
                   scale=Array3f(0.33),
                   translate=Array3f(0.772776, 0, 0.525378),)
    world.add_mesh('assets/model/king.obj', white,
                   scale=Array3f(0.33),
                   translate=Array3f(2.78441, 0, -0.990542),)
    world.update()

    print(f"Scene built in {time.time() - start_time:.2f} seconds.")

    cam = wt.Camera()
    cam.aspect_ratio      = 1920 / 1080
    cam.image_width       = 1920
    cam.samples_per_pixel = 2048
    cam.max_depth         = 50
    #cam.background        = wt.SkyBackground()
 
    cam.vfov = 36
    cam.lookfrom = dr.scalar.Array3f(-1.42347, 0.682227, 0.874744)
    cam.lookat = dr.scalar.Array3f(-0.581233, 0.361497, 0.441421)
    cam.vup = dr.scalar.Array3f(0.285198, 0.947171, -0.146731)

    # Depth of field
    cam.defocus_angle = 0.7
    cam.focus_dist = 1.5

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
    img.save('outputs/VCC.png')

    # Show the image
    plt.axis('off')
    plt.imshow(img_t)
    plt.show()

