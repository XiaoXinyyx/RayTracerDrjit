# Ray tracer implemented in Python using drjit
# https://raytracing.github.io/books/RayTracingInOneWeekend.html
import random

from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr

from weekend_tracer import Array3f, TensorXf, PCG32, Int32, Float
import weekend_tracer as wt

def main():
    #dr.set_log_level(dr.LogLevel.Info)
    #dr.set_flag(dr.JitFlag.Debug, True)

    # Random seed
    random.seed(5)

    lr = 1
    epochs = 100

    # Variables to optimize
    right_wall_color = Array3f([random.random() for _ in range(3)])  # Array3f([random.random() for _ in range(3)])
    left_wall_color = Array3f([random.random() for _ in range(3)]) # Array3f([random.random() for _ in range(3)])
    left_box_color = Array3f([random.random() for _ in range(3)])
    right_box_color = Array3f([random.random() for _ in range(3)])
    optimizer = wt.SGD(lr, 
        [right_wall_color, left_wall_color,
         left_box_color, right_box_color])

    # Materials
    right_wall_mat  = wt.Lambertian(right_wall_color)
    left_wall_mat   = wt.Lambertian(left_wall_color)
    left_box_mat = wt.Lambertian(left_box_color)
    right_box_mat = wt.Lambertian(right_box_color)
    white   = wt.Lambertian(Array3f(0.73, 0.73, 0.73))
    light   = wt.DiffuseLight(Array3f(15, 15, 15))

    # Quads
    world = wt.HittableList()
    world.add(wt.Quad.make_quad(Array3f(5.55, 0, 0), Array3f(0, 5.55, 0), Array3f(0, 0, 5.55),
            left_wall_mat))
    world.add(wt.Quad.make_quad(Array3f(0, 0, 0), Array3f(0, 5.55, 0), Array3f(0, 0, 5.55),
            right_wall_mat))
    world.add(wt.Quad.make_quad(Array3f(3.43, 5.54, 3.32), Array3f(-1.30, 0, 0), Array3f(0, 0, -1.05),
            light))
    world.add(wt.Quad.make_quad(Array3f(0, 0, 0), Array3f(5.55, 0, 0), Array3f(0, 0, 5.55),
            white))
    world.add(wt.Quad.make_quad(Array3f(5.55, 5.55, 5.55), Array3f(-5.55, 0, 0), Array3f(0, 0, -5.55),
            white))
    world.add(wt.Quad.make_quad(Array3f(0, 0, 5.55), Array3f(5.55, 0, 0), Array3f(0, 5.55, 0),
            white))
    
    world.add_box(Array3f(0.0), Array3f(1.65, 3.3, 1.65), left_box_mat,
                  rot=dr.rotate(wt.Quaternion4f, Array3f(0, 1, 0), dr.deg2rad(15)).wzyx,
                  translate=Array3f(2.65, 0, 2.95))
    world.add_box(Array3f(0.0), Array3f(1.65, 1.65, 1.65), right_box_mat,
                  rot=dr.rotate(wt.Quaternion4f, Array3f(0, 1, 0), dr.deg2rad(-18)).wzyx,
                  translate=Array3f(1.30, 0, 0.65))
    world.update()

    cam = wt.Camera()
    cam.aspect_ratio      = 1.0
    cam.image_width       = 600 # 2048
    cam.background        = wt.ConstantBackground(Array3f(0.0))

    # Save the initial image
    cam.vfov = 40
    cam.lookfrom = dr.scalar.Array3f(2.78, 2.78, -8.00)
    cam.lookat = dr.scalar.Array3f(2.78, 2.78, 0)
    cam.samples_per_pixel = 1024 * 8
    cam.max_depth = 8
    cam.ad_mode = False
    with dr.suspend_grad():
        img_t = cam.render(world)
        img_t = dr.clip(wt.linear_to_gamma(img_t), 0.0, 1.0).numpy()
    Image.fromarray((img_t * 255).astype(np.uint8)).save(f'outputs/cornell_box_init.png')

    # Setup camera for training
    cam.samples_per_pixel = 512
    cam.max_depth         = 8

    # Turn on AD mode
    cam.ad_mode = True

    # Read reference image
    ref_img = Image.open('outputs/cornell_box.png')
    ref_img = (np.array(ref_img) / 255.0).astype(np.float32)
    ref_img = wt.gamma_to_linear(TensorXf(ref_img))
    dr.eval(ref_img)

    # Log
    loss_log = []

    # Optimization
    progress_bar = tqdm(range(1, epochs+1), desc='Optimizing')
    for i in progress_bar:
        img_t = cam.render(world)
        loss = wt.L1(img_t, ref_img)
        dr.schedule(loss)

        dr.backward(loss)
        optimizer.step()

        left_box_color.xyz = dr.clip(left_box_color, Array3f(0), Array3f(1))
        right_box_color.xyz = dr.clip(right_box_color, Array3f(0), Array3f(1))
        right_wall_color.xyz = dr.clip(right_wall_color, Array3f(0), Array3f(1))
        left_wall_color.xyz = dr.clip(left_wall_color, Array3f(0), Array3f(1))

        optimizer.zero_grad()
        world.update()

        with dr.suspend_grad():
            progress_bar.set_postfix({
                'Loss': f'{loss.numpy():.4f}',
            })
            loss_log.append(loss.numpy())

            # Learning rate decay
            if i % 20 == 0:
                optimizer.lr *= 0.9

            # Save the intermediate image
            if i % 10 == 0 or i == 1:
                output = dr.clip(wt.linear_to_gamma(img_t), 0.0, 1.0)
                Image.fromarray((output.numpy() * 255).astype(np.uint8)).save(f'outputs/cornell_box_{i}.png')

    print('Optimization finished.')
    print("Right wall color:", right_wall_color)
    print("Left wall color:", left_wall_color)
    print("Left box color:", left_box_color)
    print("Right box color:", right_box_color)

    # Output the final image
    cam.samples_per_pixel = 1024 * 8
    cam.max_depth = 8
    cam.ad_mode = False
    with dr.suspend_grad():
        img_t = cam.render(world)
        img_t = dr.clip(wt.linear_to_gamma(img_t), 0.0, 1.0).numpy()
    Image.fromarray((img_t * 255).astype(np.uint8)).save(f'outputs/cornell_box_final.png')

    # Plot the loss
    plt.plot(loss_log)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig('outputs/loss.png')
    plt.show()


if __name__ == '__main__':
    main()
