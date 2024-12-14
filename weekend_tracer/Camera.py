import math
import random

import drjit as dr

from weekend_tracer.common import Bool, Float, Array2f, Array3f, Int32, PCG32, TensorXf
from weekend_tracer.Hittable import HitRecord
from weekend_tracer.HittableList import HittableList
from weekend_tracer.Ray import Ray
from weekend_tracer.Interval import Interval
from weekend_tracer.common import *
from weekend_tracer.Material import MatType
from weekend_tracer.Background import SkyBackground

class Camera:

    DRJIT_STRUCT = {
        'PCG':               PCG32,
        'pixel00_loc':       Array3f,
        'pixel_delta_u':     Array3f,
        'pixel_delta_v':     Array3f,
        'center':            Array3f,
        'defocus_disk_u':    Array3f,
        'defocus_disk_v':    Array3f,
        'image_height':      int,
        'image_width':       int,
        'samples_per_pixel': int,
        'ad_mode':           bool,
        }

    def __init__(self,
                aspect_ratio = 4.0 / 3.0,
                image_width = 10,
                samples_per_pixel = 64) -> None:
        self.aspect_ratio = aspect_ratio            # Ratio of image width over height
        if image_width < 2:
            print("Image width must be at least 2")
        self.image_width = max(2, image_width)             # Rendered image width in pixel count
        self.samples_per_pixel = samples_per_pixel         # Count of random samples for each pixel
        self.max_depth = 10                                # Maximum number of ray bounces into scene
        self.background = SkyBackground()                  # Background color

        self.vfov = 90                                    # Vertical field-of-view in degrees
        self.lookfrom = dr.scalar.Array3f(0.0, 0.0, 0.0)  # Point camera is looking from
        self.lookat = dr.scalar.Array3f(0.0, 0.0, -1.0)   # Point camera is looking at
        self.vup = dr.scalar.Array3f(0.0, 1.0, 0.0)       # Camera-relative "up" direction

        self.defocus_angle = 0.0    # Variation angle of rays through each pixel
        self.focus_dist = 10.0      # Distance from camera lookfrom point to plane of perfect focus

        self.ad_mode = False

        self.PCG = None

    def set_rotation(self, rot: Quaternion4f):
        R = dr.quat_to_matrix(rot)
        self.vup = dr.scalar.Array3f(R[0][1][0], R[1][1][0], R[2][1][0])
        
        look_dir = -dr.scalar.Array3f(R[0][2][0], R[1][2][0], R[2][2][0])
        self.lookat = self.lookfrom + look_dir

    @dr.syntax
    def ray_color(self, r: Ray, max_depth: int, world: HittableList) -> Array3f:
        ray_count     = self.image_height * self.image_width
        mask          = dr.ones(Bool, ray_count)     # Mask for active rays
        out_color     = dr.zeros(Array3f, ray_count)
        cumprod_color = dr.ones(Array3f, ray_count)
        interval      = dr.repeat(Interval(0.001, dr.inf), ray_count)

        ray_depth = 0 if self.ad_mode else UInt32(0)
        while ray_depth < max_depth:
            if dr.hint(mask, mode=COMPILATION_MODE):
                hit, rec = world.hit(r, interval, mask)
                if dr.hint(hit, mode=COMPILATION_MODE):
                    attenuation, color_from_emission, scattered_ray, reflect \
                        = world.scatter(r, rec, self.PCG.next_float32(mask),
                                        self.PCG.next_float32(mask), mask)
                    
                    out_color += color_from_emission * cumprod_color
                    cumprod_color *= dr.select(reflect, attenuation, Array3f(0.0))                     
                    r = dr.select(reflect, scattered_ray, r)
                    mask &= reflect
                else:
                    cumprod_color *= self.background.sample(r.direction)
                    out_color += cumprod_color
                    mask = Bool(False)
            ray_depth += 1

        return out_color


    @dr.syntax
    def render(self, world: HittableList):
        self.initialize()

        ray_count = self.image_height * self.image_width

        img = dr.zeros(Array3f, ray_count)

        # Pixel index
        i = dr.arange(Float, 0, self.image_width)
        j = dr.arange(Float, 0, self.image_height)
        i, j = dr.meshgrid(i, j)

        # Render
        sample = Int32(0)
        spp = dr.opaque(Int32, self.samples_per_pixel)
        while dr.hint(sample < spp, mode=COMPILATION_MODE, max_iterations=-1):
            ray = self.get_ray(i, j)
            img += self.ray_color(ray, self.max_depth, world)
            sample += 1
        img *= self.pixel_samples_scale
        
        # Schedule kernel before raveling
        dr.schedule(img, self.PCG) 

        # Convert the rendering image to a tensor
        img_t = dr.reshape(dtype=TensorXf, value=img, shape=(self.image_height, self.image_width, 3))
        return img_t


    def initialize(self):
        # Chech data format
        assert type(self.lookfrom) is dr.scalar.Array3f, "lookfrom should be dr.scalar.Array3f"
        assert type(self.lookat) is dr.scalar.Array3f, "lookat should be dr.scalar.Array3f"
        assert type(self.vup) is dr.scalar.Array3f, "vup should be dr.scalar.Array3f"
       
        self.image_height = int(self.image_width / self.aspect_ratio)
        if self.image_height < 2:
            print("Image height must be at least 2")
            self.image_height = max(2, self.image_height)
        #print(f"Using width = {self.image_width}, height = {self.image_height}")
        
        # Random number generator (in parallel)
        if self.PCG is None:
            seed = dr.opaque(UInt64, random.getrandbits(64))
            self.PCG = PCG32(size=self.image_height * self.image_width,
                             initstate=seed)

        # Color scale factor for a sum of pixel samples
        self.pixel_samples_scale = 1.0 / self.samples_per_pixel

        self.center = Array3f(self.lookfrom) # Camera centor
        
        # Determine viewport dimensions
        # x-axis is right. y-axis is up. z-axis points out of the screen.
        theta = math.radians(self.vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h * self.focus_dist
        viewport_width = viewport_height * self.image_width / self.image_height

        # Calculate the u, v, w unit basis vectors for the camera coordinate frame.
        w = Array3f(dr.normalize(self.lookfrom - self.lookat))
        u = Array3f(dr.normalize(dr.cross(self.vup, w)))
        v = dr.cross(w, u)

        # Calculate the vectors across the horizontal and down the vertical viewport edges.
        viewport_u = viewport_width * u     # Vector across viewport horizontal edge
        viewport_v = viewport_height * (-v) # Vector down viewport vertical edge

        # Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height

        # Calculate the location of the upper left pixel.
        viewport_upper_left = self.center - self.focus_dist * w \
                            - 0.5 * viewport_u - 0.5 * viewport_v
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)

        # Calculate the camera defocus disk basis vectors
        defocus_radius = self.focus_dist * math.tan(math.radians(self.defocus_angle / 2))
        self.defocus_disk_u = u * defocus_radius
        self.defocus_disk_v = v * defocus_radius

    def get_ray(self, i: Float, j: Float) -> Ray:
        # Construct a camera ray originating from the defocus disk and directed at a randomly
        # sampled point around the pixel location i, j.

        pixel_sample = Array3f(self.pixel00_loc \
            + (i + self.PCG.next_float32() - 0.5) * self.pixel_delta_u \
            + (j + self.PCG.next_float32() - 0.5) * self.pixel_delta_v)

        ray_origin = Array3f(self.center) if self.defocus_angle <= 0 else \
            self.defocus_disk_sample(self.PCG.next_float32(), self.PCG.next_float32())
        ray_direction = dr.normalize(pixel_sample - ray_origin)
        
        return Ray(ray_origin, ray_direction)

    def defocus_disk_sample(self, u1: Float, u2: Float) -> Array3f:
        # Returns a random point in the camera defocus disk.
        p = sample_unit_disk(u1, u2)
        return self.center + (p.x * self.defocus_disk_u + p.y * self.defocus_disk_v)