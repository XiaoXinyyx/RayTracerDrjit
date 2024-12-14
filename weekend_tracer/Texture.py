from weekend_tracer.common import *

@dataclass
class TextureBase:
    texture_type:   UInt32  = UInt32(0)
    float_buffer_0: Float   = Float(0.0)
    color_buffer_0: Array3f = Array3f(0.5)
    color_buffer_1: Array3f = Array3f(0.5)

    @staticmethod
    def make_constant_texture(color: Array3f):
        return TextureBase(texture_type=UInt32(0), color_buffer_0=color)
    
    @staticmethod
    def make_checkerboard_texture(grid_count: Float, color1: Array3f, color2: Array3f):
        return TextureBase(
            texture_type=UInt32(1), float_buffer_0=grid_count,
            color_buffer_0=color1, color_buffer_1=color2
        )

    def sample(self, u: Float, v: Float) -> Array3f:
        return Array3f(0)

@dataclass
class ConstantTexture(TextureBase):
    
    @staticmethod
    def sample(self, u: Float, v: Float) -> Array3f:
        color = Array3f(self.color_buffer_0)
        return color

@dataclass
class Checkerboard(TextureBase):

    @staticmethod
    def sample(self, u: Float, v: Float) -> Array3f:
        grid_count = Float(self.float_buffer_0)
        color1 = Array3f(self.color_buffer_0)
        color2 = Array3f(self.color_buffer_1)

        sines = dr.sin(grid_count * dr.pi * u) * dr.sin(grid_count * dr.pi * v)
        rst = dr.select(sines > 0, color1, color2)
        return rst

texture_sample_vcalls = [ConstantTexture.sample, Checkerboard.sample]