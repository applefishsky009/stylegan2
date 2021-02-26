# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ryl
# date: 2021/02/25
# https://www.jianshu.com/p/ed74a77524e1
# https://www.jianshu.com/p/b5f3da43733f
import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))
import numpy as np
import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib
import PIL.Image
from tqdm import tqdm
from math import ceil
import pdb
import scipy
import moviepy.editor

# _G = Instantaneous snapshot of the generator. 
# _D = Instantaneous snapshot of the discriminator.
# Gs = Long-term average of the generator
_G, _D, Gs = pretrained_networks.load_networks('./models/network-snapshot-018528.pkl')

# Get tf noise 
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
print(noise_vars)

# 从给定的种子中生成潜伏代码z
def generator_zs_from_seeds(seeds):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        zs.append(z)
    return zs

# 将生成的随机向量z生成图像
# Trunctation psi value needed for the truncation trick
def generate_images(zs, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)

    imgs = []
    for z_idx, z in tqdm(enumerate(zs)):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
    
    # Return array of PIL.Image
    return imgs

def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generator_zs_from_seeds(seeds), truncation_psi)

# 用图片的网格来显示
def creatImageGrid(images, scale=0.25, rows=1):
    w, h = images[0].size
    w = int(w * scale)
    h = int(h * scale)
    height = rows * h
    cols = ceil(len(images) / rows)
    width = cols * w
    canvas = PIL.Image.new('RGBA', (width, height), 'white')
    for i, img in enumerate(images):
        img = img.resize((w, h), PIL.Image.ANTIALIAS)
        canvas.paste(img, (w * (i % cols), h * (i // cols)))
    return canvas

# z向量的两个值之间进行插值
def interpolate(zs, steps):
    out = []
    for i in range(len(zs)- 1):
        for index in range(steps + 1):  # mod
            fraction = index / float(steps)
            out.append(zs[i + 1] * fraction + zs[i] * (1 - fraction))   # 插值
    # pdb.set_trace()
    return out

def generate_3x3():
    # generate 9 random seeds 3x3
    seeds = np.random.randint(10000000, size = 9)
    print(seeds)
    zs = generator_zs_from_seeds(seeds)
    imgs = generate_images(zs, 0.5)
    canvas = creatImageGrid(imgs, rows = 3)
    canvas.save("./results/image_9.png")
    # pdb.set_trace()

def generate_1x8():
    # 0~4,294,967,295 （Seed must be between 0 and 2**32 – 1） 1x8 渐变
    seeds = np.random.randint(10000000, size = 2)
    zs = generator_zs_from_seeds(seeds)
    zs_i = interpolate(zs, 7)   # 产生8张图
    print('zs')
    print(zs[0][0][0])
    print(zs[1][0][0])
    print('zs_i')
    for i in range(7 + 1):
        print(zs_i[i][0][0])
    imgs = generate_images(zs_i, 0.5)
    canvas = creatImageGrid(imgs, rows = 1)
    canvas.save("./results/image_8.png")
    # pdb.set_trace()

def create_image_grid(images_src, grid_size=None):
    assert images_src.ndim == 3 or images_src.ndim == 4
    images = np.zeros((images_src.shape[0], 256, 256, images_src.shape[3]))
    for idx, img in enumerate(images_src):  # 缩放
        img = PIL.Image.fromarray(images_src[0], 'RGB')
        img = img.resize((256, 256), PIL.Image.ANTIALIAS)
        images[idx] = np.array(img)
    num, img_h, img_w, channels = images.shape
    # print(img_h, img_w) # 512 512
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros([grid_h * img_h, grid_w * img_w, channels], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w] = images[idx]
    return grid

# grid_size = [3, 3]
grid_size = [1, 1]
duration_sec = 5
smoothing_sec = 1.0
image_zoom = 1
fps = 15
random_seed = np.random.randint(0, 999)

num_frames = int(np.rint(duration_sec * fps))   # 四舍五入75
random_state = np.random.RandomState(random_seed) # 伪随机数生成器 来自[0,1]均匀分布的随机数序列， 随机数一样，得到的分布就一样

# Generate latent vectors
shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
all_latents = random_state.randn(*shape).astype(np.float32) # 均匀分布
pdb.set_trace()
# 在时间轴上进行高斯模糊
all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * fps] + [0] * len(Gs.input_shape), mode = 'wrap')
all_latents /= np.sqrt(np.mean(np.square(all_latents))) # 75x9x512

# Frame generation func for moviepy
def make_frame(t):
    print('**********')
    print('t', t)
    print('**********')
    frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))  # t在write_gif才会用到, t表示fps的每一个时刻 1/15,2/15.....
    print('frame_idx', frame_idx)
    latents = all_latents[frame_idx]
    fmt = dict(func = tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi = 0.7,
                    randomize_noise=False, output_transform=fmt,
                    minibatch_size = 16)
    grid = create_image_grid(images, grid_size)
    if image_zoom > 1:
        grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], oeder=0)
    if grid.shape[2] == 1:
        grid = grid.repeat(3, 2) # grayscale => RGB
    return grid # 返回H*W*3在t时刻的帧

# https://xin053.github.io/2016/11/05/moviepy视频处理库使用详解/
# https://www.shangmayuan.com/a/721bfbf677bf46e39c141ec9.html
if __name__ == '__main__':
    # generate viedo
    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec) # 秒数5s
    # use this if u want to generate .mp4 video instead
    # video_clip.write_videofile('random_grid_%s.mp4' % random_seed, fps = fps, codec='libx264', bitrate='5M')
    # pdb.set_trace()
    video_clip.write_gif('./results/random_grid_%s.gif' % random_seed, fps = fps, loop = 1) # 15
    pdb.set_trace()












