# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""List of pre-trained StyleGAN2 networks located on Google Drive."""

import pickle
import dnnlib
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------
# StyleGAN2 Google Drive root: https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7

gdrive_urls = {
    'gdrive:networks/stylegan2-car-config-a.pkl':                           './models/stylegan2-car-config-a.pkl',
    'gdrive:networks/stylegan2-car-config-b.pkl':                           './models/stylegan2-car-config-b.pkl',
    'gdrive:networks/stylegan2-car-config-c.pkl':                           './models/stylegan2-car-config-c.pkl',
    'gdrive:networks/stylegan2-car-config-d.pkl':                           './models/stylegan2-car-config-d.pkl',
    'gdrive:networks/stylegan2-car-config-e.pkl':                           './models/stylegan2-car-config-e.pkl',
    'gdrive:networks/stylegan2-car-config-f.pkl':                           './models/stylegan2-car-config-f.pkl',
    'gdrive:networks/stylegan2-cat-config-a.pkl':                           './models/stylegan2-cat-config-a.pkl',
    'gdrive:networks/stylegan2-cat-config-f.pkl':                           './models/stylegan2-cat-config-f.pkl',
    'gdrive:networks/stylegan2-church-config-a.pkl':                        './models/stylegan2-church-config-a.pkl',
    'gdrive:networks/stylegan2-church-config-f.pkl':                        './models/stylegan2-church-config-f.pkl',
    'gdrive:networks/stylegan2-ffhq-config-a.pkl':                          './models/stylegan2-ffhq-config-a.pkl',
    'gdrive:networks/stylegan2-ffhq-config-b.pkl':                          './models/stylegan2-ffhq-config-b.pkl',
    'gdrive:networks/stylegan2-ffhq-config-c.pkl':                          './models/stylegan2-ffhq-config-c.pkl',
    'gdrive:networks/stylegan2-ffhq-config-d.pkl':                          './models/stylegan2-ffhq-config-d.pkl',
    'gdrive:networks/stylegan2-ffhq-config-e.pkl':                          './models/stylegan2-ffhq-config-e.pkl',
    'gdrive:networks/stylegan2-ffhq-config-f.pkl':                          './models/stylegan2-ffhq-config-f.pkl',
    'gdrive:networks/stylegan2-horse-config-a.pkl':                         './models/stylegan2-horse-config-a.pkl',
    'gdrive:networks/stylegan2-horse-config-f.pkl':                         './models/stylegan2-horse-config-f.pkl',
    'gdrive:networks/network-snapshot-018528.pkl':                          './models/network-snapshot-018528.pkl', # 动漫人物
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl':        './models/table2/stylegan2-car-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl':      './models/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl':        './models/table2/stylegan2-car-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl':      './models/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl':    './models/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl':      './models/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl':        './models/table2/stylegan2-car-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl':      './models/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl':        './models/table2/stylegan2-car-config-e-Gskip-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl':       './models/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl':     './models/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl':       './models/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl':     './models/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl':   './models/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl':     './models/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl':       './models/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl':     './models/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl':       './models/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl',
}

#----------------------------------------------------------------------------

def get_path_or_url(path_or_gdrive_path):
    return gdrive_urls.get(path_or_gdrive_path, path_or_gdrive_path)

#----------------------------------------------------------------------------

_cached_networks = dict()

def load_networks(path_or_gdrive_path):
    path_or_url = get_path_or_url(path_or_gdrive_path)
    if path_or_url in _cached_networks:
        return _cached_networks[path_or_url]

    if dnnlib.util.is_url(path_or_url):
        stream = dnnlib.util.open_url(path_or_url, cache_dir='.stylegan2-cache')
    else:
        stream = open(path_or_url, 'rb')

    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    _cached_networks[path_or_url] = G, D, Gs
    return G, D, Gs

#----------------------------------------------------------------------------
