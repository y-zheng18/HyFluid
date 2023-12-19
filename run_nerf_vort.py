import os
import shutil

import imageio
import torch

from utils import *
from tqdm import tqdm, trange
from taichi_encoders.mgpcg import MGPCG_3

from run_nerf_helpers import NeRFSmall, NeRFSmallPotential, save_quiver_plot, get_rays_np, get_rays, get_rays_np_continuous, to8b, batchify_query, sample_bilinear, img2mse, mse2psnr
from radam import RAdam
from load_scalarflow import load_pinf_frame_data
import torch.nn.functional as F
from torch.func import vmap, jacrev

import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=12.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def batchify_rays(rays_flat, chunk=1024 * 64, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def batchify_get_ray_pts_velocity_and_derivitive(pts, chunk=1024 * 64, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, pts.shape[0], chunk):
        ret = get_ray_pts_velocity_and_derivitives(pts[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def PDE_EQs(D_t, D_x, D_y, D_z, U, F, U_t=None, U_x=None, U_y=None, U_z=None):
    eqs = []
    dts = [D_t]
    dxs = [D_x]
    dys = [D_y]
    dzs = [D_z]

    F = torch.cat([torch.zeros_like(F[:, :1]), F], dim=1) * 0 # (N,4)
    u, v, w = U.split(1, dim=-1)  # (N,1)
    F_t, F_x, F_y, F_z = F.split(1, dim=-1)  # (N,1)
    dfs = [F_t, F_x, F_y, F_z]

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim=-1)  # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim=-1)  # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim=-1)  # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim=-1)  # [d_z, u_z, v_z, w_z]
    else:
        dfs = [F_t]

    for dt, dx, dy, dz, df in zip(dts, dxs, dys, dzs, dfs):
        _e = dt + (u * dx + v * dy + w * dz) + df
        eqs += [_e]

    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [dxs[1] + dys[2] + dzs[3]]

    return eqs


def render(H, W, K, rays=None, c2w=None,
           near=0., far=1., time_step=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: float. Focal length of pinhole camera.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    time_step = time_step[:, None, None]  # [N_t, 1, 1]
    N_t = time_step.shape[0]
    N_r = rays.shape[0]
    rays = torch.cat([rays[None].expand(N_t, -1, -1), time_step.expand(-1, N_r, -1)], -1)  # [N_t, n_rays, 7]
    rays = rays.flatten(0, 1)  # [n_time_steps * n_rays, 7]

    # Render and reshape
    all_ret = batchify_rays(rays, **kwargs)
    if 'vel_map' in all_ret:
        k_extract = ['vel_map']
    elif 'rgb_map' in all_ret:
        k_extract = ['rgb_map']
    else:
        k_extract = []
    if N_t == 1:
        for k in k_extract:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = [{k: all_ret[k] for k in all_ret if k not in k_extract}, ]
    return ret_list + ret_dict


def get_velocity_and_derivitives(pts,
                                 **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: float. Focal length of pinhole camera.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    # Render and reshape
    all_ret = batchify_get_ray_pts_velocity_and_derivitive(pts, **kwargs)

    k_extract = ['raw_vel', 'raw_f'] if kwargs['no_vel_der'] else ['raw_vel', 'raw_f', '_u_x', '_u_y', '_u_z', '_u_t']
    ret_list = [all_ret[k] for k in k_extract]
    return ret_list


def render_path(render_poses, hwf, K, gt_imgs=None, savedir=None, time_steps=None, vel_scale=0.01, sim_step=5, **render_kwargs):

    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(dt=dt)
    render_kwargs.update(chunk=512 * 16)
    psnrs = []

    for i, c2w in enumerate(tqdm(render_poses)):
        vel_map, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_vel=True, **render_kwargs)
        vel_map = vel_map.cpu().numpy()  # [H, W, 2]
        # finite difference has issues with boundary because those are not seen during training. Remove those.
        vel_map[0], vel_map[-1], vel_map[:, 0], vel_map[:, -1] = 0, 0, 0, 0

        rgb, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_sim=True, sim_step=sim_step, **render_kwargs)
        rgb8 = to8b(rgb.cpu().numpy())
        if gt_imgs is not None:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            print(f'PSNR: {p:.4g}')
            psnrs.append(p)

        if savedir is not None:
            save_quiver_plot(vel_map[..., 0], vel_map[..., 1], 128, os.path.join(savedir, 'vel_{:03d}.png'.format(i)),
                             scale=vel_scale)
            imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(i)), rgb8)

    if savedir is not None:
        merge_imgs(savedir, prefix='vel_')
        merge_imgs(savedir, prefix='rgb_')

    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"Avg PSNR over {sim_step}-step simulation: ", avg_psnr)
        with open(os.path.join(savedir, "{}step_psnrs_avg{:0.2f}.json".format(sim_step, avg_psnr)), "w") as fp:
            json.dump(psnrs, fp)

    return

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    from taichi_encoders.hash4 import Hash4Encoder
    # embed_fn, input_ch = get_encoder('hashgrid', input_dim=4, num_levels=args.num_levels, base_resolution=args.base_resolution,
    #                                  finest_resolution=args.finest_resolution, log2_hashmap_size=args.log2_hashmap_size,)
    max_res = np.array([args.finest_resolution, args.finest_resolution, args.finest_resolution, args.finest_resolution_t])
    min_res = np.array([args.base_resolution, args.base_resolution, args.base_resolution, args.base_resolution_t])

    embed_fn = Hash4Encoder(max_res=max_res, min_res=min_res, num_scales=args.num_levels,
                            max_params=2 ** args.log2_hashmap_size)
    input_ch = embed_fn.num_scales * 2  # default 2 params per scale
    embedding_params = list(embed_fn.parameters())

    model = NeRFSmall(num_layers=2,
                      hidden_dim=64,
                      geo_feat_dim=15,
                      num_layers_color=2,
                      hidden_dim_color=16,
                      input_ch=input_ch).to(device)
    print(model)
    print('Total number of trainable parameters in model: {}'.format(
        sum([p.numel() for p in model.parameters() if p.requires_grad])))
    print('Total number of parameters in embedding: {}'.format(
        sum([p.numel() for p in embedding_params if p.requires_grad])))
    grad_vars = list(model.parameters())

    network_query_fn = lambda x: model(embed_fn(x))

    # Create optimizer
    optimizer = RAdam([
        {'params': grad_vars, 'weight_decay': 1e-6},
        {'params': embedding_params, 'eps': 1e-15}
    ], lr=args.lrate_den, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, 'den', f) for f in sorted(os.listdir(os.path.join(basedir, expname, 'den'))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': model,
        'embed_fn': embed_fn,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def create_vel_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    from taichi_encoders.hash4 import Hash4Encoder
    # embed_fn, input_ch = get_encoder('hashgrid', input_dim=4, num_levels=args.num_levels, base_resolution=args.base_resolution,
    #                                  finest_resolution=args.finest_resolution, log2_hashmap_size=args.log2_hashmap_size,)
    max_res = np.array([args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v, args.finest_resolution_v_t])
    min_res = np.array([args.base_resolution_v, args.base_resolution_v, args.base_resolution_v, args.base_resolution_v_t])

    embed_fn = Hash4Encoder(max_res=max_res, min_res=min_res, num_scales=args.num_levels,
                            max_params=2 ** args.log2_hashmap_size)
    input_ch = embed_fn.num_scales * 2  # default 2 params per scale
    embedding_params = list(embed_fn.parameters())


    model = NeRFSmallPotential(num_layers=args.vel_num_layers,
                               hidden_dim=64,
                               geo_feat_dim=15,
                               num_layers_color=2,
                               hidden_dim_color=16,
                               input_ch=input_ch).to(device)
    grad_vars = list(model.parameters())
    print(model)
    print('Total number of trainable parameters in model: {}'.format(
        sum([p.numel() for p in model.parameters() if p.requires_grad])))
    print('Total number of parameters in embedding: {}'.format(
        sum([p.numel() for p in embedding_params if p.requires_grad])))

    # network_query_fn = lambda x: model(embed_fn(x))
    def network_vel_fn(x):
        with torch.enable_grad():
            if not args.no_vel_der:
                h = embed_fn(x)
                v, f = model(h)
                return v, f, h
            else:
                v, f = model(embed_fn(x))
                return v, f

    # Create optimizer
    optimizer = torch.optim.RAdam([
        {'params': grad_vars, 'weight_decay': 1e-6},
        {'params': embedding_params, 'eps': 1e-15}
    ], lr=args.lrate, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.vel_path is not None and args.vel_path != 'None':
        ckpts = [args.vel_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
             'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        print(ckpt['vel_network_fn_state_dict'].keys())
        # Load model
        model.load_state_dict(ckpt['vel_network_fn_state_dict'])
        embed_fn.load_state_dict(ckpt['vel_embed_fn_state_dict'])

        optimizer.load_state_dict(ckpt['vel_optimizer_state_dict'])


    ##########################

    render_kwargs_train = {
        'network_vel_fn': network_vel_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,

        'network_fn': model,
        'embed_fn': embed_fn,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw, z_vals, rays_d, learned_rgb=None, render_vel=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([0.1]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    noise = 0.

    alpha = raw2alpha(raw[...,-1] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]  # [N_rays, N_samples]
    if render_vel:
        rgb_map = torch.sum(weights[...,None] * raw[..., :3], -2)  # [N_rays, 3]
    else:
        rgb = torch.ones(3) * (0.6 + torch.tanh(learned_rgb)*0.4)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / (torch.sum(weights, -1) + 1e-10)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)
    depth_map[acc_map < 1e-1] = 0.

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_query_fn,
                N_samples,
                retraw=False,
                network_query_fn_vel=None,
                perturb=0.,
                ret_derivative=True,
                render_vel=False,
                render_sim=False,
                render_grid=False,
                render_den=False,
                den_grid=None,
                sim_step=0,
                dt=None,
                **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    time_step = ray_batch[0, -1]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    pts = torch.cat([pts, time_step * torch.ones((pts.shape[0], pts.shape[1], 1))], -1)  # [..., 4]
    pts_flat = torch.reshape(pts, [-1, 4])
    bbox_mask = bbox_model.insideMask(pts_flat[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True  # in case zero rays are inside the bbox
    pts = pts_flat[bbox_mask]
    ret = {}
    if render_den:
        out_dim = 1
        raw_flat = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat[bbox_mask] = network_query_fn(pts)
        raw = raw_flat.reshape(N_rays, N_samples, out_dim)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, learned_rgb=kwargs['network_fn'].rgb)
        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map}
    elif render_vel:
        out_dim = 3
        raw_flat_vel = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_vel[bbox_mask] = network_query_fn_vel(pts)[0]  # raw_vel
        raw_vel = raw_flat_vel.reshape(N_rays, N_samples, out_dim)
        out_dim = 1
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_den[bbox_mask] = network_query_fn(pts)  # raw_den
        raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)
        raw = torch.cat([raw_vel, raw_den], -1)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, render_vel=render_vel)
        vel_map = rgb_map[..., :2]
        ret['vel_map'] = vel_map
    elif render_sim:
        assert dt is not None and dt > 0, 'dt must be specified a positive number for sim_onestep'
        for i in range(sim_step):
            if pts[0, 3] - dt < 0:
                # print(f'step = {i}, now t < 0, skip simulating this timestep.')
                break

            MacCormack = False  # It marginally (but consistently) improves, but slower. Don't use it until final results.
            if not MacCormack:  # semi-lag for backtracing
                raw_vel = network_query_fn_vel(pts)[0]  # raw_vel
                pts[..., :3] = pts[..., :3] - dt * raw_vel
                pts[..., 3] = pts[..., 3] - dt
            else:  # MacCormack advection
                raw_vel = network_query_fn_vel(pts)[0]
                one_step_back_pts = pts.clone()
                one_step_back_pts[..., :3] = pts[..., :3] - dt * raw_vel
                one_step_back_pts[..., 3] = pts[..., 3] - dt
                returning_vel = network_query_fn_vel(one_step_back_pts)[0]
                returning_pts = one_step_back_pts.clone()
                returning_pts[..., :3] = one_step_back_pts[..., :3] + dt * returning_vel
                returning_pts[..., 3] = one_step_back_pts[..., 3] + dt
                pts_maccorck = one_step_back_pts.clone()
                pts_maccorck[..., :3] = pts_maccorck[..., :3] + (pts[..., :3] - returning_pts[..., :3]) / 2
                pts = pts_maccorck

        # query density
        out_dim = 1
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_den[bbox_mask] = network_query_fn(pts)  # raw_den
        raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_den, z_vals, rays_d, learned_rgb=kwargs['network_fn'].rgb)
        ret['rgb_map'] = rgb_map
    elif render_grid:  # render from a voxel grid
        assert den_grid is not None, 'den_grid must be specified for render_grid.'
        out_dim = 1
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)

        pts_world = pts[..., :3]
        pts_sim = bbox_model.world2sim(pts_world)
        pts_sample = pts_sim * 2 - 1  # ranging [-1, 1]
        den_grid = den_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, 1, Z, Y, X] i.e., [N, 1, D, H, W]
        den_sampled = F.grid_sample(den_grid, pts_sample[None, ..., None, None, :], align_corners=True)

        raw_flat_den[bbox_mask] = den_sampled.reshape(-1, 1)
        raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw_den, z_vals, rays_d, learned_rgb=kwargs['network_fn'].rgb)
        ret['rgb_map'] = rgb_map
    else:  # get density gradient for flow loss
        pts.requires_grad = True
        model = kwargs['network_fn']
        embed_fn = kwargs['embed_fn']
        def g(x):
            return model(x)
        h = embed_fn(pts)
        raw_d = model(h)
        jac = vmap(jacrev(g))(h)
        jac_x = _get_minibatch_jacobian(h, pts)
        jac = jac @ jac_x

        ret = {'raw_d':raw_d, 'pts':pts}
        # jac = _get_minibatch_jacobian(raw_d, pts)
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]
        ret['_d_x'] = _d_x
        ret['_d_y'] = _d_y
        ret['_d_z'] = _d_z
        ret['_d_t'] = _d_t
        out_dim = 1
        raw_flat = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat[bbox_mask] = raw_d
        # print(raw_flat[bbox_mask].max(), raw_flat[bbox_mask].min())
        raw = raw_flat.reshape(N_rays, N_samples, out_dim)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d,
                                                                     learned_rgb=kwargs['network_fn'].rgb)
        ret['rgb_map'] = rgb_map
        ret['disp_map'] = disp_map
        ret['acc_map'] = acc_map
    return ret

def _get_minibatch_jacobian(y, x, threshold=0.05):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)
    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        # mask = torch.abs(y[:, j]) > threshold
        # print(mask.shape, dy_j_dx.shape, mask.sum())
        # dy_j_dx = dy_j_dx * mask[:, None].float()

        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    # mask = torch.abs(y) > threshold
    return jac

def get_ray_pts_velocity_and_derivitives(
        pts,
        network_vel_fn,
        N_samples,
        **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    if kwargs['no_vel_der']:
        with torch.no_grad():
            vel_output, f_output = network_vel_fn(pts)
        ret = {}
        ret['raw_vel'] = vel_output
        ret['raw_f'] = f_output
        return ret

    def g(x):
        return model(x)[0]
    model = kwargs['network_fn']
    embed_fn = kwargs['embed_fn']
    # vel_output, f_output, h = network_vel_fn(pts)
    h = embed_fn(pts)
    vel_output, f_output = model(h)
    ret = {}
    ret['raw_vel'] = vel_output
    ret['raw_f'] = f_output
    if not kwargs['no_vel_der']:

        jac = vmap(jacrev(g))(h)
        jac_x = _get_minibatch_jacobian(h, pts)
        jac = jac @ jac_x
        assert jac.shape == (pts.shape[0], 3, 4)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)
        d = _u_x[:, 0] + _u_y[:, 1] + _u_z[:, 2]
        # _u_x, _u_y, _u_z, _u_t, d = get_vel_derivatives(pts, model, embed_fn)
        ret['raw_vel'] = vel_output
        ret['_u_x'] = _u_x
        ret['_u_y'] = _u_y
        ret['_u_z'] = _u_z
        ret['_u_t'] = _u_t

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_time", type=int, default=1,
                        help='batch size in time')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_den", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay')
    parser.add_argument("--N_iters", type=int, default=5000)
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for density network')
    parser.add_argument("--vel_path", type=str, default=None,
                        help='specific weights npy file to reload for velocity network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')

    parser.add_argument("--train_vel", action='store_true',
                        help='train velocity network')
    parser.add_argument("--run_advect_den", action='store_true',
                        help='Run advect')
    parser.add_argument("--run_future_pred", action='store_true',
                        help='Run future prediction')
    parser.add_argument("--run_view_synthesis", action='store_true',
                        help='Run novel view synthesis (test views)')
    parser.add_argument("--run_without_vort", action='store_true',
                        help='by default we use vortex now, this flag will run without vortex')
    parser.add_argument("--n_particles", type=int, default=100,
                        help='how many particles to use')
    parser.add_argument("--half_res", action='store_true',
                        help='load at half resolution')
    parser.add_argument("--sim_res_x", type=int, default=128,
                        help='simulation resolution along X/width axis')
    parser.add_argument("--sim_res_y", type=int, default=192,
                        help='simulation resolution along Y/height axis')
    parser.add_argument("--sim_res_z", type=int, default=128,
                        help='simulation resolution along Z/depth axis')
    parser.add_argument("--proj_y", type=int, default=128,
                        help='projection resolution along Y/height axis, this must be 2**n')
    parser.add_argument("--y_start", type=int, default=48,
                        help='Within sim_res_y, where to start the projection domain')
    parser.add_argument("--use_project", action='store_true',
                        help='use projection in re-simulation?')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_video", type=int, default=9999999,
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_resolution", type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--finest_resolution_t", type=int, default=256,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--num_levels", type=int, default=16,
                        help='number of levels for hashed embedding')
    parser.add_argument("--base_resolution", type=int, default=16,
                        help='base resolution for hashed embedding')
    parser.add_argument("--base_resolution_t", type=int, default=16,
                        help='base resolution for hashed embedding')
    parser.add_argument("--finest_resolution_v", type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--finest_resolution_v_t", type=int, default=256,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--base_resolution_v", type=int, default=16,
                        help='base resolution for hashed embedding')
    parser.add_argument("--base_resolution_v_t", type=int, default=16,
                        help='base resolution for hashed embedding')
    parser.add_argument("--log2_hashmap_size", type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')
    parser.add_argument("--no_vel_der", action='store_true',
                        help='do not use velocity derivatives-related losses')
    parser.add_argument("--save_fields", action='store_true',
                        help='when run_advect_density, save fields for paraview rendering')
    parser.add_argument("--vel_num_layers", type=int, default=2,
                        help='number of layers in velocity network')
    parser.add_argument("--vel_scale", type=float, default=0.01)
    parser.add_argument("--vel_weight", type=float, default=0.1)
    parser.add_argument("--d_weight", type=float, default=0.1)
    parser.add_argument("--flow_weight", type=float, default=0.001)
    parser.add_argument("--vort_weight", type=float, default=1)
    parser.add_argument("--vort_intensity", type=float, default=20)
    parser.add_argument("--vort_radius", type=float, default=0.01)
    parser.add_argument("--rec_weight", type=float, default=0)
    parser.add_argument("--sim_steps", type=int, default=1)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--save_den", action='store_true', default=False)

    return parser


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))


def train():
    parser = config_parser()
    args = parser.parse_args()
    rx, ry, rz, proj_y, use_project, y_start = args.sim_res_x, args.sim_res_y, args.sim_res_z, args.proj_y, args.use_project, args.y_start
    boundary_types = ti.Matrix([[1, 1], [2, 1], [1, 1]], ti.i32)  # boundaries: 1 means Dirichlet, 2 means Neumann
    project_solver = MGPCG_3(boundary_types=boundary_types, N=[rx, proj_y, rz], base_level=3)

    # Load data
    images_train_, poses_train, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far = \
        load_pinf_frame_data(args.datadir, args.half_res, split='train')
    images_test, poses_test, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far = \
        load_pinf_frame_data(args.datadir, args.half_res, split='test')
    global bbox_model
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale)
    print('Loaded scalarflow', images_train_.shape, render_poses.shape, hwf, args.datadir)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train_vel, render_kwargs_test_vel, start_vel, grad_vars_vel, optimizer_vel = create_vel_nerf(args)
    render_kwargs_train_vel.update(bds_dict)
    render_kwargs_test_vel.update(bds_dict)
    global_step = start
    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    N_timesteps = images_test.shape[0]
    test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
    render_kwargs_test.update(network_query_fn_vel=render_kwargs_test_vel['network_vel_fn'])
    render_kwargs_test.update(save_fields=args.save_fields)
    get_vel_der_fn = lambda pts: get_velocity_and_derivitives(pts, no_vel_der=False, **render_kwargs_test_vel)
    if args.run_without_vort:
        vort_particles = None
    else:
        vort_particles = Vortex_Particles(args.n_particles, images_train_.shape[0], R=args.vort_radius)
        os.makedirs(os.path.join(basedir, expname, 'vort'), exist_ok=True)
        ckpts = [os.path.join(basedir, expname, 'vort', f) for f in sorted(os.listdir(os.path.join(basedir, expname, 'vort'))) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            vort_particles.initialize_with_state_dict(ckpt)
        else:
            print('Initializing vortex particle trajectory.')
            with torch.no_grad():
                vort_particles_dict = generate_vort_trajectory_curl(time_steps=test_timesteps, P=args.n_particles,
                                                          bbox_model=bbox_model, rx=rx, ry=ry, rz=rz,
                                                          get_vel_der_fn=get_vel_der_fn, den_net=render_kwargs_test['network_query_fn'],
                                                          **render_kwargs_test)
            vort_particles.initialize_from_generation(vort_particles_dict)
            particle_pos_sim = bbox_model.world2sim(vort_particles.particle_pos_world)
            write_ply(particle_pos_sim.flatten(0,1).cpu().numpy(), os.path.join(basedir, expname, 'vort', 'vort_traj.ply'))
        optimizer_vort = torch.optim.RAdam(vort_particles.parameters(), lr=args.lrate, betas=(0., 0.9))

    # Short circuit if only rendering out from trained model
    if args.run_view_synthesis:
        print('Run novel view synthesis.')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'run_view_synthesis{:06d}'.format(start))
            shutil.rmtree(testsavedir, ignore_errors=True)
            os.makedirs(testsavedir, exist_ok=True)
            test_view_pose = torch.tensor(poses_test[0])
            test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)

            run_view_synthesis(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir,
                            gt_imgs=images_test, bbox_model=bbox_model, rx=rx, ry=ry, rz=rz, y_start=y_start,
                            proj_y=proj_y, use_project=use_project, project_solver=project_solver, render=render,
                            get_vel_der_fn=get_vel_der_fn, vort_particles=vort_particles,
                            **render_kwargs_test)
            return
    if args.run_advect_den:
        print('Run advect density.')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'run_advect_den_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            test_view_pose = torch.tensor(poses_test[0])
            test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)

            run_advect_den(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir,
                           gt_imgs=images_test, bbox_model=bbox_model, rx=rx, ry=ry, rz=rz, y_start=y_start,
                           proj_y=proj_y, use_project=use_project, project_solver=project_solver, render=render,
                           get_vel_der_fn=get_vel_der_fn, vort_particles=vort_particles, save_den=args.save_den,
                           **render_kwargs_test)
            return
    if args.run_future_pred:
        print('Run future prediction.')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'run_future_pred_{:06d}'.format(start))
            shutil.rmtree(testsavedir, ignore_errors=True)
            os.makedirs(testsavedir, exist_ok=True)
            test_view_pose = torch.tensor(poses_test[0])
            test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)

            run_future_pred(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir,
                           gt_imgs=images_test, bbox_model=bbox_model, rx=rx, ry=ry, rz=rz, y_start=y_start,
                           proj_y=proj_y, use_project=use_project, project_solver=project_solver, render=render,
                           get_vel_der_fn=get_vel_der_fn, vort_particles=vort_particles,
                           **render_kwargs_test)
            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # For random ray batching
    print('get rays')

    rays = []
    ij = []
    for p in poses_train[:, :3, :4]:
        r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, K, p)
        rays.append([r_o, r_d])
        ij.append([i_, j_])
    rays = np.stack(rays, 0)  # [V, ro+rd=2, H, W, 3]
    ij = np.stack(ij, 0)  # [V, 2, H, W]
    images_train = sample_bilinear(images_train_, ij)  # [T, V, H, W, 3]

    rays = np.transpose(rays, [0, 2, 3, 1, 4])  # [V, H, W, ro+rd=2, 3]
    rays = np.reshape(rays, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
    rays = rays.astype(np.float32)
    print('done')
    i_batch = 0

    # Move training data to GPU
    images_train = torch.Tensor(images_train).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]
    # images_train = images_train.reshape((images_train.shape[0], -1, 3))
    T, S, _ = images_train.shape
    rays = torch.Tensor(rays).to(device)
    ray_idxs = torch.randperm(rays.shape[0])

    loss_list = []
    psnr_list = []
    start = start + 1
    vort_loss_meter = AverageMeter()
    loss_meter, psnr_meter = AverageMeter(), AverageMeter()
    flow_loss_meter, scale_meter, norm_meter = AverageMeter(), AverageMeter(), AverageMeter()
    u_loss_meter, v_loss_meter, w_loss_meter, d_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    vel_loss_meter = AverageMeter()
    print('start training: from {} to {}'.format(start, args.N_iters))

    resample_rays = False
    for i in trange(start, args.N_iters + 1):
        # Sample random ray batch
        batch_ray_idx = ray_idxs[i_batch:i_batch + N_rand]
        batch_rays = rays[batch_ray_idx]  # [B, 2, 3]
        batch_rays = torch.transpose(batch_rays, 0, 1)  # [2, B, 3]

        i_batch += N_rand
        # temporal bilinear sampling
        time_idx = torch.randperm(T)[:args.N_time]
        time_step = time_idx / (T - 1) if T > 1 else torch.zeros_like(time_idx)
        frames = images_train[time_idx]  # [N_t, :, 3]
        points = frames[:, batch_ray_idx]  # [N_t, B, 3]
        # points = torch.from_numpy(points).to(device)
        target_s = points.flatten(0, 1)  # [N_t*B, 3]

        if i_batch >= rays.shape[0]:
            print("Shuffle data after an epoch!")
            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = True

        optimizer.zero_grad()

        optimizer_vort.zero_grad()

        extras = render(H, W, K, rays=batch_rays, time_step=time_step,
                        **render_kwargs_train)
        rgb = extras[0]
        extras = extras[1]

        pts = extras['pts']
        if args.no_vel_der:
            raw_vel, raw_f = get_velocity_and_derivitives(pts, no_vel_der=True, **render_kwargs_train_vel)
            _u_x, _u_y, _u_z, _u_t = None, None, None, None
        else:
            raw_vel, raw_f, _u_x, _u_y, _u_z, _u_t = get_velocity_and_derivitives(pts, no_vel_der=False,
                                                                           **render_kwargs_train_vel)
        _d_t = extras['_d_t']
        _d_x = extras['_d_x']
        _d_y = extras['_d_y']
        _d_z = extras['_d_z']

        vel_vort = vort_particles(pts[..., :3], time_idx.item(), chunk=1000)
        raw_vel = raw_vel + vel_vort
        split_nse = PDE_EQs(
            _d_t, _d_x, _d_y, _d_z,
            raw_vel, raw_f, _u_t, _u_x, _u_y, _u_z)
        nse_errors = [mean_squared_error(x, 0.0) for x in split_nse]
        if torch.stack(nse_errors).sum() > 10000:
            # print('dt, dx, dy, dz', _d_t.max(), _d_x.max(), _d_y.max(), _d_z.max())
            print(f'skip large loss {torch.stack(nse_errors).sum():.3g}, timestep={pts[0,3]}')
            continue

        nseloss_fine = 0.0
        split_nse_wei = [args.flow_weight, args.vel_weight, args.vel_weight, args.vel_weight, args.d_weight] if not args.no_vel_der \
            else [args.flow_weight]

        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        loss_meter.update(img_loss.item())
        psnr_meter.update(psnr.item())

        # adhoc
        flow_loss_meter.update(split_nse_wei[0] * nse_errors[0].item())
        scale_meter.update(nse_errors[-1].item())
        norm_meter.update((split_nse_wei[-1] * nse_errors[-1]).item())
        if not args.no_vel_der:
            u_loss_meter.update((split_nse_wei[1] * nse_errors[1]).item())
            v_loss_meter.update((split_nse_wei[2] * nse_errors[2]).item())
            w_loss_meter.update((split_nse_wei[3] * nse_errors[3]).item())
            d_loss_meter.update((split_nse_wei[4] * nse_errors[4]).item())

        for ei, wi in zip(nse_errors, split_nse_wei):
            nseloss_fine = ei * wi + nseloss_fine

        total_intensity = vort_particles.particle_intensity_raw.clamp(0, 1).sum()
        total_radius = torch.relu(vort_particles.radius).sum()
        vort_loss = torch.max(torch.tensor([0]), args.vort_intensity - total_intensity).pow(2)
        vort_loss_meter.update(vort_loss.item())
        vel_loss = nseloss_fine + args.rec_weight * img_loss + args.vort_weight * vort_loss
        vel_loss_meter.update(vel_loss.item())
        vel_loss.backward()

        optimizer_vort.step()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer_vort.param_groups:
            param_group['lr'] = new_lrate
        ################################
        # Rest is logging
        if i % args.i_weights == 0:
            os.makedirs(os.path.join(basedir, expname, 'den'), exist_ok=True)
            path = os.path.join(basedir, expname, 'den', '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'vel_network_fn_state_dict': render_kwargs_train_vel['network_fn'].state_dict(),
                'vel_embed_fn_state_dict': render_kwargs_train_vel['embed_fn'].state_dict(),
                'vel_optimizer_state_dict': optimizer_vel.state_dict(),
            }, path)
            path = os.path.join(basedir, expname, 'vort', '{:06d}.tar'.format(i))
            torch.save(vort_particles.state_dict(), path)

            print('Saved checkpoints at', path)

        if i % args.i_video == 0 or i == 1:
            # Turn on testing mode
            print('Run advect density.')
            with torch.no_grad():
                testsavedir = os.path.join(basedir, expname, 'run_advect_den_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                test_view_pose = torch.tensor(poses_test[0])
                N_timesteps = images_test.shape[0]
                test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
                test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)
                render_kwargs_test.update(network_query_fn_vel=render_kwargs_test_vel['network_vel_fn'])
                run_advect_den(test_view_poses, hwf, K, time_steps=test_timesteps, savedir=testsavedir,
                               gt_imgs=images_test, bbox_model=bbox_model, rx=rx, ry=ry, rz=rz, y_start=y_start,
                               proj_y=proj_y, use_project=use_project, project_solver=project_solver, render=render,
                               get_vel_der_fn=get_vel_der_fn, vort_particles=vort_particles,
                               **render_kwargs_test)
        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} PSNR:{psnr_meter.avg:.4g} Flow Loss: {flow_loss_meter.avg:.2g}, Vort Loss: {vort_loss_meter.avg:.2g}, Total intense: {total_intensity.item()}, Total r: {total_radius}")
            loss_list.append(loss_meter.avg)
            psnr_list.append(psnr_meter.avg)
            loss_psnr = {
                "losses": loss_list,
                "psnr": psnr_list,
            }
            loss_meter.reset()
            psnr_meter.reset()
            flow_loss_meter.reset()
            scale_meter.reset()
            vel_loss_meter.reset()
            norm_meter.reset()
            u_loss_meter.reset()
            v_loss_meter.reset()
            w_loss_meter.reset()
            d_loss_meter.reset()
            vort_loss_meter.reset()

            with open(os.path.join(basedir, expname, "loss_vs_time.json"), "w") as fp:
                json.dump(loss_psnr, fp)
        if resample_rays:
            print("Sampling new rays!")
            if rays is not None:
                del rays
                torch.cuda.empty_cache()
            rays = []
            ij = []
            for p in poses_train[:, :3, :4]:
                r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, K, p)
                rays.append([r_o, r_d])
                ij.append([i_, j_])
            rays = np.stack(rays, 0)  # [V, ro+rd=2, H, W, 3]
            ij = np.stack(ij, 0)  # [V, 2, H, W]
            images_train = sample_bilinear(images_train_, ij)  # [T, V, H, W, 3]
            rays = np.transpose(rays, [0, 2, 3, 1, 4])  # [V, H, W, ro+rd=2, 3]
            rays = np.reshape(rays, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
            rays = rays.astype(np.float32)

            # Move training data to GPU
            images_train = torch.Tensor(images_train).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]

            rays = torch.Tensor(rays).to(device)

            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = False

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    import ipdb
    try:
        train()
    except Exception as e:
        print(e)
        ipdb.post_mortem()
    # train()