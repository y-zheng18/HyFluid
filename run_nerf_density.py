from utils import *
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import NeRFSmall, to8b, img2mse, mse2psnr, get_rays_np, get_rays, get_rays_np_continuous, sample_bilinear
import torch.nn.functional as F
from radam import RAdam
from load_scalarflow import load_pinf_frame_data
import lpips
import torch

from skimage.metrics import structural_similarity


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
    if N_t == 1:
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = [{k: all_ret[k] for k in all_ret if k not in k_extract}, ]
    return ret_list + ret_dict


def render_path(render_poses, hwf, K, render_kwargs, gt_imgs=None, savedir=None, time_steps=None):
    def merge_imgs(save_dir, framerate=30, prefix=''):
        os.system(
            'ffmpeg -hide_banner -loglevel error -y -i {0}/{1}%03d.png -vf palettegen {0}/palette.png'.format(save_dir,
                                                                                                              prefix))
        os.system(
            'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse {1}/_{2}.gif'.format(
                framerate, save_dir, prefix))
        os.system(
            'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse {1}/_{2}.mp4'.format(
                framerate, save_dir, prefix))


    render_kwargs.update(chunk=512 * 64)
    H, W, focal = hwf
    near, far = render_kwargs['near'], render_kwargs['far']
    if time_steps is None:
        time_steps = torch.ones(render_poses.shape[0], dtype=torch.float32)

    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpipss = []

    lpips_net = lpips.LPIPS().cuda()

    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, depth, acc, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        # normalize depth to [0,1]
        depth = (depth - near) / (far - near)
        depths.append(depth.cpu().numpy())

        if gt_imgs is not None:
            gt_img = torch.tensor(gt_imgs[i].squeeze(), dtype=torch.float32)  # [H, W, 3]
            gt_img8 = to8b(gt_img.cpu().numpy())
            gt_img = gt_img[90:960, 45:540]
            rgb = rgb[90:960, 45:540]
            lpips_value = lpips_net(rgb.permute(2, 0, 1), gt_img.permute(2, 0, 1), normalize=True).item()
            p = -10. * np.log10(np.mean(np.square(rgb.detach().cpu().numpy() - gt_img.cpu().numpy())))
            ssim_value = structural_similarity(gt_img.cpu().numpy(), rgb.cpu().numpy(), data_range=1.0, channel_axis=2)
            lpipss.append(lpips_value)
            psnrs.append(p)
            ssims.append(ssim_value)
            print(f'PSNR: {p:.4g}, SSIM: {ssim_value:.4g}, LPIPS: {lpips_value:.4g}')


        if savedir is not None:
            # save rgb and depth as a figure
            rgb8 = to8b(rgbs[-1])
            imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(i)), rgb8)
            depth = depths[-1]
            colored_depth_map = plt.cm.viridis(depth.squeeze())
            imageio.imwrite(os.path.join(savedir, 'depth_{:03d}.png'.format(i)),
                            (colored_depth_map * 255).astype(np.uint8))

    if savedir is not None:
        merge_imgs(savedir, prefix='rgb_')
        merge_imgs(savedir, prefix='depth_')

    rgbs = np.stack(rgbs, 0)
    depths = np.stack(depths, 0)
    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        avg_lpips = sum(lpipss) / len(lpipss)
        avg_ssim = sum(ssims) / len(ssims)
        print("Avg PSNR over Test set: ", avg_psnr)
        print("Avg LPIPS over Test set: ", avg_lpips)
        print("Avg SSIM over Test set: ", avg_ssim)
        with open(os.path.join(savedir, "test_psnrs_{:0.4f}_lpips_{:0.4f}_ssim_{:0.4f}.json".format(avg_psnr, avg_lpips, avg_ssim)), 'w') as fp:
            json.dump(psnrs, fp)

    return rgbs, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # from encoding import get_encoder
    from taichi_encoders.hash4 import Hash4Encoder
    # embed_fn, input_ch = get_encoder('hashgrid', input_dim=4, num_levels=args.num_levels, base_resolution=args.base_resolution,
    #                                  finest_resolution=args.finest_resolution, log2_hashmap_size=args.log2_hashmap_size,)
    if args.encoder == 'ingp':
        max_res = np.array(
            [args.finest_resolution, args.finest_resolution, args.finest_resolution, args.finest_resolution_t])
        min_res = np.array([args.base_resolution, args.base_resolution, args.base_resolution, args.base_resolution_t])

        embed_fn = Hash4Encoder(max_res=max_res, min_res=min_res, num_scales=args.num_levels,
                                max_params=2 ** args.log2_hashmap_size)
        input_ch = embed_fn.num_scales * 2  # default 2 params per scale
        embedding_params = list(embed_fn.parameters())
    else:
        raise NotImplementedError

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
    ], lr=args.lrate, betas=(0.9, 0.99))
    grad_vars += list(embedding_params)
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])
        # Load optimizer
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    ##########################
    # pdb.set_trace()

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


def raw2outputs(raw, z_vals, rays_d, learned_rgb=None):
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
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.ones(3) * (0.6 + torch.tanh(learned_rgb) * 0.4)
    # rgb = 0.6 + torch.tanh(learned_rgb) * 0.4
    noise = 0.

    alpha = raw2alpha(raw[..., -1] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                      :-1]  # [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / (torch.sum(weights, -1) + 1e-10)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)
    depth_map[acc_map < 1e-1] = 0.

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_query_fn,
                N_samples,
                retraw=False,
                perturb=0.,
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
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    time_step = ray_batch[:, -1]
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1. - t_vals) + far * (t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    pts_time_step = time_step[..., None, None].expand(-1, pts.shape[1], -1)
    pts = torch.cat([pts, pts_time_step], -1)  # [..., 4]
    pts_flat = torch.reshape(pts, [-1, 4])
    out_dim = 1
    raw_flat = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)

    bbox_mask = bbox_model.insideMask(pts_flat[..., :3], to_float=False)
    if bbox_mask.sum() == 0:
        bbox_mask[0] = True  # in case zero rays are inside the bbox
    pts = pts_flat[bbox_mask]

    raw_flat[bbox_mask] = network_query_fn(pts)
    raw = raw_flat.reshape(N_rays, N_samples, out_dim)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d,
                                                                 learned_rgb=kwargs['network_fn'].rgb,)

    ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
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
    parser.add_argument("--encoder", type=str, default='ingp',
                        choices=['ingp', 'plane'])
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_time", type=int, default=1,
                        help='batch size in time')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay')
    parser.add_argument("--N_iters", type=int, default=50000)
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--half_res", action='store_true',
                        help='load at half resolution')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_video", type=int, default=9999999,
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_resolution", type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--finest_resolution_t", type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--num_levels", type=int, default=16,
                        help='number of levels for hashed embedding')
    parser.add_argument("--base_resolution", type=int, default=16,
                        help='base resolution for hashed embedding')
    parser.add_argument("--base_resolution_t", type=int, default=16,
                        help='base resolution for hashed embedding')
    parser.add_argument("--log2_hashmap_size", type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--feats_dim", type=int, default=36,
                        help='feature dimension of kplanes')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    images_train_, poses_train, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far = \
        load_pinf_frame_data(args.datadir, args.half_res, split='train')
    images_test, poses_test, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far = \
        load_pinf_frame_data(args.datadir, args.half_res, split='test')
    global bbox_model
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale)
    render_timesteps = torch.tensor(render_timesteps, dtype=torch.float32)
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
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                test_view_pose = torch.tensor(poses_test[0])
                N_timesteps = images_test.shape[0]
                test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
                test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)
                print(test_view_poses.shape)
                test_view_poses = torch.tensor(poses_train[0]).unsqueeze(0).repeat(N_timesteps, 1, 1)
                print(test_view_poses.shape)
                render_path(test_view_poses, hwf, K, render_kwargs_test, time_steps=test_timesteps, gt_imgs=images_test,
                            savedir=testsavedir)
            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # For random ray batching
    print('get rays')
    rays = []
    ij = []

    # anti-aliasing
    for p in poses_train[:, :3, :4]:
        r_o, r_d, i_, j_ = get_rays_np_continuous(H, W, K, p)
        rays.append([r_o, r_d])
        ij.append([i_, j_])
    rays = np.stack(rays, 0)  # [V, ro+rd=2, H, W, 3]
    ij = np.stack(ij, 0)  # [V, 2, H, W]
    images_train = sample_bilinear(images_train_, ij)   # [T, V, H, W, 3]

    rays = np.transpose(rays, [0, 2, 3, 1, 4])  # [V, H, W, ro+rd=2, 3]
    rays = np.reshape(rays, [-1, 2, 3])  # [VHW, ro+rd=2, 3]
    rays = rays.astype(np.float32)

    print('done')
    i_batch = 0

    # Move training data to GPU
    images_train = torch.Tensor(images_train).to(device).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]
    T, S, _ = images_train.shape
    rays = torch.Tensor(rays).to(device)
    ray_idxs = torch.randperm(rays.shape[0])

    loss_list = []
    psnr_list = []
    start = start + 1
    loss_meter, psnr_meter = AverageMeter(), AverageMeter()
    resample_rays = False
    for i in trange(start, args.N_iters + 1):
        # Sample random ray batch
        batch_ray_idx = ray_idxs[i_batch:i_batch + N_rand]
        batch_rays = rays[batch_ray_idx]  # [B, 2, 3]
        batch_rays = torch.transpose(batch_rays, 0, 1)  # [2, B, 3]

        i_batch += N_rand
        # temporal bilinear sampling
        time_idx = torch.randperm(T)[:args.N_time].float().to(device)  # [N_t]
        time_idx += torch.randn(args.N_time) - 0.5  # -0.5 ~ 0.5
        time_idx_floor = torch.floor(time_idx).long()
        time_idx_ceil = torch.ceil(time_idx).long()
        time_idx_floor = torch.clamp(time_idx_floor, 0, T - 1)
        time_idx_ceil = torch.clamp(time_idx_ceil, 0, T - 1)
        time_idx_residual = time_idx - time_idx_floor.float()
        frames_floor = images_train[time_idx_floor]  # [N_t, VHW, 3]
        frames_ceil = images_train[time_idx_ceil]  # [N_t, VHW, 3]
        frames_interp = frames_floor * (1 - time_idx_residual).unsqueeze(-1) + \
                        frames_ceil * time_idx_residual.unsqueeze(-1)  # [N_t, VHW, 3]
        time_step = time_idx / (T - 1) if T > 1 else torch.zeros_like(time_idx)
        points = frames_interp[:, batch_ray_idx]  # [N_t, B, 3]
        target_s = points.flatten(0, 1)  # [N_t*B, 3]

        if i_batch >= rays.shape[0]:
            print("Shuffle data after an epoch!")
            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = True

        #####  Core optimization loop  #####
        rgb, depth, acc, extras = render(H, W, K, rays=batch_rays, time_step=time_step,
                                         **render_kwargs_train)

        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        loss_meter.update(loss.item())
        psnr_meter.update(psnr.item())

        for param in grad_vars:  # slightly faster than optimizer.zero_grad()
            param.grad = None
        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            testsavedir = os.path.join(basedir, expname, 'spiral_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(render_poses, hwf, K, render_kwargs_test, time_steps=render_timesteps, savedir=testsavedir)

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                test_view_pose = torch.tensor(poses_test[0])
                N_timesteps = images_test.shape[0]
                test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1)
                test_view_poses = test_view_pose.unsqueeze(0).repeat(N_timesteps, 1, 1)
                render_path(test_view_poses, hwf, K, render_kwargs_test, time_steps=test_timesteps, gt_imgs=images_test,
                            savedir=testsavedir)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss_meter.avg:.2g}  PSNR: {psnr_meter.avg:.4g}")
            loss_list.append(loss_meter.avg)
            psnr_list.append(psnr_meter.avg)
            loss_psnr = {
                "losses": loss_list,
                "psnr": psnr_list,
            }
            loss_meter.reset()
            psnr_meter.reset()
            with open(os.path.join(basedir, expname, "loss_vs_time.json"), "w") as fp:
                json.dump(loss_psnr, fp)

        if resample_rays:
            print("Sampling new rays!")
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
            images_train = torch.Tensor(images_train).to(device).flatten(start_dim=1, end_dim=3)  # [T, VHW, 3]
            T, S, _ = images_train.shape
            rays = torch.Tensor(rays).to(device)

            ray_idxs = torch.randperm(rays.shape[0])
            i_batch = 0
            resample_rays = False
        global_step += 1


if __name__ == '__main__':
    import taichi as ti

    ti.init(arch=ti.cuda, device_memory_GB=6.0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    import ipdb

    try:
        train()
    except Exception as e:
        print(e)
        ipdb.post_mortem()

