import json
import numpy as np
import pdb
import torch
import vtk
from vtk.util import numpy_support

from ray_utils import get_rays, get_ray_directions, get_ndc_rays
import os, imageio, json
from tqdm import tqdm
import torch.nn.functional as F
from run_nerf_helpers import batchify_query, to8b
from lpips import LPIPS
from skimage.metrics import structural_similarity


BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')

class Vortex_Particles(torch.nn.Module):
    def __init__(self, P, T, R, fix_intensity=False):
        super(Vortex_Particles, self).__init__()
        self.P = P
        self.T = T

        self.initialized = False
        self.register_buffer('particle_time_mask', torch.zeros(P, T))   # [P, T]
        self.register_buffer('particle_pos_world', torch.zeros(P, T, 3))   # [P, T, 3]
        self.register_buffer('particle_dir_world', torch.zeros(P, T, 3))   # [P, T, 3]
        self.register_buffer('particle_intensity', torch.zeros(P, T, 1))   # [P, T, 1]
        self.register_buffer('radius', R * (0.5 * torch.rand(P, 1)+1))   # [P, 1]
        # self.radius = torch.nn.Parameter(R * torch.ones(P, 1))   # [P, 1]
        self.particle_intensity_raw = torch.nn.Parameter((10/P * torch.ones(P, 1)).clamp(0, 0.2))   # [P, 1]

        self.register_buffer('particle_time_coef', torch.zeros(P, T))   # [P, T]

    def initialize_with_state_dict(self, state_dict):
        self.load_state_dict(state_dict)
        self.particle_time_mask = self.particle_time_mask.bool()
        self.initialized = True
        print('Load vortex particles from state dict.')

    def initialize_from_generation(self, generated_dict):
        self.particle_time_mask = generated_dict['particle_time_mask']
        self.particle_pos_world = generated_dict['particle_pos_world']
        self.particle_dir_world = generated_dict['particle_dir_world']
        self.particle_time_coef = generated_dict['particle_time_coef']
        self.particle_intensity = generated_dict['particle_intensity'] / 200
        assert self.particle_time_mask.shape == (self.P, self.T)
        assert self.particle_time_coef.shape == (self.P, self.T)
        assert self.particle_pos_world.shape == (self.P, self.T, 3)
        assert self.particle_dir_world.shape == (self.P, self.T, 3)
        self.initialized = True

    def forward(self, coord_3d_world, time_idx, chunk=50):
        """
        args:
            coord_3d_world: [..., 3]
            time_idx: int
        return:
            confinement_field: [..., 3]
        """
        assert self.initialized, 'Vortex_Particles not initialized'
        mask_particle = self.particle_time_mask[:, time_idx]  # [P, T] -> [P]
        particle_pos_world = self.particle_pos_world[:, time_idx]  # [P, T, 3] -> [P, 3]
        particle_dir_world = self.particle_dir_world[:, time_idx]  # [P, T, 3] -> [P, 3]
        particle_intensity = self.particle_intensity_raw.clamp(0, 10) + 1e-8  # [P, 1]
        particle_intensity = particle_intensity.pow(0.5)  # associated with energy
        particle_intensity = particle_intensity * self.particle_intensity[:, time_idx]  # [P, 1]
        radius = torch.relu(self.radius)
        if any(mask_particle):
            confinement_field = compute_confinement_field(particle_pos_world[mask_particle], particle_dir_world[mask_particle],
                                                          particle_intensity[mask_particle], radius[mask_particle], coord_3d_world, chunk=chunk)
        else:
            confinement_field = torch.zeros_like(coord_3d_world)
        return confinement_field

def vort_kernel(x, x_p, r):
    dist = torch.norm(x - x_p, dim=-1, keepdim=True)
    influence = torch.exp(-dist ** 2 / (2 * r ** 2)) / (r**3 * 40000)
    mask = dist < 3*r
    influence = influence * mask.float().detach()
    return influence

def generate_vort_trajectory_curl(time_steps, bbox_model, rx=128, ry=192, rz=128, get_vel_der_fn=None,
                             P=100, N_sample=2**10, den_net=None, **render_kwargs):
    print('Generating vortex trajectory using curl...')
    dt = time_steps[1] - time_steps[0]
    T = len(time_steps)

    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)], indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    # initialize density field
    time_step = torch.ones_like(coord_3d_world[..., :1]) * time_steps[0]
    coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]

    # place empty vortex particles
    all_init_pos = []
    all_init_dir = []
    all_init_int = []
    all_init_time = []

    for i in range(P):
        # sample 4d points
        timesteps = 0.25 + torch.rand(N_sample) * 0.65  # sample from t=0.25 to t=0.9
        sampled_3d_coord_x = 0.25 + torch.rand(N_sample) * 0.5  # [N]
        sampled_3d_coord_y = 0.25 + torch.rand(N_sample) * 0.5  # [N]
        sampled_3d_coord_z = 0.25 + torch.rand(N_sample) * 0.5  # [N]
        sampled_3d_coord = torch.stack([sampled_3d_coord_x, sampled_3d_coord_y, sampled_3d_coord_z], dim=-1)  # [N, 3]
        sampled_3d_coord_world = bbox_model.sim2world(sampled_3d_coord)  # [N, 3]
        sampled_4d_coord_world = torch.cat([sampled_3d_coord_world, timesteps[:, None]], dim=-1)  # [N, 4]

        # compute curl of sampled points
        density = den_net(sampled_4d_coord_world)  # [N, 1]
        density = density.squeeze(-1)  # [N]
        mask = density > 1
        curls = compute_curl_batch(sampled_4d_coord_world, get_vel_der_fn)  # [N, 3]
        curls = curls[mask]
        timesteps = timesteps[mask]
        sampled_3d_coord_world = sampled_3d_coord_world[mask]
        curls_norm = curls.norm(dim=-1)  # [N]
        print(i, 'max curl norm: ', curls_norm.max().item())

        # get points with highest curl norm
        max_idx = curls_norm.argmax()  # get points with highest curl norm
        init_pos = sampled_3d_coord_world[max_idx]  # [3]
        init_dir = curls[max_idx] / curls_norm[max_idx]  # [3]
        init_int = curls_norm[max_idx]  # [1]
        init_time = timesteps[max_idx]  # [1]
        all_init_pos.append(init_pos)
        all_init_dir.append(init_dir)
        all_init_int.append(init_int)
        all_init_time.append(init_time)

    all_init_pos = torch.stack(all_init_pos, dim=0)  # [P, 3]
    all_init_dir = torch.stack(all_init_dir, dim=0)  # [P, 3]
    all_init_int = torch.stack(all_init_int, dim=0)[:, None]  # [P, 1]
    all_init_time = torch.stack(all_init_time, dim=0)[:, None]  # [P, 1]

    # initialize vortex particle position, direction, and when it spawns
    particle_start_timestep = all_init_time  # [P, 1]
    particle_start_timestep = torch.floor(particle_start_timestep * T).expand(-1, T)  # [P, T]
    particle_time_mask = torch.arange(T).unsqueeze(0).expand(P, -1) >= particle_start_timestep  # [P, T]
    particle_time_coef = particle_time_mask.float()  # [P, T]
    for time_coef in particle_time_coef:
        n = 20
        first_idx = time_coef.nonzero()[0]
        try:
            time_coef[first_idx:first_idx+n] = torch.linspace(0, 1, n)
        except:
            time_coef[first_idx:] = torch.linspace(0, 1, T - first_idx.item())
    particle_pos_world = all_init_pos  # [P, 3]
    particle_dir_world = all_init_dir  # [P, 3]
    particle_int_multiplier = torch.ones_like(all_init_int)  # [P, 1]
    particle_int = all_init_int.clone()  # [P, 1]

    all_pos = []
    all_dir = []
    all_int = []

    for i in range(T):
        # update simulation den and source den
        if i > 0:
            coord_4d_world[..., 3] = time_steps[i - 1]  # sample velocity at previous moment
            vel = batchify_query(coord_4d_world, render_kwargs['network_query_fn_vel'])  # [X, Y, Z, 3]

            # advect vortex particles
            mask_to_evolve = particle_time_mask[:, i]
            print('particles to evolve: ', mask_to_evolve.sum().item(), '/', P)
            if any(mask_to_evolve):
                particle_pos_world[mask_to_evolve] = advect_maccormack_particle(particle_pos_world[mask_to_evolve], vel, coord_3d_sim, dt, bbox_model=bbox_model, **render_kwargs)

                # stretch vortex particles
                grad_u, grad_v, grad_w = get_particle_vel_der(particle_pos_world[mask_to_evolve], bbox_model, get_vel_der_fn, time_steps[i - 1])
                particle_dir_world[mask_to_evolve], particle_int_multiplier[mask_to_evolve] = stretch_vortex_particles(particle_dir_world[mask_to_evolve], grad_u, grad_v, grad_w, dt)
                particle_int[mask_to_evolve] = particle_int[mask_to_evolve] * particle_int_multiplier[mask_to_evolve]
                particle_int[particle_int > all_init_int] = all_init_int[particle_int > all_init_int]

        all_pos.append(particle_pos_world.clone())
        all_dir.append(particle_dir_world.clone())
        all_int.append(particle_int.clone())
    particle_pos_world = torch.stack(all_pos, dim=0).permute(1, 0, 2)  # [P, T, 3]
    particle_dir_world = torch.stack(all_dir, dim=0).permute(1, 0, 2)  # [P, T, 3]
    particle_intensity = torch.stack(all_int, dim=0).permute(1, 0, 2)  # [P, T, 1]
    radius = 0.03 * torch.ones(P, 1)[:, None].expand(-1, T, -1)  # [P, T, 1]
    vort_particles = {'particle_time_mask': particle_time_mask,
                          'particle_pos_world': particle_pos_world,
                          'particle_dir_world': particle_dir_world,
                          'particle_intensity': particle_intensity,
                      'particle_time_coef': particle_time_coef,
                          'radius': radius}
    return vort_particles

def stretch_vortex_particles(particle_dir, grad_u, grad_v, grad_w, dt):
    stretch_term = torch.cat([(particle_dir * grad_u).sum(dim=-1, keepdim=True),
                              (particle_dir * grad_v).sum(dim=-1, keepdim=True),
                              (particle_dir * grad_w).sum(dim=-1, keepdim=True), ], dim=-1)  # [P, 3]
    particle_dir = particle_dir + stretch_term * dt
    particle_int = torch.norm(particle_dir, dim=-1, keepdim=True)
    particle_dir = particle_dir / (particle_int + 1e-8)
    return particle_dir, particle_int

def get_particle_vel_der(particle_pos_3d_world, bbox_model, get_vel_der_fn, t):
    time_step = torch.ones_like(particle_pos_3d_world[..., :1]) * t
    particle_pos_4d_world = torch.cat([particle_pos_3d_world, time_step], dim=-1)  # [P, 4]
    particle_pos_4d_world.requires_grad_()
    with torch.enable_grad():
        _, _, _u_x, _u_y, _u_z, _u_t = get_vel_der_fn(particle_pos_4d_world)  # [P, 3], partial der of u,v,w
    jac = torch.stack([_u_x, _u_y, _u_z], dim=-1)  # [P, 3, 3]
    grad_u_world, grad_v_world, grad_w_world = jac[:, 0], jac[:, 1], jac[:, 2]  # [P, 3]
    return grad_u_world, grad_v_world, grad_w_world

def compute_confinement_field(particle_pos_world, particle_dir_world, particle_intensity, radius, coord_3d_world, chunk=50):
    """
    :param particle_pos_world: [P, 3]
    :param particle_dir_world: [P, 3]
    :param particle_intensity: [P, 1]
    :param radius: [P, 1]
    :param coord_3d_world: [..., 3]
    :param chunk: int
    return:
        confinement_field: [..., 3]
    """
    coord_3d_world_shape = coord_3d_world.shape
    assert coord_3d_world_shape[-1] == 3
    coord_3d_world = coord_3d_world.view(-1, 3)  # [N, 3]
    P = particle_pos_world.shape[0]
    confinement_field = torch.zeros_like(coord_3d_world)  # [N, 3]
    for i in range(0, P, chunk):
        location_field = particle_pos_world[i:i+chunk, None, :] - coord_3d_world  # [P, N, 3]
        location_field = location_field / torch.norm(location_field, dim=-1, keepdim=True)  # [P, N, 3]
        vorticity_field = vort_kernel(coord_3d_world, particle_pos_world[i:i+chunk, None, :], r=radius[i:i+chunk, None, :])\
                          * particle_dir_world[i:i+chunk, None, :]  # [P, N, 3]
        confinement_field_each = particle_intensity[i:i+chunk, None, :] \
                                 * torch.cross(location_field, vorticity_field, dim=-1)  # [P, N, 3]
        confinement_field += confinement_field_each.sum(dim=0)  # [N, 3]
    confinement_field = confinement_field.view(coord_3d_world_shape)  # [..., 3]
    return confinement_field

def compute_curl_batch(pts, get_vel_der_fn, chunk=64*96*64):
    pts_shape = pts.shape
    pts = pts.view(-1, pts_shape[-1])  # [N, 3]
    N = pts.shape[0]
    curls = []
    for i in range(0, N, chunk):
        curl = compute_curl(pts[i:i+chunk], get_vel_der_fn)
        curls.append(curl)
    curl = torch.cat(curls, dim=0)  # [N, 3]
    curl = curl.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    return curl

def compute_curl(pts, get_vel_der_fn):
    """
    :param pts: [..., 4]
    :param get_vel_der_fn: function
    :return:
        curl: [..., 3]
    """
    pts_shape = pts.shape
    pts = pts.view(-1, pts_shape[-1])  # [N, 3]
    pts.requires_grad_()
    with torch.enable_grad():
        _, _, _u_x, _u_y, _u_z, _u_t = get_vel_der_fn(pts)  # [N, 3], partial der of u,v,w
    jac = torch.stack([_u_x, _u_y, _u_z], dim=-1)  # [N, 3, 3]
    curl = torch.stack([jac[:, 2, 1] - jac[:, 1, 2],
                        jac[:, 0, 2] - jac[:, 2, 0],
                        jac[:, 1, 0] - jac[:, 0, 1]], dim=-1)  # [N, 3]
    curl = curl.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    return curl

def compute_curl_FD(vel, reverse_z=True):
    X, Y, Z, _ = vel.shape
    curl = torch.zeros_like(vel)

    if reverse_z:
        curl[1:-1, 1:-1, 1:-1, 0] = (vel[1:-1, 2:, 1:-1, 2] - vel[1:-1, :-2, 1:-1, 2]) / 2.0 - (vel[1:-1, 1:-1, :-2, 1] - vel[1:-1, 1:-1, 2:, 1]) / 2.0
        curl[1:-1, 1:-1, 1:-1, 1] = (vel[1:-1, 1:-1, :-2, 0] - vel[1:-1, 1:-1, 2:, 0]) / 2.0 - (vel[2:, 1:-1, 1:-1, 2] - vel[:-2, 1:-1, 1:-1, 2]) / 2.0
        curl[1:-1, 1:-1, 1:-1, 2] = (vel[2:, 1:-1, 1:-1, 1] - vel[:-2, 1:-1, 1:-1, 1]) / 2.0 - (vel[1:-1, 2:, 1:-1, 0] - vel[1:-1, :-2, 1:-1, 0]) / 2.0

    else:
        curl[1:-1, 1:-1, 1:-1, 0] = (vel[1:-1, 2:, 1:-1, 2] - vel[1:-1, :-2, 1:-1, 2]) / 2.0 - (vel[1:-1, 1:-1, 2:, 1] - vel[1:-1, 1:-1, :-2, 1]) / 2.0
        curl[1:-1, 1:-1, 1:-1, 1] = (vel[1:-1, 1:-1, 2:, 0] - vel[1:-1, 1:-1, :-2, 0]) / 2.0 - (vel[2:, 1:-1, 1:-1, 2] - vel[:-2, 1:-1, 1:-1, 2]) / 2.0
        curl[1:-1, 1:-1, 1:-1, 2] = (vel[2:, 1:-1, 1:-1, 1] - vel[:-2, 1:-1, 1:-1, 1]) / 2.0 - (vel[1:-1, 2:, 1:-1, 0] - vel[1:-1, :-2, 1:-1, 0]) / 2.0
    return curl

def compute_grad_FD(scalar_field):
    X, Y, Z, _ = scalar_field.shape
    grad = torch.zeros((X, Y, Z, 3), dtype=scalar_field.dtype, device=scalar_field.device)

    # Compute finite differences and update grad, except for boundaries
    grad[1:-1, :, :, 0] = (scalar_field[2:, :, :, 0] - scalar_field[:-2, :, :, 0]) / 2.0
    grad[:, 1:-1, :, 1] = (scalar_field[:, 2:, :, 0] - scalar_field[:, :-2, :, 0]) / 2.0
    grad[:, :, 1:-1, 2] = (scalar_field[:, :, 2:, 0] - scalar_field[:, :, :-2, 0]) / 2.0

    return grad

def compute_curl_and_grad_batch(pts, get_vel_der_fn, chunk=64*96*64):
    pts_shape = pts.shape
    pts = pts.view(-1, pts_shape[-1])  # [N, 3]
    N = pts.shape[0]
    curls = []
    vorticity_norm_grads = []
    for i in range(0, N, chunk):
        curl, vorticity_norm_grad = compute_curl_and_grad(pts[i:i+chunk], get_vel_der_fn)
        curls.append(curl)
        vorticity_norm_grads.append(vorticity_norm_grad)
    curl = torch.cat(curls, dim=0)  # [N, 3]
    vorticity_norm_grad = torch.cat(vorticity_norm_grads, dim=0)  # [N, 3]
    curl = curl.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    vorticity_norm_grad = vorticity_norm_grad.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    return curl, vorticity_norm_grad

def compute_curl_and_grad(pts, get_vel_der_fn):
    pts_shape = pts.shape
    pts = pts.view(-1, pts_shape[-1])  # [N, 3]
    pts.requires_grad_()
    with torch.enable_grad():
        _, _, _u_x, _u_y, _u_z, _u_t = get_vel_der_fn(pts)  # [N, 3], partial der of u,v,w
        jac = torch.stack([_u_x, _u_y, _u_z], dim=-1)  # [N, 3, 3]
        curl = torch.stack([jac[:, 2, 1] - jac[:, 1, 2],
                            jac[:, 0, 2] - jac[:, 2, 0],
                            jac[:, 1, 0] - jac[:, 0, 1]], dim=-1)  # [N, 3]

        vorticity_norm = torch.norm(curl, dim=-1, keepdim=True)
        vorticity_norm_grad = []

        for j in range(vorticity_norm.shape[1]):
            dy_j_dx = torch.autograd.grad(
                vorticity_norm[:, j],
                pts,
                torch.ones_like(vorticity_norm[:, j], device=vorticity_norm.get_device()),
                retain_graph=True,
                create_graph=True,
            )[0].view(pts.shape[0], -1)
            vorticity_norm_grad.append(dy_j_dx.unsqueeze(1))
    vorticity_norm_grad = torch.cat(vorticity_norm_grad, dim=1)
    curl = curl.view(list(pts_shape[:-1]) + [3])  # [..., 3]
    vorticity_norm_grad = vorticity_norm_grad.view(list(pts_shape[:-1]) + [4])[..., :3]  # [..., 3]

    return curl, vorticity_norm_grad

def run_advect_den(render_poses, hwf, K, time_steps, savedir, gt_imgs, bbox_model, rx=128, ry=192, rz=128,
                   save_fields=False, save_den=False, vort_particles=None, render=None, get_vel_der_fn=None, **render_kwargs):
    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(chunk=512 * 16)
    psnrs = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS().cuda()  # input should be [-1, 1] or [0, 1] (normalize=True)

    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)], indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    # initialize density field
    time_step = torch.ones_like(coord_3d_world[..., :1]) * time_steps[0]
    coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]
    den = batchify_query(coord_4d_world, render_kwargs['network_query_fn'])  # [X, Y, Z, 1]
    den_ori = den
    vel = batchify_query(coord_4d_world, render_kwargs['network_query_fn_vel'])  # [X, Y, Z, 3]
    vel_saved = vel
    bbox_mask = bbox_model.insideMask(coord_3d_world[..., :3].reshape(-1, 3), to_float=False)
    bbox_mask = bbox_mask.reshape(rx, ry, rz)

    source_height = 0.25
    y_start = int(source_height * ry)
    print('y_start: {}'.format(y_start))
    render_kwargs.update(y_start=y_start)
    for i, c2w in enumerate(tqdm(render_poses)):
        # update simulation den and source den
        mask_to_sim = coord_3d_sim[..., 1] > source_height
        if i > 0:
            coord_4d_world[..., 3] = time_steps[i - 1]  # sample velocity at previous moment

            vel = batchify_query(coord_4d_world, render_kwargs['network_query_fn_vel'])  # [X, Y, Z, 3]
            vel_saved = vel
            # advect vortex particles
            if vort_particles is not None:
                confinement_field = vort_particles(coord_3d_world, i)
                print('Vortex energy over velocity: {:.2f}%'.format(torch.norm(confinement_field, dim=-1).pow(2).sum() / torch.norm(vel, dim=-1).pow(2).sum() * 100))
            else:
                confinement_field = torch.zeros_like(vel)

            vel_confined = vel + confinement_field
            den, vel = advect_maccormack(den, vel_confined, coord_3d_sim, dt, bbox_model=bbox_model, **render_kwargs)
            den_ori = batchify_query(coord_4d_world, render_kwargs['network_query_fn'])  # [X, Y, Z, 1]
            # zero grad for coord_4d_world
            # coord_4d_world.grad = None
            # coord_4d_world = coord_4d_world.detach()

            coord_4d_world[..., 3] = time_steps[i]  # source density at current moment
            den[~mask_to_sim] = batchify_query(coord_4d_world[~mask_to_sim], render_kwargs['network_query_fn'])
            den[~bbox_mask] *= 0.0

        if save_fields:
            # save_fields_to_vti(vel.permute(2, 1, 0, 3).detach().cpu().numpy(),
            #                    den.permute(2, 1, 0, 3).detach().cpu().numpy(),
            #                    os.path.join(savedir, 'fields_{:03d}.vti'.format(i)))
            np.save(os.path.join(savedir, 'den_{:03d}.npy'.format(i)), den.permute(2, 1, 0, 3).detach().cpu().numpy())
            np.save(os.path.join(savedir, 'den_ori_{:03d}.npy'.format(i)), den_ori.permute(2, 1, 0, 3).detach().cpu().numpy())
            np.save(os.path.join(savedir, 'vel_{:03d}.npy'.format(i)), vel_saved.permute(2, 1, 0, 3).detach().cpu().numpy())
        if save_den:
            # save_vdb(den[..., 0].detach().cpu().numpy(),
            #          os.path.join(savedir, 'den_{:03d}.vdb'.format(i)))
            # save npy files
            np.save(os.path.join(savedir, 'den_{:03d}.npy'.format(i)), den[..., 0].detach().cpu().numpy())
        rgb, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_grid=True, den_grid=den,
                        **render_kwargs)
        rgb8 = to8b(rgb.detach().cpu().numpy())
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
        imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(i)), rgb8)
        imageio.imsave(os.path.join(savedir, 'gt_{:03d}.png'.format(i)), gt_img8)
    merge_imgs(savedir, prefix='rgb_')
    merge_imgs(savedir, prefix='gt_')

    if gt_imgs is not None:
        avg_psnr = sum(psnrs)/len(psnrs)
        print(f"Avg PSNR over full simulation: ", avg_psnr)
        avg_ssim = sum(ssims)/len(ssims)
        print(f"Avg SSIM over full simulation: ", avg_ssim)
        avg_lpips = sum(lpipss)/len(lpipss)
        print(f"Avg LPIPS over full simulation: ", avg_lpips)
        with open(os.path.join(savedir, "psnrs_{:0.2f}_ssim_{:.2g}_lpips_{:.2g}.json".format(avg_psnr, avg_ssim, avg_lpips)), "w") as fp:
            json.dump(psnrs, fp)


def run_future_pred(render_poses, hwf, K, time_steps, savedir, gt_imgs, bbox_model, rx=128, ry=192, rz=128,
                   save_fields=False, vort_particles=None, project_solver=None, render=None, get_vel_der_fn=None, **render_kwargs):
    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(chunk=512 * 16)
    psnrs = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS().cuda()  # input should be [-1, 1] or [0, 1] (normalize=True)

    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)], indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    # initialize density field
    starting_frame = 89
    n_pred = 30
    time_step = torch.ones_like(coord_3d_world[..., :1]) * time_steps[starting_frame]
    coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]
    den = batchify_query(coord_4d_world, render_kwargs['network_query_fn'])  # [X, Y, Z, 1]
    vel = batchify_query(coord_4d_world, render_kwargs['network_query_fn_vel'])  # [X, Y, Z, 3]

    source_height = 0.25
    y_start = int(source_height * ry)
    print('y_start: {}'.format(y_start))
    render_kwargs.update(y_start=y_start)
    proj_y = render_kwargs['proj_y']
    for idx, i in enumerate(range(starting_frame+1, starting_frame+n_pred+1)):
        c2w = render_poses[0]
        mask_to_sim = coord_3d_sim[..., 1] > source_height
        n_substeps = 1
        if vort_particles is not None:
            confinement_field = vort_particles(coord_3d_world, i)
            print('Vortex energy over velocity: {:.2f}%'.format(
                torch.norm(confinement_field, dim=-1).pow(2).sum() / torch.norm(vel, dim=-1).pow(2).sum() * 100))
        else:
            confinement_field = torch.zeros_like(vel)
        vel_confined = vel + confinement_field

        for _ in range(n_substeps):
            dt_ = dt/n_substeps
            den, _ = advect_SL(den, vel_confined, coord_3d_sim, dt_, bbox_model=bbox_model, **render_kwargs)
            vel, _ = advect_SL(vel, vel, coord_3d_sim, dt_, bbox_model=bbox_model, **render_kwargs)
            vel[..., 2] *= -1  # world coord is left handed, while solver assumes right handed
            vel[:, y_start:y_start + proj_y] = project_solver.Poisson(vel[:, y_start:y_start + proj_y])
            vel[..., 2] *= -1

        try:
            coord_4d_world[..., 3] = time_steps[i]  # sample density source at current moment
            den[~mask_to_sim] = batchify_query(coord_4d_world[~mask_to_sim], render_kwargs['network_query_fn'])
            vel[~mask_to_sim] = batchify_query(coord_4d_world[~mask_to_sim], render_kwargs['network_query_fn_vel'])
        except IndexError:
            pass

        if save_fields:
            save_fields_to_vti(vel.permute(2, 1, 0, 3).cpu().numpy(),
                               den.permute(2, 1, 0, 3).cpu().numpy(),
                               os.path.join(savedir, 'fields_{:03d}.vti'.format(idx)))
            print('Saved fields to {}'.format(os.path.join(savedir, 'fields_{:03d}.vti'.format(idx))))
        rgb, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[0][None], render_grid=True, den_grid=den,
                        **render_kwargs)
        rgb8 = to8b(rgb.cpu().numpy())
        try:
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
        except IndexError:
            pass
        imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(idx)), rgb8)
        imageio.imsave(os.path.join(savedir, 'gt_{:03d}.png'.format(idx)), gt_img8)
    merge_imgs(savedir, framerate=10, prefix='rgb_')
    merge_imgs(savedir, framerate=10, prefix='gt_')

    if gt_imgs is not None:
        try:
            avg_psnr = sum(psnrs) / len(psnrs)
            print(f"Avg PSNR over full simulation: ", avg_psnr)
            avg_ssim = sum(ssims) / len(ssims)
            print(f"Avg SSIM over full simulation: ", avg_ssim)
            avg_lpips = sum(lpipss) / len(lpipss)
            print(f"Avg LPIPS over full simulation: ", avg_lpips)
            with open(os.path.join(savedir, "psnrs_{:0.2f}_ssim_{:.2g}_lpips_{:.2g}.json".format(avg_psnr, avg_ssim, avg_lpips)), "w") as fp:
                json.dump(psnrs, fp)
        except:
            pass

def run_view_synthesis(render_poses, hwf, K, time_steps, savedir, gt_imgs, bbox_model, rx=128, ry=192, rz=128,
                   save_fields=False, vort_particles=None, project_solver=None, render=None, get_vel_der_fn=None, **render_kwargs):
    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(chunk=512 * 16)
    psnrs = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS().cuda()  # input should be [-1, 1] or [0, 1] (normalize=True)

    # initialize density field
    starting_frame = 0
    n_pred = 120
    for idx, i in enumerate(range(starting_frame, starting_frame+n_pred)):
        c2w = render_poses[i]
        rgb, _ = render(H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_den=True,
                        **render_kwargs)
        rgb8 = to8b(rgb.cpu().numpy())
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
        imageio.imsave(os.path.join(savedir, 'rgb_{:03d}.png'.format(idx)), rgb8)
        imageio.imsave(os.path.join(savedir, 'gt_{:03d}.png'.format(idx)), gt_img8)
    merge_imgs(savedir, framerate=10, prefix='rgb_')
    merge_imgs(savedir, framerate=10, prefix='gt_')

    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"Avg PSNR over full simulation: ", avg_psnr)
        avg_ssim = sum(ssims) / len(ssims)
        print(f"Avg SSIM over full simulation: ", avg_ssim)
        avg_lpips = sum(lpipss) / len(lpipss)
        print(f"Avg LPIPS over full simulation: ", avg_lpips)
        with open(os.path.join(savedir, "psnrs_{:0.2f}_ssim_{:.2g}_lpips_{:.2g}.json".format(avg_psnr, avg_ssim, avg_lpips)), "w") as fp:
            json.dump(psnrs, fp)

def advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, RK=2, y_start=48, proj_y=128,
              use_project=False, project_solver=None, bbox_model=None, **kwargs):
    """Advect a scalar quantity using a given velocity field.
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
        RK: int, number of Runge-Kutta steps
        y_start: where to start at y-axis
        proj_y: simulation domain resolution at y-axis
        use_project: whether to use Poisson solver
        project_solver: Poisson solver
        bbox_model: bounding box model
    Returns:
        advected_quantity: [X, Y, Z, 1]
        vel_world: [X, Y, Z, 3]
    """
    if RK == 1:
        vel_world = vel_world_prev.clone()
        vel_world[:, y_start:y_start+proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start+proj_y]) if use_project else vel_world[:, y_start:y_start+proj_y]
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
    elif RK == 2:
        vel_world = vel_world_prev.clone()  # [X, Y, Z, 3]
        vel_world[:, y_start:y_start+proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start+proj_y]) if use_project else vel_world[:, y_start:y_start+proj_y]
        # breakpoint()
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
        coord_3d_sim_midpoint = coord_3d_sim - 0.5 * dt * vel_sim  # midpoint
        midpoint_sampled = coord_3d_sim_midpoint * 2 - 1  # [X, Y, Z, 3]
        vel_sim = F.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], midpoint_sampled.permute(2, 1, 0, 3)[None], align_corners=True, padding_mode='zeros').squeeze(0).permute(3, 2, 1, 0)  # [X, Y, Z, 3]
    else:
        raise NotImplementedError
    backtrace_coord = coord_3d_sim - dt * vel_sim  # [X, Y, Z, 3]
    backtrace_coord_sampled = backtrace_coord * 2 - 1  # ranging [-1, 1]
    q_grid = q_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, C, Z, Y, X] i.e., [N, C, D, H, W]
    q_backtraced = F.grid_sample(q_grid, backtrace_coord_sampled.permute(2, 1, 0, 3)[None, ...], align_corners=True, padding_mode='zeros')  # [N, C, D, H, W]
    q_backtraced = q_backtraced.squeeze(0).permute([3, 2, 1, 0])  # [X, Y, Z, C]
    return q_backtraced, vel_world

def advect_maccormack(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs):
    """
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
    Returns:
        advected_quantity: [X, Y, Z, C]
        vel_world: [X, Y, Z, 3]
    """
    q_grid_next, _ = advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs)
    q_grid_back, vel_world = advect_SL(q_grid_next, vel_world_prev, coord_3d_sim, -dt, **kwargs)
    q_advected = q_grid_next + (q_grid - q_grid_back) / 2
    C = q_advected.shape[-1]
    for i in range(C):
        q_max, q_min = q_grid[..., i].max(), q_grid[..., i].min()
        q_advected[..., i] = q_advected[..., i].clamp_(q_min, q_max)
    return q_advected, vel_world

def advect_SL_particle(particle_pos, vel_world_prev, coord_3d_sim, dt, RK=2, y_start=48, proj_y=128,
              use_project=False, project_solver=None, bbox_model=None, **kwargs):
    """Advect a scalar quantity using a given velocity field.
    Args:
        particle_pos: [N, 3], in world coordinate domain
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
        RK: int, number of Runge-Kutta steps
        y_start: where to start at y-axis
        proj_y: simulation domain resolution at y-axis
        use_project: whether to use Poisson solver
        project_solver: Poisson solver
        bbox_model: bounding box model
    Returns:
        new_particle_pos: [N, 3], in simulation coordinate domain
    """
    if RK == 1:
        vel_world = vel_world_prev.clone()
        vel_world[:, y_start:y_start+proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start+proj_y]) if use_project else vel_world[:, y_start:y_start+proj_y]
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
    elif RK == 2:
        vel_world = vel_world_prev.clone()  # [X, Y, Z, 3]
        vel_world[:, y_start:y_start+proj_y] = project_solver.Poisson(vel_world[:, y_start:y_start+proj_y]) if use_project else vel_world[:, y_start:y_start+proj_y]
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
        coord_3d_sim_midpoint = coord_3d_sim - 0.5 * dt * vel_sim  # midpoint
        midpoint_sampled = coord_3d_sim_midpoint * 2 - 1  # [X, Y, Z, 3]
        vel_sim = F.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], midpoint_sampled.permute(2, 1, 0, 3)[None], align_corners=True).squeeze(0).permute(3, 2, 1, 0)  # [X, Y, Z, 3]
    else:
        raise NotImplementedError
    particle_pos_sampled = bbox_model.world2sim(particle_pos) * 2 - 1  # ranging [-1, 1]
    particle_vel_sim = F.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], particle_pos_sampled[None, None, None], align_corners=True).permute([0, 2, 3, 4, 1]).flatten(0, 3)  # [N, 3]
    particle_pos_new = particle_pos + dt * bbox_model.sim2world_rot(particle_vel_sim)  # [N, 3]
    return particle_pos_new

def advect_maccormack_particle(particle_pos, vel_world_prev, coord_3d_sim, dt, **kwargs):
    """
    Args:
        particle_pos: [N, 3], in world coordinate domain
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
    Returns:
        particle_pos_new: [N, 3], in simulation coordinate domain
    """
    particle_pos_next = advect_SL_particle(particle_pos, vel_world_prev, coord_3d_sim, dt, **kwargs)
    particle_pos_back = advect_SL_particle(particle_pos_next, vel_world_prev, coord_3d_sim, -dt, **kwargs)
    particle_pos_new = particle_pos_next + (particle_pos - particle_pos_back) / 2
    return particle_pos_new


def merge_imgs(save_dir, framerate=30, prefix=''):
    os.system(
        'ffmpeg -hide_banner -loglevel error -y -i {0}/{1}%03d.png -vf palettegen {0}/palette.png'.format(save_dir,
                                                                                                          prefix))
    os.system(
        'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse {1}/_{2}.gif'.format(
            framerate, save_dir, prefix))
    os.system(
        'ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse -vcodec prores {1}/_{2}.mov'.format(
            framerate, save_dir, prefix))


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result


def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays(directions, c2w)
        
        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0]), torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0]))


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0):
    H, W, focal = hwf
    H, W = int(H), int(W)
    
    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays(directions, pose)
        rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3,:3]), -1) # 4.world to 3.target
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape) # 4.world to 3.target
    new_pose = pos_rot + pos_off
    pos_scale = new_pose / (scale_vector) # 3.target to 2.simulation
    return pos_scale

class BBox_Tool(object):
    def __init__(self, smoke_tran_inv, smoke_scale, in_min=[0.15, 0.0, 0.15], in_max=[0.85, 1., 0.85]):
        self.s_w2s = torch.tensor(smoke_tran_inv).expand([4, 4]).float()
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = torch.tensor(smoke_scale.copy()).expand([3]).float()
        self.s_min = torch.Tensor(in_min)
        self.s_max = torch.Tensor(in_max)

    def world2sim(self, pts_world):
        pts_world_homo = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)
        pts_sim_ = torch.matmul(self.s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def world2sim_rot(self, pts_world):
        pts_sim_ = torch.matmul(self.s_w2s[:3, :3], pts_world[..., None]).squeeze(-1)
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def sim2world(self, pts_sim):
        pts_sim_ = pts_sim * self.s_scale
        pts_sim_homo = torch.cat([pts_sim_, torch.ones_like(pts_sim_[..., :1])], dim=-1)
        pts_world = torch.matmul(self.s2w, pts_sim_homo[..., None]).squeeze(-1)[..., :3]
        return pts_world

    def sim2world_rot(self, pts_sim):
        pts_sim_ = pts_sim * self.s_scale
        pts_world = torch.matmul(self.s2w[:3, :3], pts_sim_[..., None]).squeeze(-1)
        return pts_world

    def isInside(self, inputs_pts):
        target_pts = pos_world2smoke(inputs_pts, self.s_w2s, self.s_scale)
        above = torch.logical_and(target_pts[...,0] >= self.s_min[0], target_pts[...,1] >= self.s_min[1] )
        above = torch.logical_and(above, target_pts[...,2] >= self.s_min[2] )
        below = torch.logical_and(target_pts[...,0] <= self.s_max[0], target_pts[...,1] <= self.s_max[1] )
        below = torch.logical_and(below, target_pts[...,2] <= self.s_max[2] )
        outputs = torch.logical_and(below, above)
        return outputs

    def insideMask(self, inputs_pts, to_float=True):
        return self.isInside(inputs_pts).to(torch.float) if to_float else self.isInside(inputs_pts)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count

def write_ply(points, filename, text=True):
    from plyfile import PlyData, PlyElement
    """ input: Nx3 or Nx6, write points to filename as PLY format. """
    if points.shape[1] == 3:
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    elif points.shape[1] == 6:
        if points[:, 3:6].max() <= 1.0:
            points[:, 3:6] *= 255
        points = [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in points]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    else:
        assert False, 'points shape:{}, not valid (2nd dim should be 3 or 6).'.format(points.shape)
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def save_time_varying_fields_to_vti(velocity_field, density_field=None, save_dir='', basename='fields'):
    """
    Save a time-varying velocity field and density field to a series of VTI files.
    args:
        velocity_field: a 5D NumPy array of shape (T, D, H, W, 3) containing the velocity field at each time step
        density_field (optional): a 5D NumPy array of shape (T, D, H, W, 1) containing the density field at each time step
        save_dir: the directory to save the VTI files
        basename: the base name of the VTI files to be saved
    """
    assert velocity_field.ndim == 5 and velocity_field.shape[4] == 3, "Invalid velocity field shape"
    if density_field is not None:
        assert density_field.ndim == 5 and density_field.shape[4] == 1, "Invalid density field shape"
        assert velocity_field.shape[:4] == density_field.shape[:4], "Velocity and density fields must have the same time and grid dimensions"

    T, D, H, W, _ = velocity_field.shape

    for t in range(T):
        save_path = os.path.join(save_dir, f"{basename}_{t:04d}.vti")
        single_time_velocity_field = velocity_field[t, :, :, :]
        single_time_density_field = None if density_field is None else density_field[t, :, :, :]

        save_fields_to_vti(single_time_velocity_field, single_time_density_field, save_path)

def save_fields_to_vti(velocity_field, density_field=None, save_path='fields.vti', vel_name='velocity', den_name='density'):
    D, H, W, _ = velocity_field.shape

    # Create a VTK image data object
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(W, H, D)
    image_data.SetSpacing(1, 1, 1)

    # Convert the velocity NumPy array to a VTK array
    vtk_velocity_array = numpy_support.numpy_to_vtk(velocity_field.reshape(-1, 3), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_velocity_array.SetName(vel_name)
    image_data.GetPointData().SetVectors(vtk_velocity_array)

    # Convert the density NumPy array to a VTK array
    if density_field is not None:
        vtk_density_array = numpy_support.numpy_to_vtk(density_field.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_density_array.SetName(den_name)
        image_data.GetPointData().SetScalars(vtk_density_array)

    # Save the image data object to a VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(image_data)
    writer.Write()

def advect_bfecc(q_grid, coord_3d_sim, coord_4d_world, dt, RK=1, vel_net=None):
    """
    Args:
        q_grid: [X, Y, Z, C]
        coord_3d_sim: [X, Y, Z, 3]
        coord_4d_world: [X, Y, Z, 4]
        dt: float
        RK: int, number of Runge-Kutta steps
        vel_net: function, velocity network
    Returns:
        advected_quantity: [XYZ, C]
    """
    X, Y, Z, _ = coord_3d_sim.shape
    C = q_grid.shape[-1]
    q_grid_next = advect_SL(q_grid, coord_3d_sim.view(-1, 3), coord_4d_world.view(-1, 4), dt, RK=RK, vel_net=vel_net)
    q_grid_back = advect_SL(q_grid_next.view(X, Y, Z, -1), coord_3d_sim.view(-1, 3), coord_4d_world.view(-1, 4), -dt, RK=RK, vel_net=vel_net)
    q_grid_corrected = q_grid + (q_grid - q_grid_back.view(X, Y, Z, -1)) / 2
    q_advected = advect_SL(q_grid_corrected, coord_3d_sim.view(-1, 3), coord_4d_world.view(-1, 4), dt, RK=RK, vel_net=vel_net)
    return q_advected

if __name__=="__main__":
    with open("data/nerf_synthetic/chair/transforms_train.json", "r") as f:
        camera_transforms = json.load(f)
    
    bounding_box = get_bbox3d_for_blenderobj(camera_transforms, 800, 800)
