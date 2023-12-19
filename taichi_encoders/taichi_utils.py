import taichi as ti

eps = 1.e-6

@ti.kernel
def copy_to(source: ti.template(), dest: ti.template()):
    for I in ti.grouped(source):
        dest[I] = source[I]

@ti.kernel
def scale_field(a: ti.template(), alpha: float, result: ti.template()):
    for I in ti.grouped(result):
        result[I] = alpha * a[I]

@ti.kernel
def add_fields(f1: ti.template(), f2: ti.template(), dest: ti.template(), multiplier: float):
    for I in ti.grouped(dest):
        dest[I] = f1[I] + multiplier * f2[I]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.kernel
def center_coords_func(pf: ti.template(), dx: float):
    for I in ti.grouped(pf):
        pf[I] = (I+0.5) * dx

@ti.kernel
def x_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j + 0.5, k + 0.5]) * dx

@ti.kernel
def y_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j, k + 0.5]) * dx

@ti.kernel
def z_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j + 0.5, k]) * dx

@ti.func
def sample(qf: ti.template(), u: float, v: float, w: float):
    u_dim, v_dim, w_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim-1))
    j = ti.max(0, ti.min(int(v), v_dim-1))
    k = ti.max(0, ti.min(int(w), w_dim-1))
    return qf[i, j, k]

@ti.kernel
def curl(vf: ti.template(), cf: ti.template(), dx: float):
    inv_dist = 1./(2*dx)
    for i, j, k in cf:
        vr = sample(vf, i+1, j, k)
        vl = sample(vf, i-1, j, k)
        vt = sample(vf, i, j+1, k)
        vb = sample(vf, i, j-1, k)
        vc = sample(vf, i, j, k+1)
        va = sample(vf, i, j, k-1)

        d_vx_dz = inv_dist * (vc.x - va.x)
        d_vx_dy = inv_dist * (vt.x - vb.x)
        
        d_vy_dx = inv_dist * (vr.y - vl.y)
        d_vy_dz = inv_dist * (vc.y - va.y)

        d_vz_dx = inv_dist * (vr.z - vl.z)
        d_vz_dy = inv_dist * (vt.z - vb.z)

        cf[i,j,k][0] = d_vz_dy - d_vy_dz
        cf[i,j,k][1] = d_vx_dz - d_vz_dx
        cf[i,j,k][2] = d_vy_dx - d_vx_dy

@ti.kernel
def get_central_vector(vx: ti.template(), vy: ti.template(), vz: ti.template(), vc: ti.template()):
    for i, j, k in vc:
        vc[i,j,k].x = 0.5 * (vx[i+1, j, k] + vx[i, j, k])
        vc[i,j,k].y = 0.5 * (vy[i, j+1, k] + vy[i, j, k])
        vc[i,j,k].z = 0.5 * (vz[i, j, k+1] + vz[i, j, k])

@ti.kernel
def split_central_vector(vc: ti.template(), vx: ti.template(), vy: ti.template(), vz: ti.template()):
    for i, j, k in vx:
        r = sample(vc, i, j, k)
        l = sample(vc, i-1, j, k)
        vx[i,j,k] = 0.5 * (r.x + l.x)
    for i, j, k in vy:
        t = sample(vc, i, j, k)
        b = sample(vc, i, j-1, k)
        vy[i,j,k] = 0.5 * (t.y + b.y)
    for i, j, k in vz:
        c = sample(vc, i, j, k)
        a = sample(vc, i, j, k-1)
        vz[i,j,k] = 0.5 * (c.z + a.z)

# # # interpolation
@ti.func
def N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 3.0/4.0 - abs_x ** 2
    elif abs_x < 1.5:
        result = 0.5 * (3.0/2.0-abs_x) ** 2
    return result
    
@ti.func
def dN_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2 * abs_x - 3)
    if x < 0.0: # if x < 0 then abs_x is -1 * x
        result *= -1
    return result

@ti.func
def interp_grad_2(vf, p, dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += 1./dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i))
                partial_y += 1./dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i))
                partial_z += 1./dx * (value * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i))
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])