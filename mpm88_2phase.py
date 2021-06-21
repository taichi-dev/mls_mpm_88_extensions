import taichi as ti

ti.init(arch=ti.gpu)

n_particles = 16384
n_grid = 128
dx = 1 / n_grid
dt = 0.5e-4

p_vol = 1 / n_particles
p_mass0 = 1 / n_particles
gravity = 9.8
bound = 3
cAir = 20
RT = cAir**2
rho_ratio = 100
E = RT * (rho_ratio - 1)

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
type_p = ti.field(ti.i32, n_particles)
color = ti.field(ti.i32, n_particles)
J = ti.field(float, n_particles)
use_C = ti.field(ti.i32, ())

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        p_mass = (1.0 + (rho_ratio - 1.0) * type_p[p]) * p_mass0
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = p_vol * RT * dt * 4 / dx**2
        if type_p[p] == 1:
            stress += -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        mu = 0.1  #粘度
        stressMu = -(C[p] + C[p].transpose()) * mu  #粘性应力矩阵
        stressMu *= dt * p_vol * 4 / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]
                            ]) + p_mass * C[p] + stressMu
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            if use_C[None]:
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def init():
    for i in range(n_particles):
        x1 = ti.random()  #i//(128)/128
        x2 = ti.random()  ##i%(128)/128
        x[i] = [x1 * 0.8 + 0.1, x2 * 0.8 + 0.1]
        v[i] = [0, -1]
        J[i] = 1


@ti.kernel
def change_type():
    for i in range(n_particles):
        if x[i].x < 0.6 and x[i].x > 0.2 and x[i].y < 0.6 and x[i].y > 0.2:
            type_p[i] = 1
            color[i] = 0xCCCCCC
            J[i] = 1


init()
for i in range(n_particles):
    color[i] = 0x068587

gui = ti.GUI('MPM88')
k = 0
while gui.running and k <= 20000:
    for s in range(100):
        substep()
    if k == 50:
        use_C[None] = 1
        change_type()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.2, color=color.to_numpy())
    gui.show()  #use gui.show(f'{k:06d}.png') to save pictures
    k += 1
