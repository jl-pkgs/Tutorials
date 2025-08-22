

using Gridap, Gridap.Geometry, Gridap.Fields, GridapGmsh
λ = 532      # Wavelength (nm)
L = 600      # Width of the numerical cell (excluding PML) (nm)
h1 = 600     # Height of the air region below the source (nm)
h2 = 200     # Height of the air region above the source (nm)
dpml = 300   # Thickness of the PML (nm)

n_metal = 0.054 + 3.429im # Silver refractive index at λ = 532 nm
n_air = 1    # Air refractive index
μ = 1        # Magnetic permeability
k = 2*π/λ    # Wavenumber (nm^-1)

model = GmshDiscreteModel("../models/RecCirGeometry.msh")

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V = TestFESpace(model, reffe, dirichlet_tags = ["DirichletEdges", "DirichletNodes"], vector_type = Vector{ComplexF64})
U = V   # mathematically equivalent to TrialFESpace(V,0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

Γ_s = BoundaryTriangulation(model; tags = ["Source"]) # Source line
dΓ_s = Measure(Γ_s, degree)

Ω_d = Triangulation(model, tags="Design")
dΩ_d = Measure(Ω_d, degree)

Ω_c = Triangulation(model, tags="Center")
dΩ_c = Measure(Ω_c, degree)

p_reffe = ReferenceFE(lagrangian, Float64, 0)
Q = TestFESpace(Ω_d, p_reffe, vector_type = Vector{Float64})
P = Q
np = num_free_dofs(P) # Number of cells in design region (number of design parameters)

pf_reffe = ReferenceFE(lagrangian, Float64, 1)
Qf = TestFESpace(Ω_d, pf_reffe, vector_type = Vector{Float64})
Pf = Qf

fem_params = (; V, U, Q, P, Qf, Pf, np, Ω, dΩ, dΩ_d, dΩ_c, dΓ_s)

R = 1e-10
LHp=(L/2, h1+h2)   # Start of PML for x,y > 0
LHn=(L/2, 0)       # Start of PML for x,y < 0
phys_params = (; k, n_metal, n_air, μ, R, dpml, LHp, LHn)

function s_PML(x; phys_params)
    σ = -3 / 4 * log(phys_params.R) / phys_params.dpml / phys_params.n_air
    xf = Tuple(x)
    u = @. ifelse(xf > 0 , xf - phys_params.LHp, - xf - phys_params.LHn)
    return @. ifelse(u > 0,  1 + (1im * σ / phys_params.k) * (u / phys_params.dpml)^2, $(1.0+0im))
end

function ds_PML(x; phys_params)
    σ = -3 / 4 * log(phys_params.R) / phys_params.dpml / phys_params.n_air
    xf = Tuple(x)
    u = @. ifelse(xf > 0 , xf - phys_params.LHp, - xf - phys_params.LHn)
    ds = @. ifelse(u > 0, (2im * σ / phys_params.k) * (1 / phys_params.dpml)^2 * u, $(0.0+0im))
    return ds.*sign.(xf)
end

struct Λ{PT} <: Function
    phys_params::PT
end

function (Λf::Λ)(x)
    s_x,s_y = s_PML(x; Λf.phys_params)
    return VectorValue(1/s_x, 1/s_y)
end

Fields.∇(Λf::Λ) = x -> TensorValue{2, 2, ComplexF64}(-(Λf(x)[1])^2 * ds_PML(x; Λf.phys_params)[1], 0, 0, -(Λf(x)[2])^2 * ds_PML(x; Λf.phys_params)[2])

r = 5/sqrt(3)               # Filter radius
β = 32.0                    # β∈[1,∞], threshold sharpness
η = 0.5                     # η∈[0,1], threshold center

a_f(r, u, v) = r^2 * (∇(v) ⋅ ∇(u))

function Filter(p0; r, fem_params)
    ph = FEFunction(fem_params.P, p0)
    op = AffineFEOperator(fem_params.Pf, fem_params.Qf) do u, v
        ∫(a_f(r, u, v))fem_params.dΩ_d + ∫(v * u)fem_params.dΩ_d, ∫(v * ph)fem_params.dΩ_d
      end
    pfh = solve(op)
    return get_free_dof_values(pfh)
end

function Threshold(pfh; β, η)
    return ((tanh(β * η) + tanh(β * (pfh - η))) / (tanh(β * η) + tanh(β * (1.0 - η))))
end

using LinearAlgebra
ξd(p, n_air, n_metal)= 1 / (n_air + (n_metal - n_air) * p)^2 - 1 / n_air^2 # in the design region

a_base(u, v; phys_params) = (1 / phys_params.n_air^2) * ((∇ .* (Λ(phys_params) * v)) ⊙ (Λ(phys_params) .* ∇(u))) - (phys_params.k^2 * phys_params.μ * (v * u))

a_design(u, v, pth; phys_params) = ((p -> ξd(p, phys_params.n_air, phys_params.n_metal)) ∘ pth) * (∇(v) ⊙ ∇(u))

function MatrixA(pth; phys_params, fem_params)
    A_mat = assemble_matrix(fem_params.U, fem_params.V) do u, v
        ∫(a_base(u, v; phys_params))fem_params.dΩ + ∫(a_design(u, v, pth; phys_params))fem_params.dΩ_d
    end
    return lu(A_mat)
end

p0 = zeros(fem_params.np)  # Here we make p=0 everywhere just for illustration purpose
pf_vec = Filter(p0;r, fem_params)
pfh = FEFunction(fem_params.Pf, pf_vec)
pth = (pf -> Threshold(pf; β, η)) ∘ pfh
A_mat = MatrixA(pth; phys_params, fem_params)
b_vec = assemble_vector(v->(∫(v)fem_params.dΓ_s), fem_params.V)
u_vec = A_mat \ b_vec
uh = FEFunction(fem_params.U, u_vec)

function MatrixOf(fem_params)
    x0 = VectorValue(0,300)  # Position of the field to be optimized
    δ = 1
    return assemble_matrix(fem_params.U, fem_params.V) do u, v
        ∫((x->(1/(2*π)*exp(-norm(x - x0)^2 / 2 / δ^2))) * (∇(u) ⋅ ∇(v)) )fem_params.dΩ_c
    end
end

using ChainRulesCore, Zygote
import ChainRulesCore: rrule
NO_FIELDS = ZeroTangent()

Dptdpf(pf, β, η) = β * (1.0 - tanh(β * (pf - η))^2) / (tanh(β * η) + tanh(β * (1.0 - η)))

Dξdpf(pf, n_air, n_metal, β, η)= 2 * (n_air - n_metal) / (n_air + (n_metal - n_air) * Threshold(pf; β, η))^3 * Dptdpf(pf, β, η)

DAdpf(u, v, pfh; phys_params, β, η) = ((p -> Dξdpf(p, phys_params.n_air, phys_params.n_metal, β, η)) ∘ pfh) * (∇(v) ⊙ ∇(u))

function gf_pf(pf_vec; β, η, phys_params, fem_params)
    pfh = FEFunction(fem_params.Pf, pf_vec)
    pth = (pf -> Threshold(pf; β, η)) ∘ pfh
    A_mat = MatrixA(pth; phys_params, fem_params)
    b_vec = assemble_vector(v->(∫(v)fem_params.dΓ_s), fem_params.V)
    u_vec = A_mat \ b_vec

    O_mat = MatrixOf(fem_params)
    real(u_vec' * O_mat * u_vec)
end

function rrule(::typeof(gf_pf), pf_vec; β, η, phys_params, fem_params)
    function U_pullback(dgdg)
      NO_FIELDS, dgdg * Dgfdpf(pf_vec; β, η, phys_params, fem_params)
    end
    gf_pf(pf_vec; β, η, phys_params, fem_params), U_pullback
end

function Dgfdpf(pf_vec; β, η, phys_params, fem_params)
    pfh = FEFunction(fem_params.Pf, pf_vec)
    pth = (pf -> Threshold(pf; β, η)) ∘ pfh
    A_mat = MatrixA(pth; phys_params, fem_params)
    b_vec = assemble_vector(v->(∫(v)fem_params.dΓ_s), fem_params.V)
    u_vec = A_mat \ b_vec
    O_mat = MatrixOf(fem_params)

    uh = FEFunction(fem_params.U, u_vec)
    w_vec =  A_mat' \ (O_mat * u_vec)
    wconjh = FEFunction(fem_params.U, conj(w_vec))

    l_temp(dp) = ∫(real(-2 * DAdpf(uh, wconjh, pfh; phys_params, β, η)) * dp)fem_params.dΩ_d
    dgfdpf = assemble_vector(l_temp, fem_params.Pf)
    return dgfdpf
end

function pf_p0(p0; r, fem_params)
    pf_vec = Filter(p0; r, fem_params)
    pf_vec
end

function rrule(::typeof(pf_p0), p0; r, fem_params)
  function pf_pullback(dgdpf)
    NO_FIELDS, Dgdp(dgdpf; r, fem_params)
  end
  pf_p0(p0; r, fem_params), pf_pullback
end

function Dgdp(dgdpf; r, fem_params)
    Af = assemble_matrix(fem_params.Pf, fem_params.Qf) do u, v
        ∫(a_f(r, u, v))fem_params.dΩ_d + ∫(v * u)fem_params.dΩ_d
    end
    wvec = Af' \ dgdpf
    wh = FEFunction(fem_params.Pf, wvec)
    l_temp(dp) = ∫(wh * dp)fem_params.dΩ_d
    return assemble_vector(l_temp, fem_params.P)
end

function gf_p(p0::Vector; r, β, η, phys_params, fem_params)
    pf_vec = pf_p0(p0; r, fem_params)
    gf_pf(pf_vec; β, η, phys_params, fem_params)
end

function gf_p(p0::Vector, grad::Vector; r, β, η, phys_params, fem_params)
    if length(grad) > 0
        dgdp, = Zygote.gradient(p -> gf_p(p; r, β, η, phys_params, fem_params), p0)
        grad[:] = dgdp
    end
    gvalue = gf_p(p0::Vector; r, β, η, phys_params, fem_params)
    open("gvalue.txt", "a") do io
        write(io, "$gvalue \n")
    end
    gvalue
end

p0 = rand(fem_params.np)
δp = rand(fem_params.np)*1e-8
grad = zeros(fem_params.np)

g0 = gf_p(p0, grad; r, β, η, phys_params, fem_params)
g1 = gf_p(p0+δp, []; r, β, η, phys_params, fem_params)
g1-g0, grad'*δp

using NLopt

function gf_p_optimize(p_init; r, β, η, TOL = 1e-4, MAX_ITER = 500, phys_params, fem_params)
    opt = Opt(:LD_MMA, fem_params.np)
    opt.lower_bounds = 0
    opt.upper_bounds = 1
    opt.ftol_rel = TOL
    opt.maxeval = MAX_ITER
    opt.max_objective = (p0, grad) -> gf_p(p0, grad; r, β, η, phys_params, fem_params)

    (g_opt, p_opt, ret) = optimize(opt, p_init)
    @show numevals = opt.numevals # the number of function evaluations
    return g_opt, p_opt
end

p_opt = fill(0.4, fem_params.np)   # Initial guess
β_list = [8.0, 16.0, 32.0]

g_opt = 0
TOL = 1e-8
MAX_ITER = 100
for bi = 1 : 3
    β = β_list[bi]
    g_opt, p_temp_opt = gf_p_optimize(p_opt; r, β, η, TOL, MAX_ITER, phys_params, fem_params)
    global p_opt = p_temp_opt
end
@show g_opt

using CairoMakie, GridapMakie
p0 = p_opt

pf_vec = pf_p0(p0; r, fem_params)
pfh = FEFunction(fem_params.Pf, pf_vec)
pth = (pf -> Threshold(pf; β, η)) ∘ pfh
A_mat = MatrixA(pth; phys_params, fem_params)
b_vec = assemble_vector(v->(∫(v)fem_params.dΓ_s), fem_params.V)
u_vec = A_mat \ b_vec
uh = FEFunction(fem_params.U, u_vec)

fig, ax, plt = plot(fem_params.Ω, pth, colormap = :binary)
Colorbar(fig[1,2], plt)
ax.aspect = AxisAspect(1)
ax.title = "Design Shape"
rplot = 110 # Region for plot
limits!(ax, -rplot, rplot, (h1)/2-rplot, (h1)/2+rplot)
save("shape.png", fig)

maxe = 30 # Maximum electric field magnitude compared to the incident plane wave
e1=abs2(phys_params.n_air^2)
e2=abs2(phys_params.n_metal^2)

fig, ax, plt = plot(fem_params.Ω, 2*(sqrt∘(abs((conj(∇(uh)) ⋅ ∇(uh))/(CellField(e1,fem_params.Ω) + (e2 - e1) * pth)))), colormap = :hot, colorrange=(0, maxe))
Colorbar(fig[1,2], plt)
ax.title = "|E|"
ax.aspect = AxisAspect(1)
limits!(ax, -rplot, rplot, (h1)/2-rplot, (h1)/2+rplot)
save("Field.png", fig)

