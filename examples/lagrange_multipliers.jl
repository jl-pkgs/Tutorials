
using Gridap

u_exact(x) = sin(x[1]) * cos(x[2])

model = CartesianDiscreteModel((0,1,0,1),(8,8))

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V = FESpace(model, reffe)

Λ = ConstantFESpace(model)

X = MultiFieldFESpace([V, Λ])

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
dΩ = Measure(Ω, 2*order)
dΓ = Measure(Γ, 2*order)

f(x) = -Δ(u_exact)(x)
g(x) = ∇(u_exact)(x)
ū = sum(∫(u_exact)dΩ)
nΓ = get_normal_vector(Γ)

a((u,λ),(v,μ)) = ∫(∇(u)⋅∇(v) + λ*v + u*μ)dΩ
l((v,μ)) = ∫(f*v + μ*ū)dΩ + ∫(v*(g⋅nΓ))*dΓ

op = AffineFEOperator(a, l, X, X)
uh, λh = solve(op)

eh = uh - u_exact
l2_error = sqrt(sum(∫(eh⋅eh)*dΩ))
ūh = sum(∫(uh)*dΩ)

writevtk(Ω, "results", cellfields=["uh"=>uh, "error"=>eh])
