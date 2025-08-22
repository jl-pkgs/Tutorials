

using Gridap
domain = (-1, +1, -1, +1)
partition = (20, 20)
model = CartesianDiscreteModel(domain, partition)

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)

V0 = TestFESpace(model, reffe, dirichlet_tags="boundary")

g(t) = x -> exp(-2 * t) * sinpi(t * x[1]) * (x[2]^2 - 1)
Ug = TransientTrialFESpace(V0, g)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

α₀(t) = x -> 1 + sin(t) * (x[1]^2 + x[2]^2) / 4
α₁(t) = x -> cos(t) * x[1]^2 / 2
α₂(t) = x -> 1 + t * (x[1]^2 + x[2]^2)
α(t, u) = α₀(t) + α₁(t) * u + α₂(t) * u * u
f(t) = x -> sin(t) * sinpi(x[1]) * sinpi(x[2])

m(t, u, v) = ∫(v * ∂t(u))dΩ
a(t, u, v) = ∫(∇(v) ⋅ (α(t, u) * ∇(u)))dΩ
l(t, v) = ∫(v * f(t))dΩ

jac_m(t, u, dtu, v) = ∫(v * dtu)dΩ
jac_α(t, u, du) = α₁(t) * du + α₂(t) * (2 * u * du)
jac_a(t, u, du, v) = ∫(∇(v) ⋅ (α(t, u) * ∇(du)))dΩ + ∫(∇(v) ⋅ (jac_α(t, u, du) * ∇(u)))dΩ

res(t, u, v) = m(t, u, v) + a(t, u, v) - l(t, v)
jac(t, u, du, v) = jac_a(t, u, du, v)
jac_t(t, u, dtu, v) = jac_m(t, u, dtu, v)

op = TransientFEOperator(res, (jac, jac_t), Ug, V0)

mass_ql(t, u, dtu, v) = ∫(dtu * v)dΩ
res_ql(t, u, v) = a(t, u, v) - l(t, v)
jac_ql(t, u, du, v) = jac_a(t, u, du, v)
jac_t_ql(t, u, dtu, v) = jac_m(t, u, dtu, v)
op_ql = TransientQuasilinearFEOperator(mass_ql, res_ql, (jac_ql, jac_t_ql), Ug, V0)

mass_sl(t, dtu, v) = ∫(dtu * v)dΩ
res_sl(t, u, v) = a(t, u, v) - l(t, v)
jac_sl(t, u, du, v) = jac_a(t, u, du, v)
jac_t_sl(t, u, dtu, v) = mass_sl(t, dtu, v)
op_sl = TransientSemilinearFEOperator(
  mass_sl, res_sl, (jac_sl, jac_t_sl),
  Ug, V0, constant_mass=true
)

lin_solver = LUSolver()
nl_solver = NLSolver(lin_solver, method=:newton, iterations=10, show_trace=false)

Δt = 0.05
θ = 0.5
solver = ThetaMethod(nl_solver, Δt, θ)

tableau = :SDIRK_2_2
solver_rk = RungeKutta(nl_solver, lin_solver, Δt, tableau)

t0, tF = 0.0, 10.0
uh0 = interpolate_everywhere(g(t0), Ug(t0))
uh = solve(solver, op_sl, t0, tF, uh0)

if !isdir("tmp_nl")
  mkdir("tmp_nl")
end

createpvd("results_nl") do pvd
  pvd[0] = createvtk(Ω, "tmp_nl/results_0" * ".vtu", cellfields=["u" => uh0])
  for (tn, uhn) in uh
    pvd[tn] = createvtk(Ω, "tmp_nl/results_$tn" * ".vtu", cellfields=["u" => uhn])
  end
end

