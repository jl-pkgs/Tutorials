
using Gridap

u(x) = x[1] + x[2]
f(x) = 0

∇u(x) = VectorValue(1,1)

import Gridap: ∇
∇(::typeof(u)) = ∇u

∇(u) === ∇u

domain = (0,1,0,1)
partition = (4,4)
model = CartesianDiscreteModel(domain,partition)

domain3d = (0,1,0,1,0,1)
partition3d = (4,4,4)
model3d = CartesianDiscreteModel(domain3d,partition3d)

writevtk(model,"model")

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
U = TrialFESpace(V0,u)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(u,v) = ∫( ∇(v)⊙∇(u) )*dΩ
b(v) = ∫( v*f )*dΩ

op = AffineFEOperator(a,b,U,V0)

uh = solve(op)

e = u - uh

writevtk(Ω,"error",cellfields=["e" => e])

el2 = sqrt(sum( ∫( e*e )*dΩ ))
eh1 = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )*dΩ ))

tol = 1.e-10
@assert el2 < tol
@assert eh1 < tol

p = 3
u(x) = x[1]^p+x[2]^p
∇u(x) = VectorValue(p*x[1]^(p-1),p*x[2]^(p-1))
f(x) = -p*(p-1)*(x[1]^(p-2)+x[2]^(p-2))

∇(::typeof(u)) = ∇u
b(v) = ∫( v*f )*dΩ

function run(n,k)

  domain = (0,1,0,1)
  partition = (n,n)
  model = CartesianDiscreteModel(domain,partition)

  reffe = ReferenceFE(lagrangian,Float64,k)
  V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
  U = TrialFESpace(V0,u)

  degree = 2*p
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(u,v) = ∫( ∇(v)⊙∇(u) )*dΩ
  b(v) = ∫( v*f )*dΩ

  op = AffineFEOperator(a,b,U,V0)

  uh = solve(op)

  e = u - uh

  el2 = sqrt(sum( ∫( e*e )*dΩ ))
  eh1 = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )*dΩ ))

  (el2, eh1)

end

function conv_test(ns,k)

  el2s = Float64[]
  eh1s = Float64[]
  hs = Float64[]

  for n in ns

    el2, eh1 = run(n,k)
    h = 1.0/n

    push!(el2s,el2)
    push!(eh1s,eh1)
    push!(hs,h)

  end

  (el2s, eh1s, hs)

end

el2s1, eh1s1, hs = conv_test([8,16,32,64,128],1);
el2s2, eh1s2, hs = conv_test([8,16,32,64,128],2);

using Plots

plot(hs,[el2s1 eh1s1 el2s2 eh1s2],
    xaxis=:log, yaxis=:log,
    label=["L2 k=1" "H1 k=1" "L2 k=2" "H1 k=2"],
    shape=:auto,
    xlabel="h",ylabel="error norm")

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

slope(hs,el2s1)
slope(hs,el2s2)

slope(hs,eh1s1)
slope(hs,eh1s2)

