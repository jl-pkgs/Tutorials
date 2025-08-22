

using DrWatson
@quickactivate "Tutorials"

using Gridap
import Gridap: ∇

params = Dict(
  "cells_per_axis" => [8,16,32,64],
  "fe_order" => [1,2]
)

dicts = dict_list(params)

p = 3
u(x) = x[1]^p+x[2]^p
∇u(x) = VectorValue(p*x[1]^(p-1),p*x[2]^(p-1))
f(x) = -p*(p-1)*(x[1]^(p-2)+x[2]^(p-2))
∇(::typeof(u)) = ∇u

function run(n::Int,k::Int)

  domain = (0,1,0,1)
  partition = (n,n)
  model = CartesianDiscreteModel(domain,partition)

  reffe = ReferenceFE(lagrangian,Float64,k)
  V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
  U = TrialFESpace(V0,u)

  degree = 2*p
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(u,v) = ∫( ∇(u)⊙∇(v) ) * dΩ
  b(v) = ∫( v*f ) * dΩ

  op = AffineFEOperator(a,b,U,V0)

  uh = solve(op)

  e = u - uh

  el2 = sqrt(sum( ∫( e*e )*dΩ ))
  eh1 = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )*dΩ ))

  (el2, eh1)

end

function run(case::Dict)
  @unpack cells_per_axis, fe_order = case
  el2, eh1 = run(cells_per_axis,fe_order)
  h = 1.0/cells_per_axis
  results = @strdict el2 eh1 h
  merge(case,results)
end

function run_or_load(case::Dict)
  produce_or_load(
    projectdir("assets","validation_DrWatson"),
    case,
    run,
    prefix="res",
    tag=true,
    verbose=true
  )
  return true
end

map(run_or_load,dicts)

using DataFrames

df = collect_results(projectdir("assets","validation_DrWatson"))

sort!(df,:h)
hs = df[(df.fe_order .== 1),:h]
el2s1 = df[(df.fe_order .== 1),:el2]
eh1s1 = df[(df.fe_order .== 1),:eh1]
el2s2 = df[(df.fe_order .== 2),:el2]
eh1s2 = df[(df.fe_order .== 2),:eh1]

using Plots

plot(hs,[el2s1 eh1s1 el2s2 eh1s2],
    xaxis=:log, yaxis=:log,
    label=["L2 k=1" "H1 k=1" "L2 k=2" "H1 k=2"],
    shape=:auto,
    xlabel="h",ylabel="error norm")

