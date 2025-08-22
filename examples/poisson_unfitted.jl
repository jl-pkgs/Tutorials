
using Gridap
using GridapEmbedded

R  = 0.5
L  = 0.5*R
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(-L,L)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin

cutgeo = cut(bgmodel,geo3)

Ω_act = Triangulation(cutgeo,ACTIVE)

Ω_bg = Triangulation(bgmodel)
writevtk(Ω_bg,"bg_trian")
writevtk(Ω_act,"act_trian")

Ω = Triangulation(cutgeo,PHYSICAL)
writevtk(Ω,"phys_trian")

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Vstd = TestFESpace(Ω_act,reffe,conformity=:H1)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

colors = color_aggregates(aggregates,bgmodel)
Ω_bg = Triangulation(bgmodel)
writevtk(Ω_bg,"aggs_on_bg_trian",celldata=["aggregate"=>aggregates,"color"=>colors])

V = AgFEMSpace(Vstd,aggregates)
U = TrialFESpace(V)

degree = 2*order
dΩ = Measure(Ω,degree)

Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

u(x) = x[1] - x[2] # Solution of the problem
const γd = 10.0    # Nitsche coefficient
const h = dp[1]/n  # Mesh size according to the parameters of the background grid

a(u,v) =
  ∫( ∇(v)⋅∇(u) )dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ

l(v) = ∫( (γd/h)*v*u - (n_Γ⋅∇(v))*u )dΓ

op = AffineFEOperator(a,l,U,V)
uh = solve(op)

e = u - uh

l2(u) = sqrt(sum( ∫( u*u )*dΩ ))
h1(u) = sqrt(sum( ∫( u*u + ∇(u)⋅∇(u) )*dΩ ))

el2 = l2(e)
eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

using Test
@test el2/ul2 < 1.e-8
@test eh1/uh1 < 1.e-7

writevtk(Ω,"results.vtu",cellfields=["uh"=>uh])

