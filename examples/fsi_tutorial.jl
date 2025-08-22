

using Gridap

model = DiscreteModelFromFile("../models/elasticFlag.json")

writevtk(model,"model")

const U = 1.0
const H = 0.41
uf_in(x) = VectorValue( 1.5 * U * x[2] * ( H - x[2] ) / ( (H/2)^2 ), 0.0 )
uf_0(x) = VectorValue( 0.0, 0.0 )
us_0(x) = VectorValue( 0.0, 0.0 )

hN(x) = VectorValue( 0.0, 0.0 )
p_jump(x) = 0.0

f(x) = VectorValue( 0.0, 0.0 )
s(x) = VectorValue( 0.0, 0.0 )
g(x) = 0.0

function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

const E_s = 1.0
const ν_s = 0.33
const (λ_s,μ_s) = lame_parameters(E_s,ν_s)

using LinearAlgebra: tr
σₛ(ε) = λ_s*tr(ε)*one(ε) + 2*μ_s*ε

const μ_f = 0.01

σ_dev_f(ε) = 2*μ_f*ε

Ω = Interior(model)
Ω_s = Interior(model,tags="solid")
Ω_f = Interior(model,tags="fluid")

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)

Γ_fs = InterfaceTriangulation(Ω_f,Ω_s)
n_Γfs = get_normal_vector(Γ_fs)

k = 2

reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
reffeₚ = ReferenceFE(lagrangian,Float64,k-1)

Vf = TestFESpace(
  Ω_f,
  reffeᵤ,
  conformity=:H1,
  dirichlet_tags=["inlet", "noSlip", "cylinder"])

Vs = TestFESpace(
  Ω_s,
  reffeᵤ,
  conformity=:H1,
  dirichlet_tags=["fixed"])

Qf = TestFESpace(
  Ω_f,
  reffeₚ,
  conformity=:C0)

Uf = TrialFESpace(Vf,[uf_in, uf_0, uf_0])
Us = TrialFESpace(Vs,[us_0])
Pf = TrialFESpace(Qf)

Y = MultiFieldFESpace([Vs,Vf,Qf])
X = MultiFieldFESpace([Us,Uf,Pf])

degree = 2*k
dΩₛ = Measure(Ω_s,degree)
dΩ_f = Measure(Ω_f,degree)

bdegree = 2*k
dΓ_out = Measure(Γ_out,bdegree)

idegree = 2*k
dΓ_fs = Measure(Γ_fs,idegree)

a((us,uf,p),(vs,vf,q)) = a_s(us,vs) + a_f((uf,p),(vf,q)) + a_fs((us,uf,p),(vs,vf,q))
l((vs,vf,q)) = l_s(vs) + l_f((vf,q)) + l_f_Γn(vf)

a_s(u,v) = ∫( ε(v) ⊙ (σₛ ∘ ε(u)) )dΩₛ

a_f((u,p),(v,q)) = ∫( ε(v) ⊙ (σ_dev_f ∘ ε(u)) - (∇⋅v)*p + q*(∇⋅u) )dΩ_f

a_fs((us,uf,p),(vs,vf,q)) = ∫( α*(jump_Γ(vf,vs)⋅jump_Γ(uf,us)) +
                               jump_Γ(vf,vs)⋅t_Γfs(1,p,uf,n_Γfs) +
                               t_Γfs(χ,q,vf,n_Γfs)⋅jump_Γ(uf,us) )dΓ_fs
const γ = 1.0
const χ = -1.0
jump_Γ(wf,ws) = wf.⁺-ws.⁻
t_Γfs(χ,q,w,n) = q.⁺*n.⁺ - χ*n.⁺⋅(σ_dev_f∘ε(w.⁺))

using Gridap.CellData
dim = num_cell_dims(Γ_fs)
h_Γfs = get_array(∫(1)dΓ_fs)
α = CellField( lazy_map(h->γ*μ_f/(h.^dim),h_Γfs), Γ_fs)

l_s(v) = ∫( v⋅s )dΩₛ

l_f((v,q)) = ∫( v⋅f + q*g )dΩ_f

l_f_Γn(v) = ∫( v⋅hN )dΓ_out

op = AffineFEOperator(a,l,X,Y)

uhs, uhf, ph = solve(op)

writevtk(Ω,"trian", cellfields=["uhs" => uhs, "uhf" => uhf, "ph" => ph])

writevtk(Ω_s,"trian_solid",cellfields=["uhs"=>uhs])
writevtk(Ω_f,"trian_fluid",cellfields=["ph"=>ph,"uhf"=>uhf])

Γ_S = BoundaryTriangulation(model,tags=["cylinder","interface"])
dΓₛ = Measure(Γ_S,bdegree)
n_ΓS = get_normal_vector(Γ_S)
FD, FL = sum( ∫( (n_ΓS⋅σ_dev_f(ε(uhf))) - ph*n_ΓS )*dΓₛ )
println("Drag force: ", FD)
println("Lift force: ", FL)

