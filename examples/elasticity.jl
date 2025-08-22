
using Gridap
model = DiscreteModelFromFile("../models/solid.json")

writevtk(model,"model")

order = 1

reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V0 = TestFESpace(model,reffe;
  conformity=:H1,
  dirichlet_tags=["surface_1","surface_2"],
  dirichlet_masks=[(true,false,false), (true,true,true)])

g1(x) = VectorValue(0.005,0.0,0.0)
g2(x) = VectorValue(0.0,0.0,0.0)

U = TrialFESpace(V0,[g1,g2])

const E = 70.0e9
const ν = 0.33
const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))
σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

a(u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ
l(v) = 0

op = AffineFEOperator(a,l,U,V0)
uh = solve(op)

writevtk(Ω,"results",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ∘ε(uh)])

using Gridap.Geometry
labels = get_face_labeling(model)
dimension = 3
tags = get_face_tag(labels,dimension)

const alu_tag = get_tag_from_name(labels,"material_1")

function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

const E_alu = 70.0e9
const ν_alu = 0.33
const (λ_alu,μ_alu) = lame_parameters(E_alu,ν_alu)

const E_steel = 200.0e9
const ν_steel = 0.33
const (λ_steel,μ_steel) = lame_parameters(E_steel,ν_steel)

function σ_bimat(ε,tag)
  if tag == alu_tag
    return λ_alu*tr(ε)*one(ε) + 2*μ_alu*ε
  else
    return λ_steel*tr(ε)*one(ε) + 2*μ_steel*ε
  end
end

a(u,v) = ∫( ε(v) ⊙ (σ_bimat∘(ε(u),tags)) )*dΩ

op = AffineFEOperator(a,l,U,V0)
uh = solve(op)

writevtk(Ω,"results_bimat",cellfields=
  ["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ_bimat∘(ε(uh),tags)])

tags_field = CellField(tags, Ω)
σ_from_tag(tag) = tag==alu_tag ? 1. : 0.
σ_bimat_cst = σ_from_tag ∘ tags_field

a(u,v) = ∫( σ_bimat_cst * ∇(u)⋅∇(v))*dΩ
writevtk(Ω,"const_law",cellfields= ["sigma"=>σ_bimat_cst])

