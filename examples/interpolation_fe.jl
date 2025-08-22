

using Test
using Gridap
using Gridap.CellData
using Gridap.Visualization

domain = (0,1,0,1)
partition = (5,5)
𝒯₁ = CartesianDiscreteModel(domain, partition)

f(x) = x[1] + x[2]

reffe₁ = ReferenceFE(lagrangian, Float64, 1)
V₁ = FESpace(𝒯₁, reffe₁)

fₕ = interpolate_everywhere(f,V₁)

using Random
pt = Point(rand(2))
pts = [Point(rand(2)) for i in 1:3]

fₕ(pt), fₕ.(pts)

@test fₕ(pt) ≈ f(pt)
@test fₕ.(pts) ≈ f.(pts)

partition = (20,20)
𝒯₂ = CartesianDiscreteModel(domain,partition)

reffe₂ = ReferenceFE(lagrangian, Float64, 2)
V₂ = FESpace(𝒯₂, reffe₂)

ifₕ = Interpolable(fₕ)

gₕ = interpolate_everywhere(ifₕ, V₂)

ḡₕ = interpolate(ifₕ, V₂)

@test gₕ.cell_dof_values ==  ḡₕ.cell_dof_values

g̃ₕ = interpolate_dirichlet(ifₕ, V₂)

g̃ₕ.cell_dof_values

@test fₕ(pt) ≈ gₕ(pt) ≈ f(pt)
@test fₕ.(pts) ≈ gₕ.(pts) ≈ f.(pts)

writevtk(get_triangulation(fₕ), "source", cellfields=["fₕ"=>fₕ])
writevtk(get_triangulation(gₕ), "target", cellfields=["gₕ"=>gₕ])

f(x) = VectorValue([x[1], x[2]])

reffe₁ = ReferenceFE(raviart_thomas, Float64, 1) # RT space of order 1
V₁ = FESpace(𝒯₁, reffe₁)
fₕ = interpolate_everywhere(f, V₁)

fₕ(pt), fₕ.(pts)

reffe₂ = ReferenceFE(raviart_thomas, Float64, 1) # RT space of order 1
V₂ = FESpace(𝒯₂, reffe₂)
ifₕ = Interpolable(fₕ)

gₕ = interpolate_everywhere(ifₕ, V₂)

@test gₕ(pt) ≈ f(pt) ≈ fₕ(pt)

f(x) = VectorValue([x[1], x[1]+x[2]])

reffe₁ = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
V₁ = FESpace(𝒯₁, reffe₁)
fₕ = interpolate_everywhere(f, V₁)

reffe₂ = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
V₂ = FESpace(𝒯₂, reffe₂)

ifₕ = Interpolable(fₕ)
gₕ = interpolate_everywhere(ifₕ, V₂)

@test gₕ(pt) ≈ f(pt) ≈ fₕ(pt)

h₁(x) = x[1]+x[2]
h₂(x) = x[1]

reffe₁ = ReferenceFE(lagrangian, Float64, 1)
V₁ = FESpace(𝒯₁, reffe₁)

V₁xV₁ = MultiFieldFESpace([V₁,V₁])
fₕ = interpolate_everywhere([h₁, h₂], V₁xV₁)

reffe₂ = ReferenceFE(lagrangian, Float64, 2)
V₂ = FESpace(𝒯₂, reffe₂)
V₂xV₂ = MultiFieldFESpace([V₂,V₂])

fₕ¹, fₕ² = fₕ
ifₕ¹ = Interpolable(fₕ¹)
ifₕ² = Interpolable(fₕ²)

gₕ = interpolate_everywhere([ifₕ¹,ifₕ²], V₂xV₂)

gₕ¹, gₕ² = gₕ
@test fₕ¹(pt) ≈ gₕ¹(pt)
@test fₕ²(pt) ≈ gₕ²(pt)

