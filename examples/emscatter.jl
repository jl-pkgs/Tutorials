
using Gridap
using GridapGmsh
using Gridap.Fields
using Gridap.Geometry

λ = 1.0          # Wavelength (arbitrary unit)
L = 4.0          # Width of the area
H = 6.0          # Height of the area
xc = [0 -1.0]    # Center of the cylinder
r = 1.0          # Radius of the cylinder
d_pml = 0.8      # Thickness of the PML   
k = 2*π/λ        # Wave number 
const ϵ₁ = 3.0   # Relative electric permittivity for cylinder

model = GmshDiscreteModel("../models/geometry.msh")

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe,dirichlet_tags="DirichletEdges",vector_type=Vector{ComplexF64})
U = V # mathematically equivalent to TrialFESpace(V,0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γ = BoundaryTriangulation(model;tags="Source")
dΓ = Measure(Γ,degree)

Rpml = 1e-12      # Tolerance for PML reflection 
σ = -3/4*log(Rpml)/d_pml # σ_0
LH = (L,H) # Size of the PML inner boundary (a rectangular center at (0,0))

function s_PML(x,σ,k,LH,d_pml)
    u = abs.(Tuple(x)).-LH./2  # get the depth into PML
    return @. ifelse(u > 0,  1+(1im*σ/k)*(u/d_pml)^2, $(1.0+0im))
end

function ds_PML(x,σ,k,LH,d_pml)
    u = abs.(Tuple(x)).-LH./2 # get the depth into PML
    ds = @. ifelse(u > 0, (2im*σ/k)*(1/d_pml)^2*u, $(0.0+0im))
    return ds.*sign.(Tuple(x))
end

struct Λ<:Function
    σ::Float64
    k::Float64
    LH::NTuple{2,Float64}
    d_pml::Float64
end

function (Λf::Λ)(x)
    s_x,s_y = s_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)
    return VectorValue(1/s_x,1/s_y) 
end

Fields.∇(Λf::Λ) = x->TensorValue{2,2,ComplexF64}(-(Λf(x)[1])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[1],0,0,-(Λf(x)[2])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[2])

labels = get_face_labeling(model)
dimension = num_cell_dims(model)
tags = get_face_tag(labels,dimension)
const cylinder_tag = get_tag_from_name(labels,"Cylinder")

function ξ(tag)
    if tag == cylinder_tag
        return 1/ϵ₁
    else
        return 1.0
    end
end

τ = CellField(tags,Ω) 
Λf = Λ(σ,k,LH,d_pml) 

a(u,v) = ∫(  (∇.*(Λf*v))⊙((ξ∘τ)*(Λf.*∇(u))) - (k^2*(v*u))  )dΩ

b(v) = ∫(v)*dΓ

op = AffineFEOperator(a,b,U,V)
uh = solve(op)

using SpecialFunctions
dbesselj(m,z) = (besselj(m-1,z)-besselj(m+1,z))/2
dhankelh1(m,z)= (hankelh1(m-1,z)-hankelh1(m+1,z))/2
α(m,n,z) = (besselj(m,n*z)*dbesselj(m,z)-1/n*besselj(m,z)*dbesselj(m,n*z))/(1/n*hankelh1(m,z)*dbesselj(m,n*z)-besselj(m,n*z)*dhankelh1(m,z))
β(m,n,z) = (hankelh1(m,z)*dbesselj(m,z)-besselj(m,z)*dhankelh1(m,z))/(1/n*dbesselj(m,n*z)*hankelh1(m,z)-besselj(m,n*z)*dhankelh1(m,z))

function H_t(x,xc,r,ϵ,λ)
    n = √ϵ
    k = 2*π/λ
    θ = angle(x[1]-xc[1]+1im*(x[2]-xc[2]))+π
    M = 40 # Number of Bessel function basis used
    H0 = 0
    if norm([x[1]-xc[1],x[2]-xc[2]])<=r
        for m=-M:M
            H0 += β(m,n,k*r)*cis(m*θ)*besselj(m,n*k*norm([x[1]-xc[1],x[2]-xc[2]]))
        end
    else
        for m=-M:M
            H0 += α(m,n,k*r)*cis(m*θ)*hankelh1(m,k*norm([x[1]-xc[1],x[2]-xc[2]]))+cis(m*θ)*besselj(m,k*norm([x[1]-xc[1],x[2]-xc[2]]))
        end
    end
    return 1im/(2*k)*H0
end
uh_t = CellField(x->H_t(x,xc,r,ϵ₁,λ),Ω)

writevtk(Ω,"demo",cellfields=["Real"=>real(uh),
        "Imag"=>imag(uh),
        "Norm"=>abs2(uh),
        "Real_t"=>real(uh_t),
        "Imag_t"=>imag(uh_t),
        "Norm_t"=>abs2(uh_t),
        "Difference"=>abs(uh_t-uh)])
function AnalyticalBox(x) # Get the "center" region
    if abs(x[1])<L/2 && abs(x[2]+0.5)<2.5
        return 1
    else
        return 0
    end
end

Difference=sqrt(sum(∫(abs2(uh_t-uh)*AnalyticalBox)*dΩ)/sum(∫(abs2(uh_t)*AnalyticalBox)*dΩ))

@assert Difference < 0.1
