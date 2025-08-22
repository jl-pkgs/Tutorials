
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.CellData

u(x) = sin(2*π*x[1])*sin(2*π*x[2])
q(x) = -∇(u)(x)  # Define the flux q = -∇u
f(x) = (∇ ⋅ q)(x) # Source term f = -Δu = -∇⋅(∇u)$

D = 2  # Problem dimension
nc = Tuple(fill(8, D))  # 4 cells in each direction
domain = Tuple(repeat([0, 1], D))  # Unit cube domain
model = simplexify(CartesianDiscreteModel(domain,nc))

Ω = Triangulation(ReferenceFE{D}, model)  # Volume triangulation
Γ = Triangulation(ReferenceFE{D-1}, model)  # Skeleton triangulation

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)  # Patch volume triangulation
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)  # Patch skeleton triangulation

order = 1  # Polynomial order
reffe_Q = ReferenceFE(lagrangian, VectorValue{D, Float64}, order; space=:P)
reffe_V = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_M = ReferenceFE(lagrangian, Float64, order; space=:P)

V = TestFESpace(Ω, reffe_V; conformity=:L2)  # Discontinuous vector space
Q = TestFESpace(Ω, reffe_Q; conformity=:L2)  # Discontinuous scalar space
M = TestFESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")  # Interface space
N = TrialFESpace(M, u)

mfs = BlockMultiFieldStyle(2,(2,1))  # Special blocking for efficient static condensation
X = MultiFieldFESpace([V, Q, N]; style=mfs)

degree = 2*(order+1)  # Integration degree
dΩp = Measure(Ωp,degree)  # Volume measure, on the patch triangulation
dΓp = Measure(Γp,degree)  # Surface measure, on the patch boundary triangulation

τ = 1.0 # HDG stabilization parameter

n = get_normal_vector(Γp)  # Face normal vector
Πn(u) = u⋅n  # Normal component
Π(u) = change_domain(u,Γp,DomainStyle(u))  # Project to skeleton

a((uh,qh,sh),(vh,wh,lh)) = ∫( qh⋅wh - uh*(∇⋅wh) - qh⋅∇(vh) )dΩp + ∫(sh*Πn(wh))dΓp +
                           ∫((Πn(qh) + τ*(Π(uh) - sh))*(Π(vh) + lh))dΓp
l((vh,wh,lh)) = ∫( f*vh )dΩp

op = MultiField.StaticCondensationOperator(ptopo,X,a,l)
uh, qh, sh = solve(op)

dΩ = Measure(Ω,degree)
eh = uh - u
l2_uh = sqrt(sum(∫(eh⋅eh)*dΩ))

writevtk(Ω,"results",cellfields=["uh"=>uh,"qh"=>qh,"eh"=>eh])

