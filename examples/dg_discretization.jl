

using Gridap
u(x) = 3*x[1] + x[2]^2 + 2*x[3]^3 + x[1]*x[2]*x[3]
f(x) = -2 - 12*x[3] 
g(x) = u(x)

∇u(x) = VectorValue(3        + x[2]*x[3],   
                    2*x[2]   + x[1]*x[3], 
                    6*x[3]^2 + x[1]*x[2])

import Gridap: ∇
∇(::typeof(u)) = ∇u

∇(u) === ∇u
 

L = 1.0
domain = (0.0, L, 0.0, L, 0.0, L)
n = 4
partition = (n,n,n)
model = CartesianDiscreteModel(domain,partition)

 writevtk(model,"model")

domain2D = (0.0, L, 0.0, L)
partition2D = (n,n)
model2D = CartesianDiscreteModel(domain2D,partition2D)

order = 3
V = TestFESpace(model,
                ReferenceFE(lagrangian,Float64,order),
                conformity=:L2)

U = TrialFESpace(V)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)

Λ = SkeletonTriangulation(model)

writevtk(Λ,"strian")

degree = 2*order

dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΛ = Measure(Λ,degree)

n_Γ = get_normal_vector(Γ)
n_Λ = get_normal_vector(Λ)

a_Ω(u,v) = ∫( ∇(v)⊙∇(u) )dΩ 
l_Ω(v) = ∫( v*f )dΩ  

h = L / n
γ = order*(order+1)
a_Γ(u,v) = ∫( - v*(∇(u)⋅n_Γ) - (∇(v)⋅n_Γ)*u + (γ/h)*v*u )dΓ
l_Γ(v)   = ∫(                - (∇(v)⋅n_Γ)*g + (γ/h)*v*g )dΓ

a_Λ(u,v) = ∫( - jump(v*n_Λ)⊙mean(∇(u)) 
              - mean(∇(v))⊙jump(u*n_Λ) 
              + (γ/h)*jump(v*n_Λ)⊙jump(u*n_Λ) )dΛ

a(u,v) = a_Ω(u,v) + a_Γ(u,v) + a_Λ(u,v)
l(v) = l_Ω(v) + l_Γ(v)

op = AffineFEOperator(a, l, U, V)
uh = solve(op)

writevtk(Λ,"jumps",cellfields=["jump_u"=>jump(uh)])

e = u - uh

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

el2 = l2(e)
eh1 = h1(e)

tol = 1.e-10
@assert el2 < tol
@assert eh1 < tol

