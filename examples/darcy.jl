
using Gridap
domain = (0,1,0,1)
partition = (100,100)
model = CartesianDiscreteModel(domain,partition)

order = 1

V = FESpace(model, ReferenceFE(raviart_thomas,Float64,order),
      conformity=:HDiv, dirichlet_tags=[5,6])

Q = FESpace(model, ReferenceFE(lagrangian,Float64,order),
      conformity=:L2)

uD = VectorValue(0.0,0.0)
U = TrialFESpace(V,uD)
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

trian = Triangulation(model)
degree = 2
dΩ = Measure(trian,degree)

neumanntags = [8,]
btrian = BoundaryTriangulation(model,tags=neumanntags)
dΓ = Measure(btrian,degree)

const kinv1 = TensorValue(1.0,0.0,0.0,1.0)
const kinv2 = TensorValue(100.0,90.0,90.0,100.0)
function σ(x,u)
   if ((abs(x[1]-0.5) <= 0.1) && (abs(x[2]-0.5) <= 0.1))
      return kinv2⋅u
   else
      return kinv1⋅u
   end
end

px = get_physical_coordinate(trian)

a((u,p), (v,q)) = ∫(v⋅(σ∘(px,u)) - (∇⋅v)*p + q*(∇⋅u))dΩ

nb = get_normal_vector(btrian)
h = -1.0

b((v,q)) = ∫((v⋅nb)*h)dΓ

op = AffineFEOperator(a,b,X,Y)
xh = solve(op)
uh, ph = xh

writevtk(trian,"darcyresults",cellfields=["uh"=>uh,"ph"=>ph])

