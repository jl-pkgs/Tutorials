

using Gridap
using GridapGmsh

model_pentagon = GmshDiscreteModel("../models/pentagon_mesh.msh")

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V_0 = TestFESpace(model_pentagon,reffe;conformity=:H1,dirichlet_tags=["l1","l2","l3","l4","l5"])

boundary_cond = [200,100,0,0,100];
U_g = TrialFESpace(V_0,boundary_cond)

degree = 2
Ω = Triangulation(model_pentagon)
dΩ = Measure(Ω,degree)

velocity_zero = VectorValue(0.0, 0.0);

velocity_nonzero = VectorValue(0.0, 2.0);

D = 0.1;

a_zero(u, v)     = ∫(v * (velocity_zero ⋅∇(u))    + ∇(v) ⋅ (D * ∇(u))) * dΩ
a_non_zero(u, v) = ∫(v * (velocity_nonzero ⋅∇(u)) + ∇(v) ⋅ (D * ∇(u))) * dΩ
b(v) = 0.0

op_zero = AffineFEOperator(a_zero,b,U_g,V_0)
uh_zero = Gridap.Algebra.solve(op_zero)

op_non_zero = AffineFEOperator(a_non_zero,b,U_g,V_0)
uh_non_zero = Gridap.Algebra.solve(op_non_zero)

writevtk(Ω,"results_zero",cellfields=["uh_zero"=>uh_zero])

writevtk(Ω,"results_non_zero",cellfields=["uh_non_zero"=>uh_non_zero])

