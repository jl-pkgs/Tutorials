
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using DataStructures

ϵ = 1e-2
r(x) = ((x[1]-0.5)^2 + (x[2]-0.5)^2)^(1/2)
u_exact(x) = 1.0 / (ϵ + r(x))

function LShapedModel(n)
  model = CartesianDiscreteModel((0,1,0,1),(n,n))
  cell_coords = map(mean,get_cell_coordinates(model))
  l_shape_filter(x) = (x[1] < 0.5) || (x[2] < 0.5)
  mask = map(l_shape_filter,cell_coords)
  model = simplexify(DiscreteModelPortion(model,mask))

  grid = get_grid(model)
  topo = get_grid_topology(model)
  return UnstructuredDiscreteModel(grid, topo, FaceLabeling(topo))
end

l2_norm(he,xh,dΩ) = ∫(he*(xh*xh))*dΩ
l2_norm(xh,dΩ) = ∫(xh*xh)*dΩ

function amr_step(model,u_exact;order=1)
  "Create FE spaces with Dirichlet boundary conditions on all boundaries"
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["boundary"])
  U = TrialFESpace(V,u_exact)
  
  "Setup integration measures"
  Ω = Triangulation(model)
  Γ = Boundary(model)
  Λ = Skeleton(model)
  
  dΩ = Measure(Ω,4*order)
  dΓ = Measure(Γ,2*order)
  dΛ = Measure(Λ,2*order)
  
  "Compute cell sizes for error estimation"
  hK = CellField(sqrt.(collect(get_array(∫(1)dΩ))),Ω)

  "Get normal vectors for boundary and interface terms"
  nΓ = get_normal_vector(Γ)
  nΛ = get_normal_vector(Λ)

  "Define the weak form"
  ∇u(x)  = ∇(u_exact)(x)
  f(x)   = -Δ(u_exact)(x)
  a(u,v) = ∫(∇(u)⋅∇(v))dΩ
  l(v)   = ∫(f*v)dΩ
  
  "Define the residual error estimator
  It includes volume residual, boundary jump, and interface jump terms"
  ηh(u)  = l2_norm(hK*(f + Δ(u)),dΩ) +           # Volume residual
           l2_norm(hK*(∇(u) - ∇u)⋅nΓ,dΓ) +       # Boundary jump
           l2_norm(jump(hK*∇(u)⋅nΛ),dΛ)          # Interface jump
  
  "Solve the FE problem"
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)
  
  "Compute error indicators"
  η = estimate(ηh,uh)
  
  "Mark cells for refinement using Dörfler marking
  This strategy marks cells containing a fixed fraction (0.9) of the total error"
  m = DorflerMarking(0.9)
  I = Adaptivity.mark(m,η)
  
  "Refine the mesh using newest vertex bisection"
  method = Adaptivity.NVBRefinement(model)
  amodel = refine(method,model;cells_to_refine=I)
  fmodel = Adaptivity.get_model(amodel)

  "Compute the global error for convergence testing"
  error = sum(l2_norm(uh - u_exact,dΩ))
  return fmodel, uh, η, I, error
end

nsteps = 5
order = 1
model = LShapedModel(10)

last_error = Inf
for i in 1:nsteps
  fmodel, uh, η, I, error = amr_step(model,u_exact;order)
  
  is_refined = map(i -> ifelse(i ∈ I, 1, -1), 1:num_cells(model))
  
  Ω = Triangulation(model)
  writevtk(
    Ω,"model_$(i-1)",append=false,
    cellfields = [
      "uh" => uh,                    # Computed solution
      "η" => CellField(η,Ω),        # Error indicators
      "is_refined" => CellField(is_refined,Ω),  # Refinement markers
      "u_exact" => CellField(u_exact,Ω),       # Exact solution
    ],
  )
  
  println("Error: $error, Error η: $(sum(η))")
  global last_error = error
  global model = fmodel
end

