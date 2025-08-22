

using Gridap
using GridapDistributed
using PartitionedArrays

function main_ex1(rank_partition,distribute)
  parts  = distribute(LinearIndices((prod(rank_partition),)))
  domain = (0,1,0,1)
  mesh_partition = (4,4)
  model = CartesianDiscreteModel(parts,rank_partition,domain,mesh_partition)
  order = 2
  u((x,y)) = (x+y)^order
  f(x) = -Δ(u,x)
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v) = ∫( v*f )dΩ
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)
  writevtk(Ω,"results_ex1",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
end

rank_partition = (2,2)
with_mpi() do distribute
  main_ex1(rank_partition,distribute)
end

using GridapPETSc

function main_ex2(rank_partition,distribute)
  parts  = distribute(LinearIndices((prod(rank_partition),)))
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    domain = (0,1,0,1)
    mesh_partition = (4,4)
    model = CartesianDiscreteModel(parts,rank_partition,domain,mesh_partition)
    order = 2
    u((x,y)) = (x+y)^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags="boundary")
    U = TrialFESpace(u,V)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)
    solver = PETScLinearSolver()
    uh = solve(solver,op)
    writevtk(Ω,"results_ex2",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

rank_partition = (2,2)
with_mpi() do distribute
  main_ex2(rank_partition,distribute)
end

using GridapP4est

function main_ex3(nparts,distribute)
  parts   = distribute(LinearIndices((nparts,)))
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    domain = (0,1,0,1)
    coarse_mesh_partition = (1,1)
    num_uniform_refinements = 2
    coarse_discrete_model = CartesianDiscreteModel(domain,coarse_mesh_partition)
    model = UniformlyRefinedForestOfOctreesDiscreteModel(parts,
                                                       coarse_discrete_model,
                                                       num_uniform_refinements)
    order = 2
    u((x,y)) = (x+y)^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags="boundary")
    U = TrialFESpace(u,V)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)
    solver = PETScLinearSolver()
    uh = solve(solver,op)
    writevtk(Ω,"results_ex3",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

nparts = 4
with_mpi() do distribute
  main_ex3(nparts,distribute)
end

using GridapGmsh
function main_ex4(nparts,distribute)
  parts  = distribute(LinearIndices((nparts,)))
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    model = GmshDiscreteModel(parts,"../models/demo.msh")
    order = 2
    u((x,y)) = (x+y)^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags=["boundary1","boundary2"])
    U = TrialFESpace(u,V)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)
    solver = PETScLinearSolver()
    uh = solve(solver,op)
    writevtk(Ω,"results_ex4",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

nparts = 4
with_mpi() do distribute 
  main_ex4(nparts,distribute)
end
