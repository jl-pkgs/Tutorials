

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using FillArrays
using Test
using InteractiveUtils

L = 2 # Domain length in each space dimension
D = 2 # Number of spatial dimensions
n = 4 # Partition (i.e., number of cells per space dimension)

function stretching(x::Point)
   m = zeros(length(x))
   m[1] = x[1]^2
   for i in 2:D
     m[i] = x[i]
   end
   Point(m)
end

pmin = Point(Fill(0,D))
pmax = Point(Fill(L,D))
partition = Tuple(Fill(n,D))
model = CartesianDiscreteModel(pmin,pmax,partition,map=stretching)

T = Float64
order = 1
pol = Polytope(Fill(HEX_AXIS,D)...)
reffe = LagrangianRefFE(T,pol,order)

Vₕ = FESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
u(x) = x[1]            # Analytical solution (for Dirichlet data)
Uₕ = TrialFESpace(Vₕ,u)

Tₕ = Triangulation(model)
Qₕ = CellQuadrature(Tₕ,4*order)

isa(Qₕ,CellDatum)

subtypes(CellDatum)

Qₕ_cell_data = get_data(Qₕ)
@test length(Qₕ_cell_data) == num_cells(Tₕ)

DomainStyle(Qₕ) == ReferenceDomain()
DomainStyle(Qₕ) == PhysicalDomain()

q = Qₕ_cell_data[rand(1:num_cells(Tₕ))]
p = get_coordinates(q)
w = get_weights(q)

Qₕ_cell_point = get_cell_points(Qₕ)

@test isa(Qₕ_cell_point, CellDatum)

@test DomainStyle(Qₕ_cell_point) == ReferenceDomain()
qₖ = get_data(Qₕ_cell_point)

dv = get_fe_basis(Vₕ)
du = get_trial_fe_basis(Uₕ)

@test Gridap.FESpaces.BasisStyle(dv) == Gridap.FESpaces.TestBasis()
@test Gridap.FESpaces.BasisStyle(du) == Gridap.FESpaces.TrialBasis()

@test isa(dv,CellField) && isa(dv,CellDatum)
@test isa(du,CellField) && isa(du,CellDatum)

@test DomainStyle(dv) == ReferenceDomain()
@test DomainStyle(du) == ReferenceDomain()

dv_at_Qₕ = evaluate(dv,Qₕ_cell_point)
du_at_Qₕ = evaluate(du,Qₕ_cell_point)

dv_at_Qₕ[rand(1:num_cells(Tₕ))]
du_at_Qₕ[rand(1:num_cells(Tₕ))]

dv_mult_du = du*dv
dv_mult_du_at_Qₕ = evaluate(dv_mult_du,Qₕ_cell_point)

m=Broadcasting(*)
A=evaluate(m,dv_at_Qₕ[rand(1:num_cells(Tₕ))],du_at_Qₕ[rand(1:num_cells(Tₕ))])
B=broadcast(*,dv_at_Qₕ[rand(1:num_cells(Tₕ))],du_at_Qₕ[rand(1:num_cells(Tₕ))])
@test all(A .≈ B)
@test all(A .≈ dv_mult_du_at_Qₕ[rand(1:num_cells(Tₕ))])

dv_array = get_data(dv)
du_array = get_data(du)
@test isa(dv_array,AbstractVector{<:AbstractVector{<:Field}})
@test isa(du_array,AbstractVector{<:AbstractArray{<:Field,2}})
@test length(dv_array) == num_cells(Tₕ)
@test length(du_array) == num_cells(Tₕ)

ϕ₃ = dv_array[1][3]
evaluate(ϕ₃,[Point(0,0),Point(1,0),Point(0,1),Point(1,1)])

ϕ = dv_array[1]
evaluate(ϕ,[Point(0,0),Point(1,0),Point(0,1),Point(1,1)])

dv_array_at_qₖ = lazy_map(evaluate,dv_array,qₖ)
du_array_at_qₖ = lazy_map(evaluate,du_array,qₖ)

uₕ = FEFunction(Uₕ,rand(num_free_dofs(Uₕ)))

@test isa(uₕ,CellField)

@test DomainStyle(uₕ) == ReferenceDomain()

uₕ_at_Qₕ = evaluate(uₕ,Qₕ_cell_point)

@test isa(uₕ_at_Qₕ,LazyArray)

print(typeof(uₕ_at_Qₕ))

uₕ_array = get_data(uₕ)

@test isa(uₕ_array,AbstractVector{<:Field})

@test isa(uₕ_array,Gridap.Fields.LazyArray)

print(typeof(uₕ_array))

uₕ³ = uₕ_array[3]
@test isa(uₕ³,Field)

Uₖ = get_cell_dof_values(uₕ)

Uₖ³ = Uₖ[3]
ϕₖ³ = dv_array[3]

manual_uₕ³ = linear_combination(Uₖ³,ϕₖ³)

@test evaluate(uₕ³,qₖ[3]) ≈ evaluate(manual_uₕ³,qₖ[3])

manual_uₕ_array = [linear_combination(Uₖ[i],dv_array[i]) for i=1:num_cells(Tₕ)]

manual_uₕ_array_at_qₖ = lazy_map(evaluate,manual_uₕ_array,qₖ)

@test all( uₕ_at_Qₕ .≈ manual_uₕ_array_at_qₖ )

@test typeof(uₕ_at_Qₕ) != typeof(manual_uₕ_array_at_qₖ)

uₕ_array_at_qₖ = lazy_map(evaluate,uₕ_array,qₖ)

@test typeof(uₕ_array_at_qₖ) == typeof(uₕ_at_Qₕ)

function smart_sum(a::LazyArray)
  cache=array_cache(a)             #Create cache out of a
  sum=copy(getindex!(cache,a,1))   #We have to copy the output
  for i in 2:length(a)
    ai = getindex!(cache,a,i)      #Compute the i-th entry of a
    sum .= sum .+ ai
  end
  sum
end

smart_sum(uₕ_array_at_qₖ)        # Execute once before to neglect JIT-compilation time
smart_sum(manual_uₕ_array_at_qₖ) # Execute once before to neglect JIT-compilation time
@time begin
        for i in 1:100_000
         smart_sum(uₕ_array_at_qₖ)
        end
      end
@time begin
        for i in 1:100_000
          smart_sum(manual_uₕ_array_at_qₖ)
        end
      end

print_op_tree(uₕ_array_at_qₖ)
print_op_tree(manual_uₕ_array_at_qₖ)

uₕ_free_dof_values = get_free_dof_values(uₕ)
uₕ_dirichlet_dof_values = get_dirichlet_dof_values(Uₕ)

σₖ = get_cell_dof_ids(Uₕ)

m = Broadcasting(PosNegReindex(uₕ_free_dof_values,uₕ_dirichlet_dof_values))
manual_Uₖ = lazy_map(m,σₖ)

@test evaluate(PosNegReindex(uₕ_free_dof_values,uₕ_dirichlet_dof_values),3) == uₕ_free_dof_values[3]
@test evaluate(PosNegReindex(uₕ_free_dof_values,uₕ_dirichlet_dof_values),-7) == uₕ_dirichlet_dof_values[7]

ξₖ = get_cell_map(Tₕ)

X = get_node_coordinates(Tₕ)

cell_node_ids = get_cell_node_ids(Tₕ)

_Xₖ = get_cell_coordinates(Tₕ)

pol = Polytope(Fill(HEX_AXIS,D)...)
reffe_g = LagrangianRefFE(Float64,pol,1)

ϕrg = get_shapefuns(reffe_g)

ϕrgₖ = Fill(ϕrg,num_cells(Tₕ))

Xₖ = lazy_map(Broadcasting(Reindex(X)),cell_node_ids)

@test evaluate(Reindex(X),3) == X[3]

@test Xₖ == _Xₖ == get_cell_coordinates(Tₕ) # check

ψₖ = lazy_map(linear_combination,Xₖ,ϕrgₖ)

@test lazy_map(evaluate,ψₖ,qₖ) == lazy_map(evaluate,ξₖ,qₖ) # check

∇ϕrg  = Broadcasting(∇)(ϕrg)
∇ϕrgₖ = Fill(∇ϕrg,num_cells(model))
J = lazy_map(linear_combination,Xₖ,∇ϕrgₖ)

lazy_map(Broadcasting(∇),ψₖ)

@test typeof(J) == typeof(lazy_map(Broadcasting(∇),ψₖ))
@test lazy_map(evaluate,J,qₖ) == lazy_map(evaluate,lazy_map(Broadcasting(∇),ψₖ),qₖ)

grad_dv = ∇(dv)
grad_du = ∇(du)

@test isa(grad_dv, Gridap.FESpaces.FEBasis)
@test isa(grad_du, Gridap.FESpaces.FEBasis)

grad_dv_array = get_data(grad_dv)
grad_du_array = get_data(grad_du)

@test DomainStyle(grad_dv) == ReferenceDomain()
@test DomainStyle(grad_du) == ReferenceDomain()

ϕr                   = get_shapefuns(reffe)
∇ϕr                  = Broadcasting(∇)(ϕr)
∇ϕrₖ                 = Fill(∇ϕr,num_cells(Tₕ))
manual_grad_dv_array = lazy_map(Broadcasting(push_∇),∇ϕrₖ,ξₖ)
∇ϕrᵀ                 = Broadcasting(∇)(transpose(ϕr))
∇ϕrₖᵀ                = Fill(∇ϕrᵀ,num_cells(Tₕ))
manual_grad_du_array = lazy_map(Broadcasting(push_∇),∇ϕrₖᵀ,ξₖ)

Jt     = lazy_map(Broadcasting(∇),ξₖ)

inv_Jt = lazy_map(Operation(inv),Jt)

low_level_manual_gradient_dv_array = lazy_map(Broadcasting(Operation(⋅)),inv_Jt,∇ϕrₖ)

@test typeof(grad_dv_array) == typeof(manual_grad_dv_array)
@test lazy_map(evaluate,grad_dv_array,qₖ) == lazy_map(evaluate,manual_grad_dv_array,qₖ)
@test lazy_map(evaluate,grad_dv_array,qₖ) == lazy_map(evaluate,low_level_manual_gradient_dv_array,qₖ)
@test lazy_map(evaluate,grad_dv_array,qₖ) == evaluate(grad_dv,Qₕ_cell_point)

ϕrₖ = Fill(ϕr,num_cells(Tₕ))
∇ϕₖ = manual_grad_dv_array
uₖ  = lazy_map(linear_combination,Uₖ,ϕrₖ)
∇uₖ = lazy_map(linear_combination,Uₖ,∇ϕₖ)

intg = ∇(uₕ)⋅∇(dv)

Iₖ = lazy_map(Broadcasting(Operation(⋅)),∇uₖ,∇ϕₖ)

@test all(lazy_map(evaluate,Iₖ,qₖ) .≈ lazy_map(evaluate,get_data(intg),qₖ))

res = integrate(∇(uₕ)⋅∇(dv),Qₕ)

Jq = lazy_map(evaluate,J,qₖ)
intq = lazy_map(evaluate,Iₖ,qₖ)
iwq = lazy_map(IntegrationMap(),intq,Qₕ.cell_weight,Jq)
@test all(res .≈ iwq)

collect(iwq)

cellvals = ∫( ∇(dv)⋅∇(uₕ) )*Qₕ

@test all(cellvals .≈ iwq)

assem = SparseMatrixAssembler(Uₕ,Vₕ)

rs = ([iwq],[σₖ])
b = allocate_vector(assem,rs)
assemble_vector!(b,assem,rs)

∇ϕₖᵀ = manual_grad_du_array
int = lazy_map(Broadcasting(Operation(⋅)),∇ϕₖ,∇ϕₖᵀ)
@test all(collect(lazy_map(evaluate,int,qₖ)) .==
            collect(lazy_map(evaluate,get_data(∇(du)⋅∇(dv)),qₖ)))
intq = lazy_map(evaluate,int,qₖ)
Jq = lazy_map(evaluate,J,qₖ)
iwq = lazy_map(IntegrationMap(),intq,Qₕ.cell_weight,Jq)
jac = integrate(∇(dv)⋅∇(du),Qₕ)
@test collect(iwq) == collect(jac)
rs = ([iwq],[σₖ],[σₖ])
A = allocate_matrix(assem,rs)
A = assemble_matrix!(A,assem,rs)

x = A \ b
uf = get_free_dof_values(uₕ) - x
ufₕ = FEFunction(Uₕ,uf)
@test sum(integrate((u-ufₕ)*(u-ufₕ),Qₕ)) <= 10^-8

@test ∑(∫(((u-ufₕ)*(u-ufₕ)))Qₕ) <= 10^-8
