
using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays
using Plots

function cartesian_model(args...; kwargs...)
  UnstructuredDiscreteModel(CartesianDiscreteModel(args...; kwargs...))
end

function plot_node_numbering(node_coords, node_ids)
  x = map(c -> c[1], node_coords)
  y = map(c -> c[2], node_coords)
  a = text.(node_ids, halign=:left, valign=:bottom)
  scatter(x, y, series_annotations = a, legend=false)
  hline!(unique(x), linestyle=:dash, color=:grey)
  vline!(unique(y), linestyle=:dash, color=:grey)
end

function plot_node_numbering(model)
  D = num_cell_dims(model)
  topo = get_grid_topology(model)
  node_coords = Geometry.get_node_coordinates(model)
  cell_node_ids = get_cell_node_ids(model)
  cell_vertex_ids = Geometry.get_faces(topo, D, 0)

  node_to_vertex = zeros(Int, length(node_coords))
  for (nodes,vertices) in zip(cell_node_ids, cell_vertex_ids)
    node_to_vertex[nodes] .= vertices
  end
  
  plot_node_numbering(node_coords, node_to_vertex)
end

model = cartesian_model((0,1,0,1),(3,3))

topo = get_grid_topology(model)

n_vertices = num_faces(topo,0)  # Number of vertices (0-dimensional entities)
n_edges = num_faces(topo,1)     # Number of edges (1-dimensional entities)
n_cells = num_faces(topo,2)     # Number of cells (2-dimensional entities)

cell_to_vertices = get_faces(topo,2,0)

cell_to_edges = get_faces(topo,2,1)

edge_to_cells = get_faces(topo,1,2)

edge_to_vertices = get_faces(topo,1,0)

cell_to_cells = get_faces(topo,2,2)  # Returns identity table

function get_face_to_face_graph(topo,Df)
  n_faces = num_faces(topo,Df)
  face_to_vertices = get_faces(topo,Df,0)  # Get vertices of each face
  vertex_to_faces = get_faces(topo,0,Df)   # Get faces incident to each vertex

  face_to_face = Vector{Vector{Int}}(undef,n_faces)
  for face in 1:n_faces
    nbors = Int[]
    for vertex in face_to_vertices[face]
      append!(nbors,vertex_to_faces[vertex]) # Add incident faces
    end
    face_to_face[face] = filter(!isequal(face),unique(nbors)) # Remove self-reference and duplicates
  end

  return face_to_face
end

cell_to_cells = get_face_to_face_graph(topo,2)  # Cells sharing vertices
edge_to_edges = get_face_to_face_graph(topo,1)  # Edges sharing vertices

grid = get_grid(model)

cell_map = get_cell_map(grid)          # Mapping from reference to physical space
cell_to_nodes = get_cell_node_ids(grid) # Node IDs for each cell
node_coordinates = get_node_coordinates(grid) # Physical coordinates of nodes

cell_to_node_coords = map(nodes -> node_coordinates[nodes], cell_to_nodes)

cell_to_node_coords = lazy_map(Broadcasting(Reindex(node_coordinates)),cell_to_nodes)

cell_reffes = get_cell_reffe(grid)     # Get reference elements for each cell
cell_basis = lazy_map(get_shapefuns,cell_reffes)  # Get basis functions
cell_map = lazy_map(linear_combination,cell_to_node_coords,cell_basis)

function F(x)
  θ = x[1]*pi   # Map x-coordinate to angle [0,π]
  z = x[2]      # Keep y-coordinate as height
  VectorValue(cos(θ),sin(θ),z)  # Convert to cylindrical coordinates
end

new_node_coordinates = map(F,node_coordinates)

new_cell_to_node_coords = lazy_map(Broadcasting(Reindex(new_node_coordinates)),cell_to_nodes)
new_cell_map = lazy_map(linear_combination,new_cell_to_node_coords,cell_basis)

reffes, cell_types = compress_cell_data(cell_reffes)
new_grid = UnstructuredGrid(new_node_coordinates,cell_to_nodes,reffes,cell_types)

writevtk(new_grid,"half_cylinder_linear")

order = 2  # Polynomial order
new_reffes = [LagrangianRefFE(Float64,QUAD,order)]  # Quadratic quadrilateral elements
new_cell_reffes = expand_cell_data(new_reffes,cell_types)

space = FESpace(model,new_cell_reffes)
new_cell_to_nodes = get_cell_dof_ids(space)

cell_dofs = lazy_map(get_dof_basis,new_cell_reffes)
cell_basis = lazy_map(get_shapefuns,new_cell_reffes)
cell_to_ref_coordinates = lazy_map(get_nodes,cell_dofs)

cell_to_phys_coordinates = lazy_map(evaluate,cell_map,cell_to_ref_coordinates)

new_n_nodes = maximum(maximum,new_cell_to_nodes)
new_node_coordinates = zeros(VectorValue{2,Float64},new_n_nodes)
for (cell,nodes) in enumerate(new_cell_to_nodes)
  for (i,node) in enumerate(nodes)
    new_node_coordinates[node] = cell_to_phys_coordinates[cell][i]
  end
end

new_node_coordinates = map(F,new_node_coordinates)

new_grid = UnstructuredGrid(new_node_coordinates,new_cell_to_nodes,new_reffes,cell_types)
writevtk(new_grid,"half_cylinder_quadratic")

model = cartesian_model((0,1,0,1),(3,3))
plot_node_numbering(model)

model = cartesian_model((0,1,0,1),(3,3),isperiodic=(true,false))
plot_node_numbering(model)

model = cartesian_model((0,1,0,1),(3,3),isperiodic=(true,true))
plot_node_numbering(model)

nc = (3,3)  # Number of cells in each direction
model = cartesian_model((0,1,0,1),nc)

node_coords = get_node_coordinates(model)  # Physical positions
cell_node_ids = get_cell_node_ids(model)  # Node connectivity
cell_type = get_cell_type(model)          # Element type
reffes = get_reffes(model)                # Reference elements

np = nc .+ 1  # Number of points in each direction
mobius_ids = collect(LinearIndices(np))

mobius_ids[end,:] = reverse(mobius_ids[1,:])

cell_vertex_ids = map(nodes -> mobius_ids[nodes], cell_node_ids)

vertex_to_node = unique(vcat(cell_vertex_ids...))
node_to_vertex = find_inverse_index_map(vertex_to_node)

cell_vertex_ids = map(nodes -> node_to_vertex[nodes], cell_vertex_ids)

cell_vertex_ids = Table(cell_vertex_ids)

vertex_coords = node_coords[vertex_to_node]
polytopes = map(get_polytope,reffes)

topo = UnstructuredGridTopology(
  vertex_coords, cell_vertex_ids, cell_type, polytopes
)

grid = UnstructuredGrid(
  node_coords, cell_node_ids, reffes, cell_type
)

labels = FaceLabeling(topo)

mobius = UnstructuredDiscreteModel(grid,topo,labels)

plot_node_numbering(mobius)

model = cartesian_model((0,1,0,1),(3,3))
topo = get_grid_topology(model)

labels = FaceLabeling(topo)

tag_names = get_tag_name(labels) # Each name is a string
tag_entities = get_tag_entities(labels) # For each tag, a vector of entities
cell_to_entity = get_face_entity(labels,2) # For each cell, its associated entity
edge_to_entity = get_face_entity(labels,1) # For each edge, its associated entity
node_to_entity = get_face_entity(labels,0) # For each node, its associated entity

writevtk(model,"labels_basic",labels=labels)

cell_to_tag = [1,1,1,2,2,3,2,2,3]
tag_to_name = ["A","B","C"]
labels_cw = Geometry.face_labeling_from_cell_tags(topo,cell_to_tag,tag_to_name)
writevtk(model,"labels_cellwise",labels=labels_cw)

vfilter(x) = abs(x[1]- 1.0) < 1.e-5
labels_vf = Geometry.face_labeling_from_vertex_filter(topo, "top", vfilter)
writevtk(model,"labels_filter",labels=labels_vf)

labels = merge!(labels, labels_cw, labels_vf)
writevtk(model,"labels_merged",labels=labels)

cell_to_tag = [1,1,1,2,2,3,2,2,3]
tag_to_name = ["A","B","C"]
labels = Geometry.face_labeling_from_cell_tags(topo,cell_to_tag,tag_to_name)

Geometry.add_tag_from_tags!(labels,"A∪B",["A","B"])

Geometry.add_tag_from_tags_intersection!(labels,"A∩B",["A","B"])

Geometry.add_tag_from_tags_complementary!(labels,"!A",["A"])

Geometry.add_tag_from_tags_setdiff!(labels,"A-B",["A"],["B"]) # set difference

writevtk(model,"labels_setops",labels=labels)

face_dim = 1
mask = get_face_mask(labels,["A","C"],face_dim) # Boolean mask
ids = findall(mask) # Edge IDs
