@tool
extends MeshInstance3D

# Attach to a MeshInstance3D at the world origin (identity transform).
# Needs Godot 4.4+ (for @export_tool_button).
@export_file("*.ply") var ply_path: String = "res://laser_points/laser_points.ply"
@export var point_size := 3.0

@export_tool_button("Load points", "Load") var load_button = load_points
@export_tool_button("Clear points", "Remove") var clear_button = clear_points


func clear_points() -> void:
	mesh = null


func load_points() -> void:
	var file := FileAccess.open(ply_path, FileAccess.READ)
	if file == null:
		push_error("Could not open %s" % ply_path)
		return

	# Parse the ASCII PLY: skip the header, then read "x y z" lines.
	var verts := PackedVector3Array()
	var in_header := true
	while not file.eof_reached():
		var line := file.get_line()
		if in_header:
			if line.begins_with("end_header"):
				in_header = false
			continue
		var parts := line.split(" ", false)
		if parts.size() < 3:
			continue
		verts.append(Vector3(float(parts[0]), float(parts[1]), float(parts[2])))

	if verts.is_empty():
		push_error("No points found in %s" % ply_path)
		return

	# Colour by height (blue = low, red = high).
	var min_y := INF
	var max_y := -INF
	for v in verts:
		min_y = minf(min_y, v.y)
		max_y = maxf(max_y, v.y)
	var span := maxf(max_y - min_y, 0.0001)

	var colors := PackedColorArray()
	colors.resize(verts.size())
	for i in verts.size():
		var t := (verts[i].y - min_y) / span
		colors[i] = Color.from_hsv(0.66 * (1.0 - t), 1.0, 1.0)

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_COLOR] = colors

	var array_mesh := ArrayMesh.new()
	array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)

	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.vertex_color_use_as_albedo = true
	mat.use_point_size = true
	mat.point_size = point_size
	array_mesh.surface_set_material(0, mat)

	mesh = array_mesh
	print("Loaded %d points" % verts.size())
