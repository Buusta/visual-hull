extends Node

# Attach to any Node. Assign the exports in the inspector, then run the game.
#
# laser:   Node3D whose origin + local +Z define the laser plane.
#          The target is moved along that normal, so it passes through the plane.
# target:  Node3D to scan (its MeshInstance3D children get the laser shader).
# cameras: one or more Camera3D that record the laser line.
@export var target: Node3D
@export var cameras: Array[Camera3D] = []
@export var laser: Node3D
@export var steps := 100
@export var output_dir := "res://laser_scan"
@export var laser_thickness := 0.002      # world units
@export var laser_color := Color.RED
@export var black_out_target := true      # target renders black, so only the line is visible
@export var settle_frames := 2

const SHADER_CODE := """
shader_type spatial;
render_mode unshaded;
uniform vec3 plane_point;
uniform vec3 plane_normal;
uniform float thickness;
uniform vec4 color : source_color;
varying vec3 world_pos;
void vertex() {
	world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}
void fragment() {
	if (abs(dot(world_pos - plane_point, plane_normal)) > thickness) {
		discard;
	}
	ALBEDO = color.rgb;
}
"""


func _ready() -> void:
	await get_tree().process_frame
	await _scan()


func _scan() -> void:
	if target == null or laser == null or cameras.is_empty():
		push_error("Assign target, laser and at least one camera in the inspector.")
		return
	for cam in cameras:
		if cam == null:
			push_error("A camera slot is empty.")
			return

	var meshes := _get_meshes()
	if meshes.is_empty():
		push_error("Target has no MeshInstance3D.")
		return

	var n := laser.global_basis.z.normalized()
	var origin := laser.global_position

	# Extent of the target's bounding box along the plane normal.
	var aabb := _world_aabb(meshes)
	var pmin := INF
	var pmax := -INF
	for i in 8:
		var d := (aabb.get_endpoint(i) - origin).dot(n)
		pmin = minf(pmin, d)
		pmax = maxf(pmax, d)

	# Laser shader as overlay, optional black base material.
	var shader := Shader.new()
	shader.code = SHADER_CODE
	var laser_mat := ShaderMaterial.new()
	laser_mat.shader = shader
	laser_mat.set_shader_parameter("plane_point", origin)
	laser_mat.set_shader_parameter("plane_normal", n)
	laser_mat.set_shader_parameter("thickness", laser_thickness)
	laser_mat.set_shader_parameter("color", laser_color)

	var black := StandardMaterial3D.new()
	black.albedo_color = Color.BLACK
	black.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED

	var old_overrides := {}
	for m in meshes:
		m.material_overlay = laser_mat
		if black_out_target:
			old_overrides[m] = m.material_override
			m.material_override = black

	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(output_dir))

	# Plane cuts the object at slice c (measured along n from the original position).
	var displacements: Array[Vector3] = []
	var images_per_step := []
	for i in steps:
		var c := lerpf(pmin, pmax, (i + 0.5) / steps)
		displacements.append(-c * n)
		images_per_step.append([])

	var start_pos := target.global_position

	# One full pass per camera.
	for ci in cameras.size():
		cameras[ci].make_current()
		for i in steps:
			target.global_position = start_pos + displacements[i]

			for f in settle_frames:
				await RenderingServer.frame_post_draw

			var file_name := "cam%d_step_%04d.png" % [ci, i]
			get_viewport().get_texture().get_image().save_png(output_dir.path_join(file_name))
			images_per_step[i].append(file_name)
		print("Camera %d/%d done" % [ci + 1, cameras.size()])

	# Restore the scene.
	target.global_position = start_pos
	for m in meshes:
		m.material_overlay = null
		if black_out_target:
			m.material_override = old_overrides[m]

	var camera_data := []
	for cam in cameras:
		var t := cam.global_transform
		camera_data.append({
			"basis_x": _v(t.basis.x),
			"basis_y": _v(t.basis.y),
			"basis_z": _v(t.basis.z),
			"origin": _v(t.origin),
			"fov_degrees": cam.fov,
			"keep_aspect": "height" if cam.keep_aspect == Camera3D.KEEP_HEIGHT else "width",
		})

	var records := []
	for i in steps:
		records.append({
			"displacement": _v(displacements[i]),  # world point - displacement = point at original target position
			"images": images_per_step[i],          # one file per camera, same order as "cameras"
		})

	var data := {
		"viewport_size": [get_viewport().size.x, get_viewport().size.y],
		"cameras": camera_data,
		"laser_plane": {"point": _v(origin), "normal": _v(n)},
		"steps": records,
	}
	var f := FileAccess.open(output_dir.path_join("scan.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(data, "  "))
	f.close()

	print("Saved %d images + scan.json to %s" % [steps * cameras.size(), output_dir])


func _get_meshes() -> Array[MeshInstance3D]:
	var result: Array[MeshInstance3D] = []
	if target is MeshInstance3D:
		result.append(target)
	for node in target.find_children("*", "MeshInstance3D", true, false):
		result.append(node)
	return result


func _world_aabb(meshes: Array[MeshInstance3D]) -> AABB:
	var box := meshes[0].global_transform * meshes[0].get_aabb()
	for m in meshes:
		box = box.merge(m.global_transform * m.get_aabb())
	return box


func _v(v: Vector3) -> Array:
	return [v.x, v.y, v.z]
