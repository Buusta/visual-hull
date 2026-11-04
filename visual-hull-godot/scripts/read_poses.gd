@tool
extends Node

# Attach to a Node in the scene whose origin = the ArUco marker.
# Needs Godot 4.4+ (for @export_tool_button).
@export_file("*.json") var poses_path: String = "res://camera_poses.json"

@export_tool_button("Place cameras", "Camera3D") var place_button = place_cameras
@export_tool_button("Clear cameras", "Remove") var clear_button = clear_cameras


func clear_cameras() -> void:
	for child in get_children():
		if child.has_meta("pose_camera"):
			remove_child(child)
			child.queue_free()


func place_cameras() -> void:
	clear_cameras()

	var file := FileAccess.open(poses_path, FileAccess.READ)
	if file == null:
		push_error("Could not open %s" % poses_path)
		return

	var poses = JSON.parse_string(file.get_as_text())
	if typeof(poses) != TYPE_ARRAY:
		push_error("Invalid JSON in %s" % poses_path)
		return

	var scene_root := get_tree().edited_scene_root

	for pose in poses:
		var cam := Camera3D.new()
		cam.name = String(pose["image"]).get_basename()
		add_child(cam)
		cam.owner = scene_root
		cam.set_meta("pose_camera", true)
		cam.set_meta("image", pose["image"])

		var p: Array = pose["position"]
		var r: Array = pose["rotation_degrees"]
		cam.position = Vector3(p[0], p[1], p[2])
		# Default rotation order is YXZ, matching the export.
		cam.rotation_degrees = Vector3(r[0], r[1], r[2])

		cam.keep_aspect = Camera3D.KEEP_HEIGHT
		cam.fov = pose["fov_vertical_degrees"]

	print("Placed %d cameras" % poses.size())
