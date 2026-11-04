extends Node

# Attach to any Node in the scene, then drag the node that holds the
# cameras (the one with place_cameras.gd) into "Cameras Root".
@export var cameras_root: Node

var cameras: Array[Camera3D] = []
var index := 0
var label: Label


func _ready() -> void:
	if cameras_root == null:
		push_error("Assign Cameras Root in the inspector.")
		return

	for child in cameras_root.get_children():
		if child is Camera3D:
			cameras.append(child)

	if cameras.is_empty():
		push_error("No cameras found. Click 'Place cameras' first.")
		return

	var layer := CanvasLayer.new()
	label = Label.new()
	label.position = Vector2(16, 16)
	layer.add_child(label)
	add_child(layer)

	_show_camera(0)


func _unhandled_input(event: InputEvent) -> void:
	if cameras.is_empty():
		return
	if event.is_action_pressed("ui_right"):
		_show_camera(index + 1)
	elif event.is_action_pressed("ui_left"):
		_show_camera(index - 1)
	elif event is InputEventKey and event.pressed and event.keycode == KEY_S:
		var img := get_viewport().get_texture().get_image()
		img.save_png("user://%s.png" % cameras[index].name)
		print("Saved ", ProjectSettings.globalize_path("user://%s.png" % cameras[index].name))


func _show_camera(i: int) -> void:
	index = posmod(i, cameras.size())
	cameras[index].make_current()
	label.text = "%s  (%d/%d)" % [cameras[index].name, index + 1, cameras.size()]
