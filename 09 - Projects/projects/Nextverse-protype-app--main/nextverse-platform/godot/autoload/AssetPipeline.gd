extends Node
## Tripio is a mobile image-to-3D asset generator. This autoload is the import
## contract for renders produced there (GLB/GLTF/FBX/OBJ) dropped into
## res://assets/tripio/. It does NOT provide networking or presence.

const IMPORT_DIRECTORY: String = "res://assets/tripio"
## Tripo AI exports GLB / FBX / OBJ / USD(SZ) for engines. STL and 3MF are
## print formats and are intentionally not accepted by the runtime pipeline.
const ALLOWED_EXTENSIONS: Array[String] = ["glb", "gltf", "fbx", "obj", "usd", "usdz"]

var assets: Dictionary = {}

func validate_path(path: String) -> bool:
	if path.is_empty():
		return false
	return ALLOWED_EXTENSIONS.has(path.get_extension().to_lower())

func register_asset(asset_id: String, path: String, source: String = "tripo_mobile",
		license_note: String = "cc-by-4.0-free-plan") -> bool:
	if asset_id.is_empty() or not validate_path(path):
		return false
	assets[asset_id] = {
		"path": path,
		"source": source,
		"license": license_note,
	}
	return true

func has_asset(asset_id: String) -> bool:
	return assets.has(asset_id)

func asset_count() -> int:
	return assets.size()

func manifest() -> Array:
	return assets.values().duplicate(true)
