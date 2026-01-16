from __future__ import annotations

def _get_visible_object_ids(controller) -> set[str]:
    try:
        event = controller.step(action="GetVisibleObjects")
    except Exception as exc:
        print(f"[Visibility] Failed to query visible objects: {exc}")
        return set()
    visible = event.metadata.get("actionReturn") or []
    if not isinstance(visible, list):
        return set()
    return set(visible)


def scan_for_object(controller, object_id: str, rotations: int = 4, look_angle: int = 30) -> bool:
    if not object_id:
        return False

    if object_id in _get_visible_object_ids(controller):
        return True

    for _ in range(rotations):
        controller.step(action="LookUp", degrees=look_angle)
        if object_id in _get_visible_object_ids(controller):
            controller.step(action="LookDown", degrees=look_angle)
            return True

        controller.step(action="LookDown", degrees=look_angle * 2)
        if object_id in _get_visible_object_ids(controller):
            controller.step(action="LookUp", degrees=look_angle)
            return True

        controller.step(action="LookUp", degrees=look_angle)
        controller.step(action="RotateRight", degrees=90)

        if object_id in _get_visible_object_ids(controller):
            return True

    return False


def ensure_object_visible(controller, object_id: str) -> bool:
    return scan_for_object(controller, object_id)
