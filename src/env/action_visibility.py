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

    def _step(action: str, **kwargs) -> bool:
        try:
            event = controller.step(action=action, **kwargs)
        except Exception as exc:
            print(f"[Visibility] Failed to step {action}: {exc}")
            return False
        success = event.metadata.get("lastActionSuccess")
        if success is False:
            error_msg = event.metadata.get("errorMessage") or ""
            print(f"[Visibility] Step {action} failed: {error_msg}")
            return False
        return True

    for _ in range(rotations):
        if not _step("LookUp", degrees=look_angle):
            return False
        if object_id in _get_visible_object_ids(controller):
            _step("LookDown", degrees=look_angle)
            return True

        if not _step("LookDown", degrees=look_angle * 2):
            return False
        if object_id in _get_visible_object_ids(controller):
            _step("LookUp", degrees=look_angle)
            return True

        if not _step("LookUp", degrees=look_angle):
            return False
        if not _step("RotateRight", degrees=90):
            return False

        if object_id in _get_visible_object_ids(controller):
            return True

    return False


def ensure_object_visible(controller, object_id: str) -> bool:
    return scan_for_object(controller, object_id)


def should_check_visibility(action_name: str) -> bool:
    return action_name not in {"NavigateTo"}
