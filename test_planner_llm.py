#!/usr/bin/env python3
"""Quick sanity test for Neural-TAMP planner LLM stack.

Run from the repo root (e.g. /workspace/Neural-TAMP):
  python test_planner_llm.py

Environment variables (optional):
  OPENAI_BASE_URL   default: http://127.0.0.1:8000/v1
  OPENAI_API_KEY    default: EMPTY
  PLANNER_MODEL     default: Qwen2.5-32B

This script is defensive:
  - It supports either module path: src.planner.* OR src.planning.*
  - It creates a minimal "scene graph" object if your SceneGraph constructor is complex.
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Optional, Tuple


def _http_get_json(url: str, timeout: float = 3.0) -> Tuple[int, Any]:
    """Tiny HTTP GET helper without extra dependencies."""
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return status, json.loads(body)
            except Exception:
                return status, body
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, body
    except Exception as e:
        return -1, str(e)


def _import_planner_modules():
    """Try importing from src.planner.* then src.planning.*."""
    candidates = [
        ("src.planner.llm_interface", "LLMInterface"),
        ("src.planner.prompt_builder", "PromptBuilder"),
        ("src.planner.decomposer", "TaskDecomposer"),
    ]
    fallback = [
        ("src.planning.llm_interface", "LLMInterface"),
        ("src.planning.prompt_builder", "PromptBuilder"),
        ("src.planning.decomposer", "TaskDecomposer"),
    ]

    def try_set(mod_path: str, symbol: str):
        m = __import__(mod_path, fromlist=[symbol])
        return getattr(m, symbol)

    # First try src.planner
    try:
        LLMInterface = try_set(candidates[0][0], candidates[0][1])
        PromptBuilder = try_set(candidates[1][0], candidates[1][1])
        TaskDecomposer = try_set(candidates[2][0], candidates[2][1])
        return "src.planner", LLMInterface, PromptBuilder, TaskDecomposer
    except Exception as e1:
        # Then try src.planning
        try:
            LLMInterface = try_set(fallback[0][0], fallback[0][1])
            PromptBuilder = try_set(fallback[1][0], fallback[1][1])
            TaskDecomposer = try_set(fallback[2][0], fallback[2][1])
            return "src.planning", LLMInterface, PromptBuilder, TaskDecomposer
        except Exception as e2:
            raise ImportError(
                "Failed to import planner modules.\n"
                "Tried: src.planner.* and src.planning.*\n\n"
                f"src.planner import error: {e1}\n"
                f"src.planning import error: {e2}\n\n"
                "Fix: ensure your files are under src/planner/ (or src/planning/) AND the imports inside decomposer.py match that package name."
            )


def _make_minimal_scene_graph():
    """Create a minimal scene graph object with attributes used by PromptBuilder.

    PromptBuilder only needs:
      - scene_graph.nodes: dict[str, node] where node has id, label, room_id, state
      - scene_graph.edges: list[edge] where edge has source_id, target_id, relation

    We'll create two rooms + three objects:
      Kitchen room, Living room
      Fridge (closed) in Kitchen
      Apple inside Fridge
      Table in Living room
    """
    kitchen = SimpleNamespace(id="KitchenRoom|1", label="KitchenRoom", room_id="KitchenRoom|1", state={})
    living = SimpleNamespace(id="LivingRoom|1", label="LivingRoom", room_id="LivingRoom|1", state={})
    fridge = SimpleNamespace(id="Fridge|1", label="Fridge", room_id="KitchenRoom|1", state={"open_state": "closed"})
    apple = SimpleNamespace(id="Apple|1", label="Apple", room_id="KitchenRoom|1", state={})
    table = SimpleNamespace(id="Table|1", label="Table", room_id="LivingRoom|1", state={})

    # Apple is inside Fridge
    e1 = SimpleNamespace(source_id="Fridge|1", target_id="Apple|1", relation="inside")

    nodes = {n.id: n for n in [kitchen, living, fridge, apple, table]}
    sg = SimpleNamespace(nodes=nodes, edges=[e1])
    return sg


def main() -> int:
    repo_root = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, repo_root)

    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    model = os.getenv("PLANNER_MODEL", "Qwen2.5-32B")

    print("=== Neural-TAMP Planner LLM Sanity Test ===")
    print(f"Repo root: {repo_root}")
    print(f"OPENAI_BASE_URL: {base_url}")
    print(f"PLANNER_MODEL: {model}")

    # 1) Imports
    print("\n[1/5] Importing planner modules...")
    try:
        pkg, LLMInterface, PromptBuilder, TaskDecomposer = _import_planner_modules()
        print(f"  ✅ Imported from: {pkg}")
        print(f"  - LLMInterface: {LLMInterface}")
        print(f"  - PromptBuilder: {PromptBuilder}")
        print(f"  - TaskDecomposer: {TaskDecomposer}")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return 2

    # 2) Server reachability
    print("\n[2/5] Checking vLLM/OpenAI server...")
    health_url = base_url.replace("/v1", "") + "/health"
    status, body = _http_get_json(health_url)
    if status == -1:
        print(f"  ❌ Cannot reach server at {health_url}: {body}")
        print("  Fix: start vLLM, or set OPENAI_BASE_URL.")
        return 3
    print(f"  ✅ /health status={status}, body={body}")

    models_url = base_url + "/models"
    status, models_body = _http_get_json(models_url)
    print(f"  ✅ /v1/models status={status}")
    if isinstance(models_body, dict) and "data" in models_body:
        served = [m.get("id") for m in models_body.get("data", []) if isinstance(m, dict)]
        print(f"  Served models: {served[:10]}")
        if model not in served:
            print(f"  ⚠️ Model '{model}' not listed. Your --served-model-name may differ.")
    else:
        print(f"  ⚠️ Unexpected /models response: {str(models_body)[:200]}")

    # 3) LLMInterface JSON parse
    print("\n[3/5] Testing LLMInterface JSON response...")
    llm = LLMInterface(model=model, base_url=base_url, api_key=api_key)
    test_prompt = (
        "Return a JSON object with keys: ok (bool), model (string), note (string)."
    )
    out = llm.predict("You are a strict JSON generator.", test_prompt)
    print("  Output:", out)
    if not isinstance(out, dict) or out.get("ok") is not True:
        print("  ⚠️ LLMInterface test did not return expected JSON. This may still be OK, but check your prompts.")

    # 4) PromptBuilder
    print("\n[4/5] Testing PromptBuilder prompt generation...")
    sg = _make_minimal_scene_graph()
    prompter = PromptBuilder()
    d_prompt = prompter.build_decomposition_prompt("Put the apple in the fridge.", sg)
    a_prompt = prompter.build_action_prompt([
        "Navigate to the Apple",
        "Pick up the Apple",
        "Navigate to the Fridge",
        "Open the Fridge",
        "Put the Apple inside the Fridge",
        "Close the Fridge",
    ], sg)
    print("  ✅ Decomposition prompt length:", len(d_prompt))
    print("  ✅ Action prompt length:", len(a_prompt))

    # 5) TaskDecomposer end-to-end
    print("\n[5/5] Testing TaskDecomposer.plan() end-to-end...")
    decomposer = TaskDecomposer(model_name=model, base_url=base_url, api_key=api_key)

    t0 = time.time()
    actions = decomposer.plan("Put the apple in the fridge.", sg)
    dt = time.time() - t0

    print(f"  ✅ plan() returned {len(actions)} actions in {dt:.2f}s")
    print("  Actions:")
    for i, a in enumerate(actions[:50], 1):
        print(f"    {i}. {a}")

    print("\n✅ All tests finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
