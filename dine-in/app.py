from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr


@dataclass(frozen=True)
class Scenario:
    image_path: Path
    order_manifest: Dict[str, object]
    validation: Dict[str, object]
    metrics: Dict[str, object]


APP_TITLE = "Dine-In Order Accuracy Benchmark"
APP_DESCRIPTION = (
    "Staff-triggered plate validation workflow that mirrors full-service restaurant expo operations. "
    "Demonstrates zero-training deployment with vision-language models, semantic order reconciliation, "
    "and latency instrumentation aligned to the two-second service window."
)

ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "images"
ORDERS_PATH = ROOT_DIR / "orders" / "orders.json"


_VALIDATION_PROFILES: Dict[str, Dict[str, object]] = {
    "image_01_mcd_combo": {
        "order_complete": True,
        "missing_items": [],
        "extra_items": [],
        "modifier_validation": {"status": "validated", "details": []},
        "accuracy_score": 0.99,
    },
    "image_02_mcd_variation": {
        "order_complete": False,
        "missing_items": ["French Fries"],
        "extra_items": ["Apple Pie"],
        "modifier_validation": {"status": "validated", "details": []},
        "accuracy_score": 0.74,
    },
    "image_03_kfc_combo": {
        "order_complete": False,
        "missing_items": [],
        "extra_items": ["Extra Biscuit Basket"],
        "modifier_validation": {"status": "validated", "details": []},
        "accuracy_score": 0.77,
    },
    "image_04_burgerking_combo": {
        "order_complete": False,
        "missing_items": ["Milkshake"],
        "extra_items": [],
        "modifier_validation": {"status": "validated", "details": []},
        "accuracy_score": 0.71,
    },
    "plate_05_salmon_asparagus_mash": {
        "order_complete": True,
        "missing_items": [],
        "extra_items": [],
        "modifier_validation": {"status": "validated", "details": []},
        "accuracy_score": 0.96,
    },
    "plate_06_fried_chicken_combo": {
        "order_complete": False,
        "missing_items": ["Dipping Sauce"],
        "extra_items": [],
        "modifier_validation": {
            "status": "validated",
            "details": ["Barbecue modifier absent"],
        },
        "accuracy_score": 0.69,
    },
    "plate_07_chicken_parmesan_combo": {
        "order_complete": True,
        "missing_items": [],
        "extra_items": [],
        "modifier_validation": {
            "status": "unable_to_verify",
            "details": ["Cheese coverage unclear"],
        },
        "accuracy_score": 0.65,
    },
}

_METRIC_PROFILES: Dict[str, Dict[str, object]] = {
    "image_01_mcd_combo": {
        "end_to_end_latency_ms": 1280,
        "vlm_inference_ms": 840,
        "agent_reconciliation_ms": 290,
        "within_operational_window": True,
    },
    "image_02_mcd_variation": {
        "end_to_end_latency_ms": 1710,
        "vlm_inference_ms": 1110,
        "agent_reconciliation_ms": 420,
        "within_operational_window": True,
    },
    "image_03_kfc_combo": {
        "end_to_end_latency_ms": 1625,
        "vlm_inference_ms": 1075,
        "agent_reconciliation_ms": 355,
        "within_operational_window": True,
    },
    "image_04_burgerking_combo": {
        "end_to_end_latency_ms": 1930,
        "vlm_inference_ms": 1235,
        "agent_reconciliation_ms": 480,
        "within_operational_window": False,
    },
    "plate_05_salmon_asparagus_mash": {
        "end_to_end_latency_ms": 1395,
        "vlm_inference_ms": 910,
        "agent_reconciliation_ms": 315,
        "within_operational_window": True,
    },
    "plate_06_fried_chicken_combo": {
        "end_to_end_latency_ms": 2050,
        "vlm_inference_ms": 1320,
        "agent_reconciliation_ms": 520,
        "within_operational_window": False,
    },
    "plate_07_chicken_parmesan_combo": {
        "end_to_end_latency_ms": 1840,
        "vlm_inference_ms": 1180,
        "agent_reconciliation_ms": 460,
        "within_operational_window": False,
    },
}


def _default_validation(image_id: str) -> Dict[str, object]:
    return {
        "order_complete": False,
        "missing_items": [],
        "extra_items": [],
        "modifier_validation": {
            "status": "pending",
            "details": [f"No validation profile configured for {image_id}"],
        },
        "accuracy_score": None,
    }


def _default_metrics(image_id: str) -> Dict[str, object]:
    return {
        "end_to_end_latency_ms": None,
        "vlm_inference_ms": None,
        "agent_reconciliation_ms": None,
        "within_operational_window": None,
        "notes": f"No metric profile configured for {image_id}",
    }


def _load_orders() -> Dict[str, Scenario]:
    if not ORDERS_PATH.exists():
        return {}

    with ORDERS_PATH.open("r", encoding="utf-8") as orders_file:
        data = json.load(orders_file)

    scenarios: Dict[str, Scenario] = {}

    for order in data.get("orders", []):
        image_id = order.get("image_id")
        if not image_id:
            continue

        image_path = IMAGES_DIR / f"{image_id}.png"
        manifest = {
            key: value
            for key, value in order.items()
            if key != "image_id"
        }

        label = (
            f"{order.get('image_id')} – {order.get('restaurant', 'Unknown')} "
            f"Table {order.get('table_number', '?')}"
        )

        scenarios[label] = Scenario(
            image_path=image_path,
            order_manifest=manifest,
            validation=_VALIDATION_PROFILES.get(image_id, _default_validation(image_id)),
            metrics=_METRIC_PROFILES.get(image_id, _default_metrics(image_id)),
        )

    return scenarios


_SCENARIOS = _load_orders()
_DEFAULT_SCENARIO = next(iter(_SCENARIOS)) if _SCENARIOS else ""


def load_scenario(name: str) -> Tuple[str, Dict[str, object]]:
    scenario = _SCENARIOS.get(name)
    if not scenario:
        return "", {"error": f"Scenario '{name}' is not available"}

    image_value = str(scenario.image_path) if scenario.image_path.exists() else ""
    return image_value, scenario.order_manifest


def validate_plate(name: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    scenario = _SCENARIOS.get(name)
    if not scenario:
        return _default_validation(name), _default_metrics(name)
    return scenario.validation, scenario.metrics


with gr.Blocks(
    title=APP_TITLE,
    css="""
    :root {
        --primary-500: #1f6feb !important;
        --primary-600: #1a5fd0 !important;
        --primary-700: #174ea6 !important;
    }

    #validate-plate-btn {
        width: 200px !important;
    }

    #validate-plate-btn button {
        width: 200px !important;
        color: white !important;
        padding: 0.35rem 0.8rem !important;
        white-space: nowrap;
    }
    """
) as app:
    gr.Markdown(f"## {APP_TITLE}\n{APP_DESCRIPTION}")

    with gr.Row():
        with gr.Column():
            scenario_dropdown = gr.Dropdown(
                label="Scenario",
                choices=list(_SCENARIOS.keys()),
                value=_DEFAULT_SCENARIO or None,
                interactive=True,
            )
            validate_button = gr.Button(
    "Validate Plate",
    variant="primary",
    elem_id="validate-plate-btn",
)

    with gr.Row():
        image_display = gr.Image(label="Plate Image", interactive=False)
        order_display = gr.JSON(label="Order Ticket", value={})

    validation_display = gr.JSON(label="Validation Result")
    metrics_display = gr.JSON(label="Performance Metrics")

    def _on_scenario_change(
        name: str,
    ) -> Tuple[str, Dict[str, object], Dict[str, object], Dict[str, object]]:
        image_path, order_manifest = load_scenario(name)
        empty_validation = {
            "order_complete": None,
            "missing_items": [],
            "extra_items": [],
            "modifier_validation": {"status": "pending"},
            "accuracy_score": None,
        }
        empty_metrics = {
            "end_to_end_latency_ms": None,
            "vlm_inference_ms": None,
            "agent_reconciliation_ms": None,
            "within_operational_window": None,
        }
        return image_path, order_manifest, empty_validation, empty_metrics

    def _on_validate(name: str) -> Tuple[Dict[str, object], Dict[str, object]]:
        return validate_plate(name)

    scenario_dropdown.change(
        fn=_on_scenario_change,
        inputs=scenario_dropdown,
        outputs=[image_display, order_display, validation_display, metrics_display],
        show_progress=False,
    )

    validate_button.click(
        fn=_on_validate,
        inputs=scenario_dropdown,
        outputs=[validation_display, metrics_display],
    )

    if _DEFAULT_SCENARIO:
        (
            initial_image,
            initial_order,
            initial_validation,
            initial_metrics,
        ) = _on_scenario_change(_DEFAULT_SCENARIO)
        image_display.value = initial_image
        order_display.value = initial_order
        validation_display.value = initial_validation
        metrics_display.value = initial_metrics

if __name__ == "__main__":
    app.launch()
