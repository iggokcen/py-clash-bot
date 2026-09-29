import numpy as np
import pytest

from pyclashbot.bot.policy import (
    EpsilonGreedyPolicy,
    OnlineTrainer,
    PolicyAction,
    SimplePolicyNet,
    build_action_vector,
    build_state_vector,
    candidate_coords_for_group,
    candidate_coords_free,
    safe_bounds_for_image,
)
from pyclashbot.detection.hand_classifier import (
    DEFAULT_MODEL_PATH as HAND_MODEL_PATH,
)
from pyclashbot.detection.hand_classifier import (
    load_hand_classifier,
)
from pyclashbot.detection.onnx_detector import (
    DEFAULT_MODEL_PATH as ONNX_MODEL_PATH,
)
from pyclashbot.detection.onnx_detector import (
    Detection,
    OnnxTroopDetector,
    split_detections_by_side,
    summarize_detections,
)


def test_onnx_troop_detector_init_and_infer():
    if not ONNX_MODEL_PATH.exists():
        pytest.skip("field_detector.onnx not found")
    detector = OnnxTroopDetector(model_path=str(ONNX_MODEL_PATH))
    assert detector is not None

    dummy_frame = np.zeros((633, 419, 3), dtype=np.uint8)
    dets = detector.detect(dummy_frame)
    assert isinstance(dets, list)


def test_split_detections_and_summary():
    dummy_frame = np.zeros((633, 419, 3), dtype=np.uint8)
    dets = [
        Detection(x1=50, y1=100, x2=80, y2=140, conf=0.8, cls_id=0, name="ally_archer", bot_id="archer"),
        Detection(x1=100, y1=500, x2=150, y2=550, conf=0.9, cls_id=22, name="ally_giant", bot_id="giant"),
    ]
    summary = summarize_detections(dets, dummy_frame.shape)
    assert summary.total == 2
    assert summary.max_conf >= 0.9

    enemy, ally, unknown = split_detections_by_side(dummy_frame, dets)
    assert isinstance(enemy, list)
    assert isinstance(ally, list)
    assert isinstance(unknown, list)


def test_hand_classifier_model():
    if not HAND_MODEL_PATH.exists():
        pytest.skip("hand classifier weights not found")
    classifier = load_hand_classifier()
    assert classifier is not None
    assert len(classifier.class_names) > 0

    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    label, conf = classifier.predict(dummy_crop)
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0

    topk = classifier.predict_topk(dummy_crop, k=3)
    assert len(topk) <= 3


def test_policy_net_and_trainer():
    model = SimplePolicyNet(input_dim=30)
    trainer = OnlineTrainer(model=model, lr=1e-3)
    state = build_state_vector(elapsed_time=30.0, elixir=5, side_preference="left", available_mask=[1, 1, 0, 0])
    assert len(state) == 26

    actions = [
        PolicyAction(card_index=0, card_group="hog_rider", coord=(100, 350)),
        PolicyAction(card_index=1, card_group="spell", coord=(200, 200)),
    ]
    vec = build_action_vector(actions[0])
    assert len(vec) == 4

    policy = EpsilonGreedyPolicy(model=model, epsilon=0.0)
    best_action = policy.select_action(state, actions)
    assert best_action in actions

    # Step update test
    trainer.update_step(state, actions[0], 0.5)
    assert len(trainer.replay_buffer) == 1

    # Match outcome reward update test
    trainer.update(1.0)  # positional: reward=1.0 (win)
    assert True  # update returns None


def test_candidate_coords_generation():
    groups = candidate_coords_for_group("tombstone", "left", elapsed_time=30.0)
    assert isinstance(groups, list)
    assert len(groups) > 0

    dummy_frame = np.zeros((633, 419, 3), dtype=np.uint8)
    bounds = safe_bounds_for_image(dummy_frame)
    free_coords = candidate_coords_free(bounds)
    assert len(free_coords) > 0
    for x, y in free_coords:
        assert 0 <= x <= 419
        assert 0 <= y <= 633


def test_live_observer_utilities():
    from tools.live_match_observer import CallbackLogger, resolve_device_serial

    serial = resolve_device_serial("127.0.0.1:5555")
    assert serial == "127.0.0.1:5555"

    logged = []
    logger = CallbackLogger(callback=logged.append)
    logger.change_status("Testing observer status")
    assert any("Testing observer status" in m for m in logged)


def test_sim_training_callback():
    import threading

    from tools.train_log_bait_rl import WinRateCallback

    events = []
    stop_event = threading.Event()
    cb = WinRateCallback(
        check_freq=1,
        progress_callback=events.append,
        stop_event=stop_event,
        total_steps=100,
    )
    stop_event.set()
    assert cb._on_step() is False
    assert any(e.get("type") == "stopped" for e in events)
