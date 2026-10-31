"""Tests for the stream_analyzer pre-screen."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.exceptions import ServiceValidationError

from custom_components.llmvision import (
    ServiceCallData,
    _parse_prescreen_response,
    _run_prescreen,
    setup,
)
from custom_components.llmvision.const import DEFAULT_PRESCREEN_PROMPT
from custom_components.llmvision.media_handlers import MediaProcessor

from tests.test_init import _base_service_data, _build_data_call, _make_hass


def _call(**overrides) -> ServiceCallData:
    return ServiceCallData(_build_data_call(_base_service_data(**overrides)))


def _processor_with_frames(frames=None):
    processor = Mock()
    processor.prescreen_frames = (
        [("camera0-frame-02", "b64-a"), ("camera1-frame-05", "b64-b")]
        if frames is None
        else frames
    )
    return processor


def _verdict(worth, confidence="high", subjects=None, reason="r"):
    return {
        "structured_response": {
            "worth_analyzing": worth,
            "subjects": subjects or [],
            "confidence": confidence,
            "reason": reason,
        }
    }


class TestServiceCallDataPrescreen:
    def test_defaults_leave_prescreen_off_and_in_shadow_mode(self):
        call = _call()
        assert call.prescreen_model is None
        assert call.prescreen_provider is None
        assert call.prescreen_frames == 4
        assert call.prescreen_target_width == 1024
        assert call.prescreen_media_resolution == "medium"
        assert call.prescreen_message == DEFAULT_PRESCREEN_PROMPT
        assert call.prescreen_max_tokens == 1024
        assert call.prescreen_fail_open is True
        assert call.prescreen_enforce is False

    def test_reads_overrides(self):
        call = _call(
            prescreen_model="lite",
            prescreen_provider="other",
            prescreen_frames=2,
            prescreen_target_width=384,
            prescreen_media_resolution="low",
            prescreen_message="anything?",
            prescreen_max_tokens=300,
            prescreen_fail_open=False,
            prescreen_enforce=True,
        )
        assert call.prescreen_model == "lite"
        assert call.prescreen_provider == "other"
        assert call.prescreen_frames == 2
        assert call.prescreen_target_width == 384
        assert call.prescreen_media_resolution == "low"
        assert call.prescreen_message == "anything?"
        assert call.prescreen_max_tokens == 300
        assert call.prescreen_fail_open is False
        assert call.prescreen_enforce is True


class TestParsePrescreenResponse:
    def test_structured(self):
        parsed = _parse_prescreen_response(
            _verdict(True, "HIGH", ["person"], "walking up")
        )
        assert parsed == {
            "worth_analyzing": True,
            "subjects": ["person"],
            "confidence": "high",
            "reason": "walking up",
        }

    def test_text_with_fences(self):
        body = json.dumps(
            {"worth_analyzing": False, "subjects": [], "confidence": "medium", "reason": "wind"}
        )
        parsed = _parse_prescreen_response({"response_text": f"```json\n{body}\n```"})
        assert parsed["worth_analyzing"] is False
        assert parsed["confidence"] == "medium"

    @pytest.mark.parametrize(
        "response",
        [
            {"response_text": "not json"},
            {"structured_response": {"worth_analyzing": "yes", "confidence": "high"}},
            {"structured_response": {"worth_analyzing": True, "confidence": "sure"}},
            {"response_text": "[1, 2]"},
        ],
    )
    def test_invalid_raises(self, response):
        with pytest.raises(ValueError):
            _parse_prescreen_response(response)


class TestRunPrescreen:
    async def _run(self, call, response=None, side_effect=None, frames=None):
        request = Mock()
        request.call = AsyncMock(return_value=response, side_effect=side_effect)
        with patch("custom_components.llmvision.Request", return_value=request):
            verdict = await _run_prescreen(Mock(), call, _processor_with_frames(frames))
        return verdict, request

    @pytest.mark.asyncio
    async def test_builds_isolated_cheap_call(self):
        call = _call(
            prescreen_model="lite",
            response_format="json",
            structure='{"type": "object"}',
            title_field="title",
            use_memory=True,
            llm_pick_keyframe=True,
            expose_images=True,
            generate_title=True,
            store_in_timeline=True,
            media_resolution="high",
        )
        verdict, request = await self._run(call, _verdict(True, subjects=["person"]))

        assert [c.kwargs for c in request.add_frame.call_args_list] == [
            {"base64_image": "b64-a", "filename": "camera0-frame-02"},
            {"base64_image": "b64-b", "filename": "camera1-frame-05"},
        ]
        screen_call = request.call.await_args.args[0]
        assert screen_call is not call
        assert screen_call.model == "lite"
        assert screen_call.provider == "entry_id"
        assert screen_call.max_tokens == 1024
        assert screen_call.response_format == "json"
        assert json.loads(screen_call.structure)["required"] == [
            "worth_analyzing",
            "subjects",
            "confidence",
            "reason",
        ]
        assert screen_call.message.startswith(DEFAULT_PRESCREEN_PROMPT)
        assert screen_call.use_memory is False
        assert screen_call.llm_pick_keyframe is False
        assert screen_call.expose_images is False
        assert screen_call.generate_title is False
        assert screen_call.store_in_timeline is False
        assert screen_call.title_field == ""
        assert screen_call.media_resolution == "medium"

        # The main call is untouched
        assert call.model == "test-model"
        assert call.message == "message"
        assert call.structure == '{"type": "object"}'
        assert call.use_memory is True
        assert call.media_resolution == "high"

        assert verdict["passed"] is True
        assert verdict["subjects"] == ["person"]
        assert verdict["frames"] == ["camera0-frame-02", "camera1-frame-05"]
        assert verdict["enforced"] is False

    @pytest.mark.asyncio
    async def test_uses_prescreen_provider_when_set(self):
        call = _call(prescreen_model="lite", prescreen_provider="cheap_entry")
        _, request = await self._run(call, _verdict(True))
        assert request.call.await_args.args[0].provider == "cheap_entry"
        assert call.provider == "entry_id"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("worth", "confidence", "passed"),
        [
            (True, "high", True),
            (True, "low", True),
            (False, "high", False),
            (False, "medium", False),
            (False, "low", True),  # ties go to analysis
        ],
    )
    async def test_pass_rule(self, worth, confidence, passed):
        call = _call(prescreen_model="lite")
        verdict, _ = await self._run(call, _verdict(worth, confidence))
        assert verdict["passed"] is passed
        assert "error" not in verdict

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"side_effect": ServiceValidationError("quota")},
            {"side_effect": asyncio.TimeoutError()},
            {"response": {"response_text": "garbage"}},
            {"response": _verdict(False), "frames": []},
        ],
    )
    async def test_fails_open(self, kwargs):
        call = _call(prescreen_model="lite")
        verdict, _ = await self._run(call, **kwargs)
        assert verdict["passed"] is True
        assert verdict["error"]

    @pytest.mark.asyncio
    async def test_fail_closed_raises(self):
        call = _call(prescreen_model="lite", prescreen_fail_open=False)
        with pytest.raises(ServiceValidationError, match="Pre-screen failed: quota"):
            await self._run(call, side_effect=ServiceValidationError("quota"))


class TestStreamAnalyzerPrescreen:
    async def _stream(self, call, prescreen_verdict):
        hass = _make_hass()
        setup(hass, {})
        handlers = {c.args[1]: c.args[2] for c in hass.services.register.call_args_list}

        request_obj = Mock()
        request_obj.call = AsyncMock(return_value={"response_text": "ok"})
        processor = Mock()
        processor.key_frame = ""
        processor.add_streams = AsyncMock(return_value=request_obj)
        processor.select_and_expose_keyframe = AsyncMock()
        memory = Mock()
        memory._update_memory = AsyncMock()
        run_prescreen = AsyncMock(return_value=prescreen_verdict)
        create_event = AsyncMock()

        with (
            patch("custom_components.llmvision.ServiceCallData", return_value=call),
            patch("custom_components.llmvision.Request", return_value=request_obj),
            patch("custom_components.llmvision.MediaProcessor", return_value=processor),
            patch("custom_components.llmvision.Memory", return_value=memory),
            patch("custom_components.llmvision._run_prescreen", new=run_prescreen),
            patch("custom_components.llmvision._create_event", new=create_event),
        ):
            result = await handlers["stream_analyzer"](Mock())
        return SimpleNamespace(
            result=result,
            request=request_obj,
            processor=processor,
            run_prescreen=run_prescreen,
            create_event=create_event,
        )

    @pytest.mark.asyncio
    async def test_off_by_default(self):
        out = await self._stream(_call(expose_images=True), {"passed": False})
        assert out.processor.add_streams.await_args.kwargs["prescreen_frames"] == 0
        out.run_prescreen.assert_not_awaited()
        out.request.call.assert_awaited_once()
        assert "prescreen" not in out.result
        assert "skipped" not in out.result

    @pytest.mark.asyncio
    async def test_enforced_rejection_skips_everything(self):
        verdict = {"ran": True, "passed": False, "reason": "wind"}
        out = await self._stream(
            _call(prescreen_model="lite", prescreen_enforce=True, expose_images=True),
            verdict,
        )
        assert out.processor.add_streams.await_args.kwargs["prescreen_frames"] == 4
        assert out.result == {"prescreen": verdict, "skipped": True}
        out.request.call.assert_not_awaited()
        out.processor.select_and_expose_keyframe.assert_not_awaited()
        out.create_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shadow_mode_rejection_still_runs_main_call(self):
        verdict = {"ran": True, "passed": False, "enforced": False}
        out = await self._stream(_call(prescreen_model="lite"), verdict)
        out.request.call.assert_awaited_once()
        out.create_event.assert_awaited_once()
        assert out.result["prescreen"] == verdict
        assert out.result["skipped"] is False
        assert out.result["response_text"] == "ok"

    @pytest.mark.asyncio
    async def test_pass_runs_main_call(self):
        verdict = {"ran": True, "passed": True}
        out = await self._stream(
            _call(prescreen_model="lite", prescreen_enforce=True), verdict
        )
        out.request.call.assert_awaited_once()
        assert out.result["prescreen"] == verdict
        assert out.result["skipped"] is False


class TestStashPrescreenFrames:
    @pytest.fixture
    def processor(self):
        processor = MediaProcessor.__new__(MediaProcessor)
        processor.prescreen_frames = []
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data: f"{image_data}@{target_width}"
        )
        return processor

    @pytest.mark.asyncio
    async def test_keeps_most_changed_in_capture_order(self, processor):
        ranked = [  # already sorted most-changed first
            ("camera0-frame-04", "d4", 0.1),
            ("camera1-frame-02", "e2", 0.2),
            ("camera0-frame-01", "d1", 0.3),
            ("camera0-frame-03", "d3", 0.9),
        ]
        first = {"camera.a": ("camera0-frame-00", "d0")}
        await processor._stash_prescreen_frames(ranked, first, ["camera.a"], 3, 512)
        assert processor.prescreen_frames == [
            ("camera0-frame-01", "d1@512"),
            ("camera0-frame-04", "d4@512"),
            ("camera1-frame-02", "e2@512"),
        ]

    @pytest.mark.asyncio
    async def test_falls_back_to_first_frames(self, processor):
        first = {
            "camera.a": ("camera0-frame-00", "a0"),
            "camera.b": ("camera1-frame-00", "b0"),
        }
        await processor._stash_prescreen_frames(
            [], first, ["camera.a", "camera.b"], 4, 256
        )
        assert processor.prescreen_frames == [
            ("camera0-frame-00", "a0@256"),
            ("camera1-frame-00", "b0@256"),
        ]


class TestRecordWiring:
    @pytest.mark.asyncio
    async def test_record_stashes_prescreen_frames(self):
        from tests.test_media_handlers import _make_jpeg_bytes

        hass = Mock()
        hass.loop = Mock()
        hass.loop.run_in_executor = AsyncMock(
            side_effect=lambda _executor, func, *args: func(*args)
        )
        hass.states = Mock()
        hass.states.get.return_value = SimpleNamespace(
            attributes={"entity_picture": "/api/camera_proxy/camera.front"}
        )
        with patch(
            "custom_components.llmvision.media_handlers.async_get_clientsession"
        ):
            processor = MediaProcessor(hass, Mock())

        clock = {"now": 0.0}

        async def fake_sleep(delay):
            clock["now"] += delay

        frame_iter = iter(
            [_make_jpeg_bytes(c) for c in ("black", "gray", "white", "red")]
        )

        async def fake_fetch(*args, **kwargs):
            frame = next(frame_iter, None)
            if frame is None:
                clock["now"] = 100.0
            return frame

        processor._fetch = AsyncMock(side_effect=fake_fetch)
        # frame-01 changed most, then frame-03, then frame-02
        processor._similarity_score = Mock(side_effect=[0.1, 0.8, 0.5])
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data: f"w{target_width}"
        )

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), patch(
            "custom_components.llmvision.media_handlers.time.time",
            side_effect=lambda: clock["now"],
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            await processor.add_streams(
                image_entities=["camera.front"],
                duration=20,
                max_frames=4,
                target_width=1024,
                include_filename=False,
                expose_images=False,
                prescreen_frames=2,
                prescreen_target_width=512,
            )

        assert processor.prescreen_frames == [
            ("camera0-frame-01", "w512"),
            ("camera0-frame-03", "w512"),
        ]
        # The main call still gets all four frames at full width
        assert len(processor.client.add_frame.call_args_list) == 4
        assert {
            c.kwargs["base64_image"] for c in processor.client.add_frame.call_args_list
        } == {"w1024"}
