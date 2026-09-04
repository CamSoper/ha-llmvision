"""Unit tests for media_handlers.py module."""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from PIL import Image
import io
import base64
from types import SimpleNamespace
from homeassistant.exceptions import ServiceValidationError
from custom_components.llmvision.media_handlers import MediaProcessor


def _make_jpeg_bytes(color):
    """Create JPEG bytes for a solid-color test image."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


class _AsyncResponseContext:
    """Minimal async context manager for aiohttp-style responses."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _StreamingContent:
    """Streaming response content stub."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncWriter:
    """Async file writer compatible with aiofile.async_open usage."""

    def __init__(self, path):
        self._path = path
        self._file = None

    async def __aenter__(self):
        self._file = open(self._path, "wb")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
        return False

    async def write(self, data):
        if self._file is None:
            raise RuntimeError("Writer not opened")
        self._file.write(data)


class TestMediaProcessor:
    """Test MediaProcessor class."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = Mock()
        hass.loop = Mock()
        hass.loop.run_in_executor = AsyncMock()
        hass.states = Mock()
        hass.states.get = Mock()
        hass.config = Mock()
        hass.config.path = Mock(return_value="/mock/path")
        return hass

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        client = Mock()
        client.add_frame = Mock()
        return client

    @pytest.fixture
    def processor(self, mock_hass, mock_client):
        """Create a MediaProcessor instance."""
        with patch('custom_components.llmvision.media_handlers.async_get_clientsession'):
            return MediaProcessor(mock_hass, mock_client)

    def test_init(self, processor, mock_hass, mock_client):
        """Test MediaProcessor initialization."""
        assert processor.hass == mock_hass
        assert processor.client == mock_client
        assert processor.base64_images == []
        assert processor.filenames == []
        assert processor.key_frame == ""
        assert processor.candidate_frames == []

    @pytest.mark.asyncio
    async def test_encode_image(self, processor):
        """Test _encode_image method."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')

        result = await processor._encode_image(img)

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's valid base64
        base64.b64decode(result)

    @pytest.mark.asyncio
    async def test_save_clip_writes_base64_image(self, processor, tmp_path):
        """_save_clip should decode base64 image payloads before writing."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        output = tmp_path / "snapshot.jpg"

        with patch("custom_components.llmvision.media_handlers.os.makedirs"):
            await processor._save_clip(
                image_data=base64.b64encode(b"image-bytes").decode("utf-8"),
                image_path=str(output),
            )

        assert output.read_bytes() == b"image-bytes"

    @pytest.mark.asyncio
    async def test_expose_image_uses_frame_path_once(self, processor, tmp_path):
        """_expose_image should derive image data from a file only for the first keyframe."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        source = tmp_path / "frame.jpg"
        source.write_bytes(_make_jpeg_bytes("navy"))
        processor._save_clip = AsyncMock()

        with patch("custom_components.llmvision.media_handlers.os.makedirs"):
            await processor._expose_image(
                frame_name="7",
                image_data=None,
                uid="deadbeef",
                frame_path=str(source),
            )
            await processor._expose_image(
                frame_name="8",
                image_data="ignored",
                uid="second",
            )

        assert processor.key_frame.endswith("deadbeef-7.jpg")
        processor._save_clip.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_select_keyframe_index_picks_lowest_similarity(self, processor):
        """_select_keyframe_index should return the most dissimilar candidate."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        processor._similarity_score = Mock(side_effect=[0.8, 0.2, 0.5])

        result = await processor._select_keyframe_index(
            _make_jpeg_bytes("black"),
            [
                _make_jpeg_bytes("gray"),
                _make_jpeg_bytes("white"),
                _make_jpeg_bytes("red"),
            ],
        )

        assert result == 1

    def test_convert_to_rgb_rgba(self, processor):
        """Test _convert_to_rgb with RGBA image."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))

        result = processor._convert_to_rgb(img)

        assert result.mode == 'RGB'

    def test_convert_to_rgb_already_rgb(self, processor):
        """Test _convert_to_rgb with RGB image."""
        img = Image.new('RGB', (100, 100), color='red')

        result = processor._convert_to_rgb(img)

        assert result.mode == 'RGB'

    @pytest.mark.asyncio
    async def test_resize_image_with_img(self, processor):
        """Test resize_image with PIL Image object."""
        # Create a test image
        img = Image.new('RGB', (200, 100), color='blue')

        result = await processor.resize_image(
            target_width=100,
            img=img
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resize_image_with_image_path(self, processor, tmp_path):
        """resize_image should support file-based inputs."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        image_path = tmp_path / "photo.jpg"
        image_path.write_bytes(_make_jpeg_bytes("green"))

        result = await processor.resize_image(
            target_width=8, image_path=str(image_path)
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resize_image_requires_input(self, processor):
        """resize_image should reject empty inputs."""
        with pytest.raises(ServiceValidationError, match="No image data provided"):
            await processor.resize_image(target_width=10)

    def test_similarity_score(self, processor):
        """Test _similarity_score method."""
        # Create two similar images
        img1 = Image.new('L', (100, 100), color=128)
        img2 = Image.new('L', (100, 100), color=130)

        score = processor._similarity_score(img1, img2)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_expose_keyframe_by_index(self, processor):
        """Test expose_keyframe_by_index calls _expose_image with correct data."""
        processor.candidate_frames = [
            ("camera0-frame-1", "base64data1", "camera0"),
            ("camera0-frame-2", "base64data2", "camera0"),
            ("camera0-frame-3", "base64data3", "camera0"),
        ]
        processor._expose_image = AsyncMock()

        await processor.expose_keyframe_by_index(1)

        processor._expose_image.assert_called_once()
        call_kwargs = processor._expose_image.call_args
        assert call_kwargs[1]["frame_name"] == "camera0"
        assert call_kwargs[1]["image_data"] == "base64data2"
        assert "uid" in call_kwargs[1]

    @pytest.mark.asyncio
    async def test_select_and_expose_keyframe(self, processor):
        """Test select_and_expose_keyframe runs SSIM and exposes winner."""
        # Create small test images as base64
        img1 = Image.new('RGB', (10, 10), color='red')
        img2 = Image.new('RGB', (10, 10), color='blue')
        img3 = Image.new('RGB', (10, 10), color='green')
        buffers = []
        for img in [img1, img2, img3]:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buffers.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        processor.candidate_frames = [
            ("cam-frame-1", buffers[0], "cam"),
            ("cam-frame-2", buffers[1], "cam"),
            ("cam-frame-3", buffers[2], "cam"),
        ]
        processor._expose_image = AsyncMock()

        await processor.select_and_expose_keyframe()

        # Should have called _expose_image exactly once
        processor._expose_image.assert_called_once()
        call_kwargs = processor._expose_image.call_args[1]
        assert call_kwargs["frame_name"] == "cam"

    @pytest.mark.asyncio
    async def test_select_and_expose_keyframe_empty_candidates(self, processor):
        """Test select_and_expose_keyframe is no-op when candidates empty."""
        processor.candidate_frames = []
        processor._expose_image = AsyncMock()

        await processor.select_and_expose_keyframe()

        processor._expose_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_retries_until_success(self, processor):
        """_fetch should retry failed responses before succeeding."""
        first = SimpleNamespace(ok=False, status=500)
        second = SimpleNamespace(
            ok=True, status=200, read=AsyncMock(return_value=b"ok")
        )
        processor.session.get.side_effect = [
            _AsyncResponseContext(first),
            _AsyncResponseContext(second),
        ]

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            AsyncMock(),
        ):
            result = await processor._fetch("http://camera/frame.jpg", retry_delay=0)

        assert result == b"ok"

    @pytest.mark.asyncio
    async def test_fetch_streams_to_target_file(self, processor, tmp_path):
        """_fetch should stream large responses to disk when target_file is provided."""
        target = tmp_path / "clip.mp4"
        response = SimpleNamespace(
            ok=True,
            status=200,
            content=_StreamingContent([b"abc", b"def"]),
        )
        processor.session.get.return_value = _AsyncResponseContext(response)

        with patch(
            "custom_components.llmvision.media_handlers.async_open",
            side_effect=lambda path, mode: _FakeAsyncWriter(path),
        ):
            result = await processor._fetch(
                "http://camera/clip.mp4",
                target_file=str(target),
            )

        assert result is None
        assert target.read_bytes() == b"abcdef"

    @pytest.mark.asyncio
    async def test_fetch_returns_none_after_exhausted_errors(self, processor):
        """_fetch should return None after repeated exceptions."""
        processor.session.get.side_effect = RuntimeError("boom")

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            AsyncMock(),
        ):
            result = await processor._fetch(
                "http://camera/frame.jpg",
                max_retries=2,
                retry_delay=0,
                entity_name="front",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_record_returns_frames_in_capture_order(self, processor):
        """Selected stream frames should be emitted in capture order."""
        clock = {"now": 0.0}

        async def fake_sleep(delay):
            clock["now"] += delay

        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        processor.hass.states.get.return_value = SimpleNamespace(
            attributes={"entity_picture": "/api/camera_proxy/camera.front"}
        )
        frame_iter = iter(
            [
                _make_jpeg_bytes("black"),
                _make_jpeg_bytes("gray"),
                _make_jpeg_bytes("white"),
            ]
        )

        async def fake_fetch(*args, **kwargs):
            frame = next(frame_iter, None)
            if frame is None:
                clock["now"] = 10.0
            return frame

        processor._fetch = AsyncMock(side_effect=fake_fetch)
        processor._similarity_score = Mock(side_effect=[0.9, 0.1])
        processor.resize_image = AsyncMock(
            side_effect=["encoded-0", "encoded-1", "encoded-2"]
        )
        processor._select_keyframe_index = AsyncMock(return_value=0)

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
            await processor.record(
                image_entities=["camera.front"],
                duration=2.5,
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        # Fork uses zero-padded frame counters so labels sort lexically.
        assert [
            call.kwargs["filename"]
            for call in processor.client.add_frame.call_args_list
        ] == ["camera0-frame-00", "camera0-frame-01", "camera0-frame-02"]
        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["encoded-0", "encoded-1", "encoded-2"]

    @pytest.mark.asyncio
    async def test_record_raises_when_no_cameras_available(self, processor):
        """record should fail when all cameras are unavailable."""
        clock = {"now": 0.0}

        async def fake_sleep(delay):
            clock["now"] += delay

        processor.hass.states.get.return_value = None

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), patch(
            "custom_components.llmvision.media_handlers.time.time",
            side_effect=lambda: clock["now"],
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            side_effect=fake_sleep,
        ), pytest.raises(
            ServiceValidationError, match="No cameras available"
        ):
            await processor.record(
                image_entities=["camera.front"],
                duration=0.1,
                max_frames=1,
                target_width=64,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_supports_entities_and_paths(self, processor, tmp_path):
        """add_images should add frames from both camera entities and file paths."""
        image_path = tmp_path / "still.jpg"
        image_path.write_bytes(_make_jpeg_bytes("teal"))
        entity_state = SimpleNamespace(
            attributes={
                "entity_picture": "/api/camera_proxy/camera.front",
                "friendly_name": "Front Door",
            }
        )
        processor.hass.states.get.return_value = entity_state
        processor._fetch = AsyncMock(return_value=b"entity-bytes")
        processor.resize_image = AsyncMock(side_effect=["entity-frame", "path-frame"])
        processor._expose_image = AsyncMock()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            result = await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[str(image_path)],
                target_width=128,
                include_filename=True,
                expose_images=True,
            )

        assert result is processor.client
        assert [
            call.kwargs["filename"]
            for call in processor.client.add_frame.call_args_list
        ] == ["Front Door", "still"]
        assert processor._expose_image.await_count == 2

    @pytest.mark.asyncio
    async def test_add_images_raises_for_missing_file(self, processor):
        """add_images should reject missing local image files."""
        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="does not exist"):
            await processor.add_images(
                image_entities=[],
                image_paths=["/missing/file.jpg"],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_raises_when_all_cameras_unavailable(self, processor):
        """add_images should fail when no camera image can be obtained."""
        processor.hass.states.get.return_value = None

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="No cameras available"):
            await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_wraps_attribute_errors(self, processor):
        """add_images should surface malformed entity state access as validation errors."""
        processor.hass.states.get.return_value = object()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="Error accessing camera entity"):
            await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_returns_selected_frames_in_capture_order(self, processor):
        """Selected video keyframes should be added in capture order."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        frame_bytes = [
            _make_jpeg_bytes("black"),
            _make_jpeg_bytes("gray"),
            _make_jpeg_bytes("white"),
        ]
        encoded_by_frame = {
            frame_bytes[0]: "encoded-0",
            frame_bytes[1]: "encoded-1",
            frame_bytes[2]: "encoded-2",
        }

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self, payload):
                self.pid = 1234
                self.stdout = FakeStdout([payload, b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        processor._similarity_score = Mock(side_effect=[0.9, 0.1])
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data=None, **kwargs: encoded_by_frame[
                image_data or b""
            ]
        )

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess(b"".join(frame_bytes))),
        ):
            await processor.add_video(
                video_path="/tmp/test_clip.mp4",
                base_url="http://ha.local",
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["encoded-0", "encoded-1", "encoded-2"]

    @pytest.mark.asyncio
    async def test_add_video_signs_api_path_and_exposes_keyframe(self, processor):
        """add_video should sign local API paths and expose the selected keyframe."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        frame_bytes = [_make_jpeg_bytes("black")]
        temp_video = "/tmp/signed-video.mp4"

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 55
                self.stdout = FakeStdout([b"".join(frame_bytes), b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        class FakeTempFile:
            name = temp_video

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        processor.resize_image = AsyncMock(return_value="encoded-frame")
        processor._select_keyframe_index = AsyncMock(return_value=0)
        processor._expose_image = AsyncMock()
        processor._fetch = AsyncMock(return_value=None)

        with patch(
            "custom_components.llmvision.media_handlers.async_sign_path",
            return_value="/api/signed/path?authSig=1",
        ), patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            return_value=FakeTempFile(),
        ), patch(
            "custom_components.llmvision.media_handlers.os.path.exists",
            side_effect=lambda path: path == temp_video,
        ), patch(
            "custom_components.llmvision.media_handlers.os.path.getsize",
            return_value=10,
        ), patch(
            "custom_components.llmvision.media_handlers.os.remove",
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ):
            await processor.add_video(
                video_path="/api/frigate/notifications/test/clip.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=True,
                expose_images=True,
            )

        assert (
            processor.client.add_frame.call_args.kwargs["filename"]
            == "path?authSig=1 (frame 1)"
        )
        # Fork defers keyframe exposure: frames are stashed as candidates for
        # llm_pick_keyframe rather than exposed immediately during add_video.
        assert len(processor.candidate_frames) == 1

    @pytest.mark.asyncio
    async def test_add_video_raises_when_http_download_is_empty(
        self, processor, tmp_path
    ):
        """add_video should fail when the downloaded remote clip is empty."""
        temp_video = tmp_path / "download.mp4"

        class FakeTempFile:
            def __init__(self, path):
                self.name = str(path)

            def __enter__(self):
                temp_video.write_bytes(b"")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        processor._fetch = AsyncMock(return_value=None)

        with patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            return_value=FakeTempFile(temp_video),
        ), pytest.raises(ServiceValidationError, match="Failed to fetch video"):
            await processor.add_video(
                video_path="http://example.com/clip.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_raises_when_no_frames_are_extracted(self, processor):
        """add_video should fail cleanly if ffmpeg yields no JPEG frames."""

        class FakeStdout:
            async def read(self, _size):
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 44
                self.stdout = FakeStdout()
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ), pytest.raises(ServiceValidationError, match="No frames extracted"):
            await processor.add_video(
                video_path="/tmp/empty.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_raises_when_stdout_is_missing(self, processor):
        """add_video should reject ffmpeg processes without a stdout pipe."""

        class FakeProcess:
            def __init__(self):
                self.pid = 77
                self.stdout = None
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ), pytest.raises(ServiceValidationError, match="stdout not available"):
            await processor.add_video(
                video_path="/tmp/no-stdout.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_videos_processes_event_ids(self, processor):
        """add_videos should convert event ids into Frigate clip URLs."""
        processor.add_video = AsyncMock()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            result = await processor.add_videos(
                video_paths=["/tmp/local.mp4"],
                event_ids=["abc123"],
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert result is processor.client
        assert processor.add_video.await_count == 2
        assert any(
            call.kwargs["video_path"] == "/api/frigate/notifications/abc123/clip.mp4"
            for call in processor.add_video.await_args_list
        )

    @pytest.mark.asyncio
    async def test_add_streams_delegates_to_record(self, processor):
        """add_streams should proxy to record when image entities are present."""
        processor.record = AsyncMock()

        result = await processor.add_streams(
            image_entities=["camera.front"],
            duration=5,
            max_frames=2,
            target_width=128,
            include_filename=False,
            expose_images=False,
        )

        assert result is processor.client
        processor.record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_visual_data_delegates_to_add_images(self, processor):
        """add_visual_data should proxy to add_images."""
        processor.add_images = AsyncMock()

        result = await processor.add_visual_data(
            image_entities=["camera.front"],
            image_paths=["/tmp/still.jpg"],
            target_width=128,
            include_filename=False,
            expose_images=False,
        )

        assert result is processor.client
        processor.add_images.assert_awaited_once()


async def _drive_record_two_cameras(
    processor,
    mock_hass,
    mock_client,
    image_entities,
    *,
    frames_per_camera=4,
    ssim_scores_per_camera=None,
    include_filename=True,
    max_frames=20,
):
    """Drive MediaProcessor.record() against two cameras with mocked I/O.

    Returns the ordered list of filenames passed to client.add_frame.
    """
    # Each camera serves a sequence of distinct JPEGs, then None to end.
    state_counters = {entity: 0 for entity in image_entities}

    def fake_get_state(entity_id):
        if entity_id not in image_entities:
            return None
        state = Mock()
        state.attributes = {"entity_picture": f"/api/camera/{entity_id}.jpg"}
        return state

    mock_hass.states.get = Mock(side_effect=fake_get_state)

    async def fake_executor(executor, fn, *args):
        return fn(*args)

    mock_hass.loop.run_in_executor = fake_executor

    async def fake_fetch(url, entity_name=None, **kwargs):
        n = state_counters[entity_name]
        state_counters[entity_name] = n + 1
        if n >= frames_per_camera:
            return None
        # Vary brightness so SSIM differs slightly.
        return _make_jpeg_bytes((20 + n * 40,) * 3)

    processor._fetch = fake_fetch

    if ssim_scores_per_camera is not None:
        # Hand out scripted SSIM scores per camera. Each fetch tags which
        # camera is currently mid-iteration so the next _similarity_score call
        # consumes from that camera's queue.
        score_queues = {
            entity: list(ssim_scores_per_camera[entity])
            for entity in image_entities
        }
        last_entity = {"name": None}
        original_fetch = fake_fetch

        async def tagged_fetch(url, entity_name=None, **kwargs):
            last_entity["name"] = entity_name
            return await original_fetch(url, entity_name=entity_name, **kwargs)

        processor._fetch = tagged_fetch

        def queued_score(prev, curr):
            queue = score_queues.get(last_entity["name"])
            if queue:
                return queue.pop(0)
            return 0.5

        processor._similarity_score = queued_score

    async def fake_resize(target_width=None, image_data=None, **kwargs):
        return base64.b64encode(image_data or b"").decode()

    processor.resize_image = fake_resize

    # Patch get_url + asyncio.sleep + monotonic time so the loop terminates fast.
    time_state = {"now": 0.0}

    def fake_time():
        time_state["now"] += 0.5
        return time_state["now"]

    async def fake_sleep(_):
        return None

    with (
        patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://localhost:8123",
        ),
        patch(
            "custom_components.llmvision.media_handlers.time.time",
            side_effect=fake_time,
        ),
        patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            side_effect=fake_sleep,
        ),
    ):
        await processor.record(
            image_entities=image_entities,
            duration=2,
            max_frames=max_frames,
            target_width=100,
            include_filename=include_filename,
            expose_images=False,
        )

    filenames = []
    for c in mock_client.add_frame.call_args_list:
        # add_frame is called with kwargs (base64_image=..., filename=...)
        filenames.append(c.kwargs.get("filename"))
    return filenames


class TestRecordOrdering:
    """Tests for the per-camera grouping / label format produced by record()."""

    @pytest.fixture
    def mock_hass(self):
        hass = Mock()
        hass.loop = Mock()
        hass.loop.run_in_executor = AsyncMock()
        hass.states = Mock()
        hass.states.get = Mock()
        hass.config = Mock()
        hass.config.path = Mock(return_value="/mock/path")
        return hass

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.add_frame = Mock()
        return client

    @pytest.fixture
    def processor(self, mock_hass, mock_client):
        with patch(
            "custom_components.llmvision.media_handlers.async_get_clientsession"
        ):
            return MediaProcessor(mock_hass, mock_client)

    @pytest.mark.asyncio
    async def test_frames_grouped_by_camera_in_image_entities_order(
        self, processor, mock_hass, mock_client
    ):
        """All camera-A frames precede all camera-B frames."""
        image_entities = ["camera.front_door", "camera.back_yard"]

        filenames = await _drive_record_two_cameras(
            processor,
            mock_hass,
            mock_client,
            image_entities,
            frames_per_camera=4,
            include_filename=True,
        )

        assert filenames, "expected at least one frame to be added"

        front_door_indices = [
            i for i, name in enumerate(filenames) if name.startswith("front_door-")
        ]
        back_yard_indices = [
            i for i, name in enumerate(filenames) if name.startswith("back_yard-")
        ]

        assert front_door_indices, "expected front_door frames in output"
        assert back_yard_indices, "expected back_yard frames in output"
        assert max(front_door_indices) < min(back_yard_indices), (
            f"camera groups not separated: {filenames}"
        )

    @pytest.mark.asyncio
    async def test_within_camera_frames_are_in_capture_order(
        self, processor, mock_hass, mock_client
    ):
        """Even with non-monotonic SSIM, output order within a camera follows
        the frame counter (capture order)."""
        image_entities = ["camera.front_door", "camera.back_yard"]

        # Non-monotonic SSIM scores so SSIM-sort would scramble order.
        # Three scored frames per camera (frame 0 is the unscored first frame).
        ssim_scores = {
            "camera.front_door": [0.9, 0.1, 0.5],
            "camera.back_yard": [0.2, 0.8, 0.4],
        }

        filenames = await _drive_record_two_cameras(
            processor,
            mock_hass,
            mock_client,
            image_entities,
            frames_per_camera=4,
            ssim_scores_per_camera=ssim_scores,
            include_filename=True,
        )

        for camera_prefix in ("front_door", "back_yard"):
            this_camera = [n for n in filenames if n.startswith(f"{camera_prefix}-")]
            assert this_camera, f"no frames for {camera_prefix}"
            # Sorted by frame counter ascending — the labels (zero-padded) are
            # already lexically sortable, so the output should already match.
            assert this_camera == sorted(this_camera), (
                f"{camera_prefix} frames not in capture order: {this_camera}"
            )

    @pytest.mark.asyncio
    async def test_labels_are_zero_padded_and_have_no_timestamp_suffix(
        self, processor, mock_hass, mock_client
    ):
        """Labels are `<name>-frame-NN` with no `[t+X.Xs]` suffix."""
        image_entities = ["camera.front_door", "camera.back_yard"]

        filenames = await _drive_record_two_cameras(
            processor,
            mock_hass,
            mock_client,
            image_entities,
            frames_per_camera=4,
            include_filename=True,
        )

        assert filenames
        for name in filenames:
            assert "[t+" not in name, f"unexpected timestamp suffix in {name!r}"
            # Expected shape: <camera>-frame-NN, two-digit zero-padded counter.
            parts = name.split("-")
            assert len(parts) >= 3, f"unexpected label shape: {name!r}"
            counter = parts[-1]
            assert counter.isdigit(), f"non-numeric counter in {name!r}"
            assert len(counter) >= 2, (
                f"counter not zero-padded in {name!r}: {counter!r}"
            )

    @pytest.mark.asyncio
    async def test_camera_name_extractable_via_split_dash(
        self, processor, mock_hass, mock_client
    ):
        """Downstream code uses fname.split('-')[0] to pull the camera name.
        Regression check that the new label format keeps that contract."""
        image_entities = ["camera.front_door", "camera.back_yard"]

        filenames = await _drive_record_two_cameras(
            processor,
            mock_hass,
            mock_client,
            image_entities,
            frames_per_camera=4,
            include_filename=True,
        )

        seen = {name.split("-")[0] for name in filenames}
        assert seen <= {"front_door", "back_yard"}, (
            f"unexpected camera names: {seen}"
        )
        # Both cameras should appear (each contributes its first frame).
        assert seen == {"front_door", "back_yard"}
