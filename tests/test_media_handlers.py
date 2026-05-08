"""Unit tests for media_handlers.py module."""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from PIL import Image
import io
import base64
from custom_components.llmvision.media_handlers import MediaProcessor


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


def _make_jpeg_bytes(brightness):
    """Build a tiny JPEG with a given uniform brightness (0-255)."""
    img = Image.new("RGB", (16, 16), color=(brightness, brightness, brightness))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


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
        return _make_jpeg_bytes(20 + n * 40)

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
