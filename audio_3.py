import argparse
import asyncio
from typing import Awaitable, Callable

import numpy as np
from pydub import AudioSegment

from google import genai
from google.genai import types


class GeminiRealtimeAudioAgent:
    def __init__(
        self,
        tools: dict[str, Callable[[str], Awaitable[str]]],
        play_audio_cb: Callable[[np.ndarray], Awaitable[None]],
        model: str = "models/gemini-2.5-flash-native-audio-preview-09-2025",
        api_key: str | None = None,
    ) -> None:
        self._tools = tools
        self._play_audio_cb = play_audio_cb
        self._model = model

        self._client = genai.Client(
            api_key=api_key, http_options={"api_version": "v1beta"}
        )

        tools_config = [
            {"function_declarations": [{"name": name} for name in tools.keys()]}
        ]

        self._config = {"response_modalities": ["AUDIO"], "tools": tools_config}

        self._session_cm = None
        self._session = None
        self._receiver_task: asyncio.Task | None = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self._session is not None:
            return

        self._session_cm = self._client.aio.live.connect(
            model=self._model,
            config=self._config,  # type: ignore[arg-type]
        )
        self._session = await self._session_cm.__aenter__()

        # Background task to pull audio + tool calls
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def aclose(self) -> None:
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None

        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self._session = None

    async def __aenter__(self) -> "GeminiRealtimeAudioAgent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ---------- public API ----------

    async def handle_audio_chunk(self, chunk: np.ndarray) -> None:
        assert self._session is not None
        pcm_bytes = self._prepare_audio(chunk)

        await self._session.send_realtime_input(
            media=types.Blob(
                data=pcm_bytes,
                mime_type="audio/pcm",  # 16 kHz mono PCM
            )
        )

    async def end_turn(self, text: str = "") -> None:
        assert self._session is not None

        await self._session.send_client_content(turns={"parts": [{"text": text}]})

    # ---------- internals ----------

    def _prepare_audio(self, chunk: np.ndarray) -> bytes:
        arr = np.asarray(chunk)

        # Stereo -> mono
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                left = arr[:, 0]
                right = arr[:, 1]
            elif arr.shape[0] == 2:
                left = arr[0]
                right = arr[1]
            else:
                left = arr[..., 0]
                right = left
            mono = (left.astype(np.float32) + right.astype(np.float32)) / 2.0
        else:
            mono = arr.astype(np.float32)

        # Normalize if int16
        if chunk.dtype == np.int16:
            mono = mono / 32768.0

        # Naive downsample 48k -> 16k (factor 3)
        mono_16k = mono[::3]

        mono_16k = np.clip(mono_16k, -1.0, 1.0)
        pcm_int16 = (mono_16k * 32767.0).astype(np.int16)

        return pcm_int16.tobytes()

    async def _receive_loop(self) -> None:
        assert self._session is not None

        try:
            async for response in self._session.receive():
                # Audio data (24kHz mono int16 PCM)
                if getattr(response, "data", None):
                    pcm = np.frombuffer(response.data, dtype=np.int16)
                    await self._play_audio_cb(pcm)

                # Optional text / thought (just print for now)
                if getattr(response, "text", None):
                    print("[model text]", response.text)
                if getattr(response, "thought", None):
                    # You can inspect this if you want the reasoning trace
                    print("[model thought]", response.thought)
                    pass

                # Tool calls
                if getattr(response, "tool_call", None):
                    await self._handle_tool_call(response.tool_call)

        except asyncio.CancelledError:
            pass

    async def _handle_tool_call(self, tool_call) -> None:
        assert self._session is not None
        calls = getattr(tool_call, "function_calls", None) or []
        if not calls:
            return

        function_responses: list[types.FunctionResponse] = []

        for fc in calls:
            name = getattr(fc, "name", None)
            assert name is not None
            func = self._tools.get(name)
            if func is None:
                continue

            args = getattr(fc, "args", {}) or {}
            if isinstance(args, dict):
                text = (
                    args.get("input")
                    or args.get("text")
                    or args.get("query")
                    or str(args)
                )
            else:
                text = str(args)

            result = await func(text)

            function_responses.append(
                types.FunctionResponse(
                    id=fc.id,
                    name=name,
                    response={"result": result},
                )
            )

        if function_responses:
            await self._session.send_tool_response(
                function_responses=function_responses
            )


async def sample_tool(text: str) -> str:
    print(f"[tool] got text: {text!r}")
    return "done"


async def play_audio_callback(pcm: np.ndarray) -> None:
    arr = np.asarray(pcm, dtype=np.int16).ravel()
    print("[audio] first 5 samples:", arr[:5])


def load_mp3_as_48k_stereo(path: str) -> np.ndarray:
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(48000).set_channels(2).set_sample_width(2)

    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    samples = samples.reshape(-1, 2)  # interleaved [L0, R0, L1, R1, ...]
    return samples


async def stream_mp3_into_agent(
    agent: GeminiRealtimeAudioAgent,
    mp3_path: str,
    chunk_frames: int = 4800,  # ~0.1s @ 48kHz
) -> None:
    stereo_48k = load_mp3_as_48k_stereo(mp3_path)
    num_frames = stereo_48k.shape[0]

    for start in range(0, num_frames, chunk_frames):
        chunk = stereo_48k[start : start + chunk_frames]
        if chunk.size == 0:
            break
        await agent.handle_audio_chunk(chunk)


async def main(mp3_path: str) -> None:
    tools: dict[str, Callable[[str], Awaitable[str]]] = {
        "ask_computer_agent": sample_tool,
    }

    agent = GeminiRealtimeAudioAgent(
        tools=tools,
        play_audio_cb=play_audio_callback,
    )

    async with agent:
        # Send all audio
        await stream_mp3_into_agent(agent, mp3_path)

        # End the turn to trigger a response
        await agent.end_turn("Describe what you heard in this audio.")

        # Keep the session open long enough to receive audio + any text/thought
        await asyncio.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3_path", help="Path to an input .mp3 file")
    args = parser.parse_args()

    asyncio.run(main(args.mp3_path))
