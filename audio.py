import argparse
import asyncio
from typing import Awaitable, Callable

import numpy as np
from google import genai
from google.genai import types
from pydub import AudioSegment


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
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )

        # Declare tools to Gemini (minimal declaration: name only)
        func_decls = [
            types.FunctionDeclaration(
                name=name, description="Ask the computer agent to perform an action"
            )
            for name in tools.keys()
        ]

        self._config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            tools=[types.ToolsConfig(function_declarations=func_decls)],
        )

        self._session_cm = None
        self._session = None
        self._receiver_task: asyncio.Task | None = None

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self._session is not None:
            return

        self._session_cm = self._client.aio.live.connect(
            model=self._model,
            config=self._config,
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
        """
        Accept a 48kHz, 2-channel waveform (numpy array) and send it to Gemini.

        chunk: shape (n_samples, 2) or equivalent, dtype float32 in [-1, 1] or int16.
        """
        if self._session is None:
            raise RuntimeError(
                "Agent not started. Use 'await start()' or 'async with'."
            )

        pcm_bytes = self._prepare_audio(chunk)
        await self._session.send(input={"data": pcm_bytes, "mime_type": "audio/pcm"})

    # ---------- internals ----------

    def _prepare_audio(self, chunk: np.ndarray) -> bytes:
        """
        Convert 48kHz stereo -> 16kHz mono int16 PCM bytes.
        """
        arr = np.asarray(chunk)

        # Ensure float32 for mixing
        if arr.ndim == 2:
            # Try (n, 2) then (2, n)
            if arr.shape[1] == 2:
                left = arr[:, 0]
                right = arr[:, 1]
            elif arr.shape[0] == 2:
                left = arr[0]
                right = arr[1]
            else:
                # Fallback: take first channel
                left = arr[..., 0]
                right = left
            mono = (left.astype(np.float32) + right.astype(np.float32)) / 2.0
        else:
            mono = arr.astype(np.float32)

        # If original was int-based, normalize to [-1, 1]
        if chunk.dtype == np.int16:
            mono = mono / 32768.0

        # Naive downsample 48k -> 16k by factor 3 (minimal, not hi-fi)
        mono_16k = mono[::3]

        # Clip to [-1, 1] and convert to int16 PCM
        mono_16k = np.clip(mono_16k, -1.0, 1.0)
        pcm_int16 = (mono_16k * 32767.0).astype(np.int16)

        return pcm_int16.tobytes()

    async def _receive_loop(self) -> None:
        """
        Continuously read from the live session:
        - Forward audio to play_audio_cb as int16 numpy array (24kHz mono from Gemini).
        - Handle tool calls by invoking Python async functions and sending responses back.
        """
        assert self._session is not None

        try:
            async for response in self._session.receive():
                # Audio data (24kHz mono int16 PCM)
                if getattr(response, "data", None):
                    pcm = np.frombuffer(response.data, dtype=np.int16)
                    await self._play_audio_cb(pcm)

                # Optional textual output (if you care)
                if getattr(response, "text", None):
                    # You could log or handle this if needed
                    pass

                # Tool calls
                if getattr(response, "tool_call", None):
                    await self._handle_tool_call(response.tool_call)

        except asyncio.CancelledError:
            # Normal shutdown
            pass

    async def _handle_tool_call(self, tool_call) -> None:
        """
        Map Gemini function calls to local async text->text tools and
        send FunctionResponse back.
        """
        calls = getattr(tool_call, "function_calls", None) or []
        if not calls:
            return

        responses: list[types.FunctionResponse] = []

        for fc in calls:
            name = getattr(fc, "name", None)
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

            responses.append(
                types.FunctionResponse(
                    id=fc.id,
                    name=name,
                    response={"result": result},
                )
            )

        if responses:
            await self._session.send_tool_response(function_responses=responses)


# Import or define GeminiRealtimeAudioAgent above this point
# from your_module import GeminiRealtimeAudioAgent


# ---- tool + audio callbacks ----


async def use_computer_tool(text: str) -> str:
    print(f"[tool] got text: {text!r}")
    return "done"


async def play_audio_callback(pcm: np.ndarray) -> None:
    """
    pcm: int16 mono at 24kHz from Gemini.
    Just print the first 5 samples.
    """
    # Ensure it's a 1D array of int16
    arr = np.asarray(pcm, dtype=np.int16).ravel()
    print("[audio] first 5 samples:", arr[:5])


# ---- MP3 -> 48kHz stereo chunked streaming ----


def load_mp3_as_48k_stereo(path: str) -> np.ndarray:
    """
    Load an MP3 file, convert to 48kHz, stereo, 16-bit samples,
    and return as numpy array of shape (num_frames, 2), dtype int16.
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(48000).set_channels(2).set_sample_width(2)

    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

    # pydub returns interleaved [L0, R0, L1, R1, ...]
    samples = samples.reshape(-1, 2)
    return samples


async def stream_mp3_into_agent(
    agent: GeminiRealtimeAudioAgent,
    mp3_path: str,
    chunk_frames: int = 4800,  # 0.1s @ 48kHz
) -> None:
    """
    Reads the MP3, converts to 48kHz stereo, and feeds it into the agent in chunks.
    """
    stereo_48k = load_mp3_as_48k_stereo(mp3_path)
    num_frames = stereo_48k.shape[0]

    for start in range(0, num_frames, chunk_frames):
        chunk = stereo_48k[start : start + chunk_frames]
        if chunk.size == 0:
            break
        await agent.handle_audio_chunk(chunk)


# ---- main ----


async def main(mp3_path: str) -> None:
    tools: dict[str, Callable[[str], Awaitable[str]]] = {
        "ask_computer_agent": use_computer_tool,
    }

    # GEMINI_API_KEY is taken from the environment by default in the class;
    # pass api_key=... explicitly here if you prefer.
    agent = GeminiRealtimeAudioAgent(
        tools=tools,
        play_audio_cb=play_audio_callback,
    )

    async with agent:
        await stream_mp3_into_agent(agent, mp3_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3_path", help="Path to an input .mp3 file")
    args = parser.parse_args()

    asyncio.run(main(args.mp3_path))
