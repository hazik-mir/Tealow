"""Streaming helpers for the TeaLow SDK.

Provides thin wrappers around provider stream iterators that
accumulate the full response text while yielding incremental chunks,
so callers can both iterate over live tokens and inspect the final
aggregated :class:`~TeaLow.models.Response` once streaming completes.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator, List, Optional

from .exceptions import StreamingError
from .models import Response, StreamChunk, Usage


class StreamAccumulator:
    """Accumulates streamed text chunks into a final response.

    Example:
        >>> accumulator = StreamAccumulator(model="gpt-4o-mini", provider="openai")
        >>> for chunk in accumulator.consume(raw_stream):
        ...     print(chunk.delta, end="")
        >>> final = accumulator.finalize()
    """

    def __init__(self, *, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider
        self._pieces: List[str] = []
        self._finish_reason: Optional[str] = None
        self._last_raw: object = None

    def consume(self, chunks: Iterator[StreamChunk]) -> Iterator[StreamChunk]:
        """Iterate over ``chunks``, recording text as it is produced."""
        try:
            for chunk in chunks:
                if chunk.delta:
                    self._pieces.append(chunk.delta)
                if chunk.finish_reason:
                    self._finish_reason = chunk.finish_reason
                self._last_raw = chunk.raw
                yield chunk
        except Exception as exc:  # noqa: BLE001 - re-raised as StreamingError
            raise StreamingError(
                f"Error while consuming stream: {exc}", provider=self.provider
            ) from exc

    async def aconsume(self, chunks: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """Asynchronous counterpart to :meth:`consume`."""
        try:
            async for chunk in chunks:
                if chunk.delta:
                    self._pieces.append(chunk.delta)
                if chunk.finish_reason:
                    self._finish_reason = chunk.finish_reason
                self._last_raw = chunk.raw
                yield chunk
        except Exception as exc:  # noqa: BLE001 - re-raised as StreamingError
            raise StreamingError(
                f"Error while consuming stream: {exc}", provider=self.provider
            ) from exc

    def finalize(self) -> Response:
        """Build the final aggregated :class:`Response` after streaming ends."""
        return Response(
            text="".join(self._pieces),
            model=self.model,
            provider=self.provider,
            usage=Usage(),
            finish_reason=self._finish_reason,
            raw=self._last_raw,
        )
