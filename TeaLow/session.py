"""HTTP session management for the TeaLow SDK.

Wraps :mod:`requests` (synchronous) and :mod:`httpx` (asynchronous) to
provide a consistent interface for making HTTP calls to provider APIs,
including proxy support, custom headers, timeouts, and translation of
transport-level failures into TeaLow exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import httpx
import requests

from .constants import DEFAULT_TIMEOUT, USER_AGENT
from .exceptions import (
    APIResponseError,
    ConnectionError as TeaLowConnectionError,
    InvalidAPIKeyError,
    RateLimitError,
    TimeoutError as TeaLowTimeoutError,
)
from .utils import get_logger, is_rate_limit_status

logger = get_logger("session")


class HTTPSession:
    """Synchronous HTTP session wrapper built on :mod:`requests`.

    Args:
        timeout: Default timeout (seconds) applied to every request
            unless overridden per-call.
        proxies: Optional proxy configuration, e.g.
            ``{"http": "http://proxy:8080", "https": "http://proxy:8080"}``.
        custom_headers: Optional headers merged into every outbound request.
        verify_ssl: Whether to verify TLS certificates.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        proxies: Optional[Dict[str, str]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        if proxies:
            self._session.proxies.update(proxies)
        self._session.verify = verify_ssl
        self._session.headers.update({"User-Agent": USER_AGENT})
        if custom_headers:
            self._session.headers.update(custom_headers)

    @property
    def headers(self) -> Dict[str, str]:
        """Return the default headers configured on this session."""
        return dict(self._session.headers)

    def update_headers(self, headers: Dict[str, str]) -> None:
        """Merge additional default headers into this session."""
        self._session.headers.update(headers)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
        provider: Optional[str] = None,
    ) -> requests.Response:
        """Issue an HTTP request and translate low-level errors.

        Raises:
            TeaLowTimeoutError: If the request times out.
            TeaLowConnectionError: On DNS/connection failures.
            InvalidAPIKeyError: On HTTP 401/403 responses.
            RateLimitError: On HTTP 429 responses.
            APIResponseError: On other non-2xx responses.
        """
        try:
            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                params=params,
                data=data,
                files=files,
                timeout=timeout if timeout is not None else self.timeout,
                stream=stream,
            )
        except requests.exceptions.Timeout as exc:
            raise TeaLowTimeoutError(
                f"Request to {url} timed out.", provider=provider
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise TeaLowConnectionError(
                f"Failed to connect to {url}: {exc}", provider=provider
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise TeaLowConnectionError(
                f"Request to {url} failed: {exc}", provider=provider
            ) from exc

        self._raise_for_status(response, provider=provider)
        return response

    @staticmethod
    def _raise_for_status(
        response: requests.Response, *, provider: Optional[str]
    ) -> None:
        if response.ok:
            return

        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text

        status = response.status_code
        message = f"Provider '{provider}' returned HTTP {status}."

        if status in (401, 403):
            raise InvalidAPIKeyError(
                message, provider=provider, status_code=status, response_body=body
            )
        if is_rate_limit_status(status):
            retry_after_header = response.headers.get("Retry-After")
            retry_after = None
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = None
            raise RateLimitError(
                message,
                provider=provider,
                status_code=status,
                response_body=body,
                retry_after=retry_after,
            )
        raise APIResponseError(
            message, provider=provider, status_code=status, response_body=body
        )

    def stream_lines(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> Iterator[str]:
        """Issue a streaming request and yield decoded lines as they arrive."""
        response = self.request(
            method,
            url,
            headers=headers,
            json=json,
            timeout=timeout,
            stream=True,
            provider=provider,
        )
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line:
                    yield raw_line
        finally:
            response.close()

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._session.close()

    def __enter__(self) -> "HTTPSession":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class AsyncHTTPSession:
    """Asynchronous HTTP session wrapper built on :mod:`httpx`.

    Mirrors :class:`HTTPSession` but exposes ``async``/``await``
    compatible methods for use from :class:`TeaLow.async_client.AsyncTeaLow`.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        proxies: Optional[Dict[str, str]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ) -> None:
        self.timeout = timeout
        headers = {"User-Agent": USER_AGENT}
        if custom_headers:
            headers.update(custom_headers)
        client_kwargs: Dict[str, Any] = {
            "timeout": timeout,
            "headers": headers,
            "verify": verify_ssl,
        }
        if proxies:
            mounts = {}
            for scheme, proxy_url in proxies.items():
                pattern = f"{scheme}://" if not scheme.endswith("://") else scheme
                mounts[pattern] = httpx.AsyncHTTPTransport(proxy=proxy_url)
            client_kwargs["mounts"] = mounts
        self._client = httpx.AsyncClient(**client_kwargs)

    def update_headers(self, headers: Dict[str, str]) -> None:
        """Merge additional default headers into this session."""
        self._client.headers.update(headers)

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> httpx.Response:
        """Issue an asynchronous HTTP request and translate low-level errors."""
        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                params=params,
                timeout=timeout if timeout is not None else self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise TeaLowTimeoutError(
                f"Request to {url} timed out.", provider=provider
            ) from exc
        except httpx.ConnectError as exc:
            raise TeaLowConnectionError(
                f"Failed to connect to {url}: {exc}", provider=provider
            ) from exc
        except httpx.HTTPError as exc:
            raise TeaLowConnectionError(
                f"Request to {url} failed: {exc}", provider=provider
            ) from exc

        self._raise_for_httpx_status(response, provider=provider)
        return response

    @staticmethod
    def _raise_for_httpx_status(
        response: httpx.Response, *, provider: Optional[str]
    ) -> None:
        if response.status_code < 400:
            return

        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text

        status = response.status_code
        message = f"Provider '{provider}' returned HTTP {status}."

        if status in (401, 403):
            raise InvalidAPIKeyError(
                message, provider=provider, status_code=status, response_body=body
            )
        if is_rate_limit_status(status):
            retry_after_header = response.headers.get("Retry-After")
            retry_after = None
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = None
            raise RateLimitError(
                message,
                provider=provider,
                status_code=status,
                response_body=body,
                retry_after=retry_after,
            )
        raise APIResponseError(
            message, provider=provider, status_code=status, response_body=body
        )

    async def stream_lines(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> Any:
        """Issue a streaming request, yielding decoded lines as they arrive."""
        req_timeout = timeout if timeout is not None else self.timeout
        async with self._client.stream(
            method, url, headers=headers, json=json, timeout=req_timeout
        ) as response:
            self._raise_for_httpx_status(response, provider=provider)
            async for raw_line in response.aiter_lines():
                if raw_line:
                    yield raw_line

    async def close(self) -> None:
        """Close the underlying async client."""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncHTTPSession":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()
