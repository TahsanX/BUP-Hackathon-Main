"""Shared async HTTP plumbing and status-code -> typed-error mapping."""

from __future__ import annotations

from typing import Any

import httpx

from gridwise.llm.types import (
    AuthFailure,
    ProviderTimeout,
    RateLimited,
    ServerError,
)


async def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    provider: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers or {})
    except httpx.TimeoutException as exc:
        raise ProviderTimeout(f"{provider} timed out after {timeout}s") from exc
    except httpx.HTTPError as exc:
        raise ServerError(f"{provider} transport failure: {exc}") from exc

    _raise_for_status(provider, response)

    try:
        return response.json()
    except ValueError as exc:
        raise ServerError(f"{provider} returned a non-JSON body") from exc


def _raise_for_status(provider: str, response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return

    detail = response.text[:300]
    if status in (401, 403):
        raise AuthFailure(f"{provider} rejected the credentials ({status})")
    if status == 429:
        raise RateLimited(f"{provider} rate limited ({status}): {detail}")
    if status == 404:
        raise ServerError(f"{provider} model or endpoint not found ({status}): {detail}")
    if status >= 500:
        raise ServerError(f"{provider} server error ({status}): {detail}")
    raise ServerError(f"{provider} rejected the request ({status}): {detail}")
