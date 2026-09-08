# TileMaxSim management API

`tilemaxsimd` exposes an HTTP/1.1 management surface on its status socket. The
contract version is `tilemaxsim.management.v1`.

## Transport and trust boundary

Read endpoints are available on the Unix status socket and, when configured,
the TCP status listener. Every management `POST` is rejected on TCP with 403;
writes are accepted only through the permission-controlled Unix socket. The TCP
listener must still remain on a private network because status and configuration
are operational information.

The daemon accepts neither chunked transfer encoding nor request bodies larger
than 1 MiB. It closes the connection after each response. Callers should discover
the current paths, limits, and supported runtime cache profiles from
`GET /v1/config` rather than assuming them.

## Read endpoints

| Endpoint | Meaning |
| --- | --- |
| `GET /livez` | Process is alive. |
| `GET /healthz` | Process accepts scoring work; returns 503 while starting or draining. |
| `GET /metrics` | Prometheus metrics. |
| `GET /v1/config` | Effective immutable configuration and management capabilities. |
| `GET /v1/cache` | Versioned, timestamped L0/L1/cache and accelerator snapshot. |
| `GET /v1/operations/{id}` | Process-local asynchronous operation state. |

Operation states are `queued`, `running`, `succeeded`, or `failed`. IDs and the
bounded 1,024-record history reset when the daemon restarts. Detailed errors are
available on the Unix socket; TCP responses redact local paths and vendor
diagnostics.

## Asynchronous writes

Successful submissions return HTTP 202:

```json
{
  "accepted": true,
  "operation_id": 42,
  "operation_url": "/v1/operations/42"
}
```

The executor queue holds at most 64 operations. A full queue or history returns
503. Once drain starts, further management writes return 409.

### Prewarm

`POST /v1/cache/prewarm`:

```json
{
  "descriptors": [{
    "candidate_id": 7,
    "contract": "model@version",
    "digest": "HEX_SHA256",
    "rows": 128,
    "dimension": 320,
    "dtype": 2
  }],
  "batch_size": 256,
  "pin": false
}
```

`dtype` uses the scoring protocol values (`1` for FP32 and `2` for FP16 source
tensors). Runtime prewarm currently populates only the `exact-fp16` cache
namespace and obeys normal admission limits. Startup resident manifests are the
only forced preload. A prewarm operation is idempotent but not atomic across
batches: if a later batch fails, successfully warmed earlier batches remain safe
and usable.

### Pin and unpin

`POST /v1/cache/pin` and `POST /v1/cache/unpin` accept the same `descriptors`
array; `batch_size` and `pin` are ignored. Every tensor must already be resident
in the `exact-fp16` namespace. The mutation is transactional within the request:
validation or pinned-budget failure restores the original pin states.

### Reload and circuit probe

- `POST /v1/reload` reloads the immutable shard indexes. Quantization activation
  state is read atomically on each PQ-family request and needs no reload.
  Existing in-flight requests retain their acquired data.
- `POST /v1/devices/{ordinal}/tensor-circuit/probe` requests an immediate
  half-open Tensor Core probe. It does not force a kernel choice and fails when
  the device or matrix engine is unavailable.

### Graceful drain

`POST /v1/drain` immediately withdraws readiness and stops accepting scoring or
management work. Already accepted requests finish according to their deadlines;
the daemon then removes its sockets and exits. Repeating drain is idempotent.

## Error handling

- 400: malformed HTTP/JSON, invalid identifier, or oversized/truncated request.
- 403: a management write was attempted through TCP.
- 404: endpoint or operation ID does not exist.
- 409: the daemon is draining.
- 503: operation queue/history is full or its executor is unavailable.

HTTP 202 means only that an operation was queued. Controllers must poll the
returned URL and treat only `succeeded` as completion.
