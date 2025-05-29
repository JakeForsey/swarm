(
  JAX_PLATFORMS=cpu LLAMA_SERVER_HOST=cortex1:8080 uv run rq worker &
  JAX_PLATFORMS=cpu LLAMA_SERVER_HOST=cortex1:8080 uv run rq worker &

  JAX_PLATFORMS=cpu LLAMA_SERVER_HOST=cortex2:8080 uv run rq worker &
  JAX_PLATFORMS=cpu LLAMA_SERVER_HOST=cortex2:8080 uv run rq worker &

  JAX_PLATFORMS=cpu LLAMA_SERVER_HOST=cortex2:8081 uv run rq worker &
  JAX_PLATFORMS=cpu LLAMA_SERVER_HOST=cortex2:8081 uv run rq worker &

  wait
)