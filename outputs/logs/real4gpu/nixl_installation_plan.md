# Isolated NIXL installation plan (not executed)

## Preferred: native vLLM NixlConnector

```bash
conda create -n vllm-nixl --clone vllm
conda activate vllm-nixl
python -m pip install nixl
python - <<'PY'
import vllm, nixl
print("vllm:", vllm.__file__)
print("nixl:", nixl)
assert vllm.__file__.startswith("/home/tjfeng/vllm/"), vllm.__file__
PY
```

Stop immediately if `vllm.__file__` points to a site-packages copy instead of
`/home/tjfeng/vllm`. Do not run `pip install vllm`.

## Secondary: LMCache + NIXL

```bash
conda create -n vllm-lmcache --clone vllm
conda activate vllm-lmcache
python -m pip install lmcache nixl
python - <<'PY'
import vllm, lmcache, nixl
print("vllm:", vllm.__file__)
print("lmcache:", lmcache.__file__)
print("nixl:", nixl)
assert vllm.__file__.startswith("/home/tjfeng/vllm/"), vllm.__file__
PY
```

Cloning prevents dependency changes from damaging the existing colocated/TP4
environment. Installation success does not prove that NIXL/UCX KV transfer or
real PD will work on this host.
