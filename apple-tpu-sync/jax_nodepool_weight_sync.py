#!/usr/bin/env python3
"""Pure JAX cross-nodepool weight transfer & 2D resharding via tpu-sync."""

import argparse
import asyncio
import datetime
import functools
import gc
import os
import socket
import time

_T0 = time.perf_counter()
_ROLE_TAG = "INIT"
_RANK_TAG = os.environ.get("JOB_COMPLETION_INDEX", "?")


def log_stage(stage: str, msg: str) -> None:
  ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
  elapsed = time.perf_counter() - _T0
  print(
      f"[{ts}Z +{elapsed:6.2f}s] [{_ROLE_TAG} rank={_RANK_TAG}] {stage}: {msg}",
      flush=True,
  )


# Bypass GCE Metadata Server lookups (blocked/delayed by GKE Workload Identity)
# and provide explicit 4x8 v5e topology bounds (8 hosts x 4 chips/host) before
# importing JAX/libtpu. Keep GKE's TPU_WORKER_HOSTNAMES & TPU_WORKER_ID intact!
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "2,4,1")

for _k in list(os.environ):
  if _k.startswith("MEGASCALE_"):
    removed = os.environ.pop(_k, None)
    log_stage("PY STAGE 1/8", f"Stripped env {_k}={removed}")

log_stage(
    "PY STAGE 1/8",
    f"Started on host={socket.gethostname()} "
    f"TPU_WORKER_ID={os.environ.get('TPU_WORKER_ID')} "
    f"TPU_HOST_BOUNDS={os.environ.get('TPU_HOST_BOUNDS')} "
    f"TPU_SKIP_MDS_QUERY={os.environ.get('TPU_SKIP_MDS_QUERY')}",
)

log_stage("PY STAGE 2/8", "Importing jax and tpu_sync C++ extensions...")
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import weight_synchronizer
from tpu_sync.frameworks.jax import utils
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2
log_stage("PY STAGE 2/8", f"Imports complete (jax={jax.__version__}).")


def resolve_local_ip() -> str:
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    sock.connect(("10.255.255.255", 1))
    return sock.getsockname()[0]
  finally:
    sock.close()


def wait_for_dns(host_port: str, timeout_s: float = 180.0) -> str:
  host, port_s = (
      host_port.rsplit(":", 1) if ":" in host_port else (host_port, "8471")
  )
  port = int(port_s)
  deadline = time.time() + timeout_s
  attempt = 0
  while time.time() < deadline:
    attempt += 1
    try:
      infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
      resolved_ip = infos[0][4][0]
      log_stage("DNS", f"Resolved {host}:{port} -> {resolved_ip} (attempt {attempt})")
      return resolved_ip
    except socket.gaierror as exc:
      if attempt == 1 or attempt % 5 == 0:
        log_stage("DNS", f"Waiting for {host}:{port} (attempt {attempt}: {exc})...")
      time.sleep(2.0)
  raise TimeoutError(f"DNS did not resolve {host_port} within {timeout_s}s")


def create_mesh(mesh_str: str) -> jax.sharding.Mesh:
  dims = tuple(int(x) for x in mesh_str.split(","))
  devices = jax.devices("tpu")
  if len(devices) != int(np.prod(dims)):
    raise ValueError(
        f"Mesh {dims} expects {int(np.prod(dims))} TPU chips, found {len(devices)}"
    )
  try:
    dev_grid = mesh_utils.create_device_mesh(dims, devices)
  except (ValueError, RuntimeError):
    dev_grid = np.array(devices).reshape(dims)
  return jax.sharding.Mesh(dev_grid, ("fsdp", "tp"))


def expected_tensor_fn(shape: tuple[int, int], var_idx: int, dtype) -> jax.Array:
  rows = jnp.arange(shape[0], dtype=jnp.float32)[:, None] * 1e-4
  cols = jnp.arange(shape[1], dtype=jnp.float32)[None, :] * 1e-5
  return (rows + cols + float(var_idx + 1) * 0.25).astype(dtype)


def build_variable_protos(
    arrays: list[jax.Array], sharding: jax.sharding.NamedSharding
) -> list[raiden_service_pb2.VariableMetadataProto]:
  protos = []
  spec_axes = [a if isinstance(a, str) else "" for a in sharding.spec]
  for idx, arr in enumerate(arrays):
    g_shape = tuple(arr.shape)
    l_shape = sharding.shard_shape(g_shape)
    sharding_shape = [g // l for g, l in zip(g_shape, l_shape)]
    layout = tuple(range(len(g_shape) - 1, -1, -1))
    g_indices = utils.global_shard_indices_for_array(
        arr, [s.device for s in arr.addressable_shards]
    )
    protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=f"layer_{idx}.weight",
            shape=g_shape,
            mesh_shape=sharding_shape,
            layout=layout,
            item_size=arr.dtype.itemsize,
            layer_idx=idx,
            sharding_spec=spec_axes,
            global_shard_indices=g_indices,
        )
    )
  return protos


def extract_shard_endpoints(
    ws: weight_synchronizer.WeightSynchronizer,
    arr: jax.Array,
    mesh: jax.sharding.Mesh,
    self_ip: str,
) -> list[str]:
  local_shards = list(arr.addressable_shards)
  n_local = len(local_shards)
  raw_eps = ws.get_local_endpoints()
  if not raw_eps:
    return [f"{self_ip}:{ws.local_port}"] * n_local
  flat_devs = list(mesh.devices.flat)
  out = [""] * n_local
  for i, s in enumerate(local_shards):
    g_idx = flat_devs.index(s.device)
    for ep in raw_eps:
      ep_s = ep.get("shards") or []
      ep_g = ep.get("global_shards") or ep_s
      if i in ep_s or g_idx in ep_g:
        out[i] = str(ep["endpoint"])
        break
    if not out[i]:
      out[i] = f"{self_ip}:{ws.local_port}"
  return out


def register_with_retry(
    ctrl_addr: str,
    unit: raiden_controller.RaidenId,
    shards: list[str],
    listener_addr: str,
    mesh: jax.sharding.Mesh,
    var_protos: list[raiden_service_pb2.VariableMetadataProto],
) -> raiden_controller.RaidenControllerClientFacade:
  wait_for_dns(ctrl_addr)
  client = raiden_controller.RaidenControllerClientFacade(
      ctrl_addr, name_resolver=None
  )
  for attempt in range(1, 91):
    try:
      client.register_work_unit(
          unit,
          shards,
          listener_addr,
          mesh_shape=[mesh.shape["fsdp"], mesh.shape["tp"]],
          variables=var_protos,
          mesh_axes=["fsdp", "tp"],
      )
      return client
    except Exception as exc:  # pylint: disable=broad-except
      if attempt == 90:
        raise
      if attempt == 1 or attempt % 5 == 0:
        log_stage(
            "PY STAGE 6/8",
            f"Waiting for RaidenController at {ctrl_addr} (attempt {attempt}: {exc})...",
        )
      time.sleep(2.0)
  return client


def init_jax_slice(args: argparse.Namespace) -> tuple[int, str]:
  self_ip = resolve_local_ip()
  log_stage("PY STAGE 3/8", f"Local Pod IP={self_ip}. Verifying all slice peer DNS entries...")

  # Ensure all 8 slice hostnames in TPU_WORKER_HOSTNAMES resolve before libtpu starts
  worker_hosts = os.environ.get("TPU_WORKER_HOSTNAMES", "")
  if worker_hosts:
    for h in worker_hosts.split(","):
      if h.strip():
        wait_for_dns(f"{h.strip()}:8471")
  wait_for_dns(args.jax_coordinator)

  log_stage(
      "PY STAGE 3/8",
      f"Calling jax.distributed.initialize(coordinator={args.jax_coordinator}, "
      f"num_processes={args.num_processes}, process_id={args.process_id}, "
      "cluster_detection_method='deactivate')...",
  )
  jax.distributed.initialize(
      coordinator_address=args.jax_coordinator,
      num_processes=args.num_processes,
      process_id=args.process_id,
      cluster_detection_method="deactivate",
  )
  proc = jax.process_index()
  log_stage(
      "PY STAGE 3/8",
      f"jax.distributed.initialize() SUCCEEDED! process_index={proc}/{jax.process_count()}",
  )

  log_stage("PY STAGE 4/8", "Calling jax.devices('tpu') to initialize 4x8 libtpu slice...")
  t_tpu = time.perf_counter()
  local_devs = jax.local_devices()
  all_devs = jax.devices("tpu")
  log_stage(
      "PY STAGE 4/8",
      f"TPU backend READY in {time.perf_counter() - t_tpu:.2f}s! "
      f"local_devices={len(local_devs)}, global_devices={len(all_devs)}",
  )
  return proc, self_ip


def run_source(args: argparse.Namespace) -> None:
  proc, self_ip = init_jax_slice(args)

  ctrl_server = None
  worker_rpc_client = None
  if proc == 0:
    ctrl_port = int(args.controller_address.rsplit(":", 1)[1])
    log_stage("PY STAGE 5/8", f"Starting RaidenControllerServer on 0.0.0.0:{ctrl_port}...")
    worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
    controller = raiden_controller.RaidenController(
        port=ctrl_port,
        worker_rpc_client=worker_rpc_client,
    )
    controller.start_transfer = functools.partial(
        controller.start_transfer,
        parallelism=args.parallelism,
        skip_d2h=True,
        group_size=args.group_size,
    )
    ctrl_server = raiden_controller.RaidenControllerServer(controller)
    ctrl_server.start()
    log_stage("PY STAGE 5/8", f"RaidenControllerServer ACTIVE on :{ctrl_port}")

  multihost_utils.sync_global_devices("source_controller_ready")

  log_stage(
      "PY STAGE 5/8",
      f"Building source mesh ({args.src_mesh}) & allocating {args.num_vars} tensors of ({args.var_shape})...",
  )
  src_mesh = create_mesh(args.src_mesh)
  pspec = jax.sharding.PartitionSpec("fsdp", "tp")
  src_sharding = jax.sharding.NamedSharding(src_mesh, pspec)
  shape = tuple(int(x) for x in args.var_shape.split(","))
  dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

  src_vars = [
      jax.jit(
          functools.partial(expected_tensor_fn, shape, i, dtype),
          out_shardings=src_sharding,
      )()
      for i in range(args.num_vars)
  ]
  jax.block_until_ready(src_vars)
  log_stage("PY STAGE 5/8", "Source HBM weight tensors allocated.")

  log_stage("PY STAGE 6/8", "Creating WeightSynchronizer & staging HBM -> Host DRAM (D2H)...")
  ws = weight_synchronizer.WeightSynchronizer(
      src_vars,
      local_port=0,
      listener_port=0,
      parallelism=args.parallelism,
      unsafe_skip_buffer_lock=False,
      bind_ip=self_ip,
      auto_h2d=False,
  )

  t0 = time.perf_counter()
  ws.d2h()
  multihost_utils.sync_global_devices("source_d2h_complete")
  d2h_s = time.perf_counter() - t0
  total_gb = sum(np.prod(a.shape) * a.dtype.itemsize for a in src_vars) / 1e9
  log_stage(
      "PY STAGE 6/8",
      f"D2H staging complete: {total_gb:.3f} GB in {d2h_s*1e3:.2f} ms ({total_gb/max(d2h_s, 1e-9):.2f} GB/s)",
  )

  shards = extract_shard_endpoints(ws, src_vars[0], src_mesh, self_ip)
  var_protos = build_variable_protos(src_vars, src_sharding)
  unit = raiden_controller.RaidenId("jax_source", str(proc), "weights")
  register_with_retry(
      args.controller_address,
      unit,
      shards,
      f"{self_ip}:{ws.listener_port}",
      src_mesh,
      var_protos,
  )
  log_stage(
      "PY STAGE 7/8",
      f"Registered {len(shards)} source shards (data={shards[0]}, ctrl={self_ip}:{ws.listener_port}). "
      "Serving transfers until destination completes...",
  )

  if proc == 0 and ctrl_server is not None:
    while not ctrl_server._server._stopped:  # pylint: disable=protected-access
      time.sleep(0.5)
    log_stage("PY STAGE 8/8", "Destination signaled shutdown. Stopping workers & controller...")
    loop = asyncio.new_event_loop()
    try:
      loop.run_until_complete(worker_rpc_client.shutdown_workers())
    finally:
      loop.close()
    ctrl_server.stop()

  multihost_utils.sync_global_devices("source_all_transfers_done")
  ws.shutdown()
  del ws
  gc.collect()
  log_stage("PY STAGE 8/8", "Source exited cleanly.")


def run_destination(args: argparse.Namespace) -> None:
  proc, self_ip = init_jax_slice(args)

  log_stage(
      "PY STAGE 5/8",
      f"Building destination mesh ({args.dst_mesh}) & allocating {args.num_vars} zeroed tensors...",
  )
  dst_mesh = create_mesh(args.dst_mesh)
  pspec = jax.sharding.PartitionSpec("fsdp", "tp")
  dst_sharding = jax.sharding.NamedSharding(dst_mesh, pspec)
  shape = tuple(int(x) for x in args.var_shape.split(","))
  dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

  dst_vars = [
      jax.jit(lambda: jnp.zeros(shape, dtype=dtype), out_shardings=dst_sharding)()
      for _ in range(args.num_vars)
  ]
  jax.block_until_ready(dst_vars)
  log_stage("PY STAGE 5/8", "Destination HBM tensors allocated.")

  log_stage("PY STAGE 6/8", "Creating WeightSynchronizer & registering destination shards...")
  ws = weight_synchronizer.WeightSynchronizer(
      dst_vars,
      local_port=0,
      listener_port=0,
      parallelism=args.parallelism,
      unsafe_skip_buffer_lock=False,
      bind_ip=self_ip,
      auto_h2d=False,
  )

  shards = extract_shard_endpoints(ws, dst_vars[0], dst_mesh, self_ip)
  var_protos = build_variable_protos(dst_vars, dst_sharding)
  unit = raiden_controller.RaidenId("jax_dest", str(proc), "weights")
  ctrl_client = register_with_retry(
      args.controller_address,
      unit,
      shards,
      f"{self_ip}:{ws.listener_port}",
      dst_mesh,
      var_protos,
  )
  log_stage(
      "PY STAGE 6/8",
      f"Registered {len(shards)} dest shards (data={shards[0]}, ctrl={self_ip}:{ws.listener_port}).",
  )

  total_chips = len(jax.devices("tpu"))
  poll = 0
  while True:
    poll += 1
    meta = ctrl_client.get_metadata()
    src_m = [m for m in meta if m.unit.job_name == "jax_source"]
    dst_m = [m for m in meta if m.unit.job_name == "jax_dest"]
    n_src = sum(len(m.shards) for m in src_m)
    n_dst = sum(len(m.shards) for m in dst_m)
    if proc == 0 and (poll == 1 or poll % 5 == 0):
      log_stage(
          "PY STAGE 7/8",
          f"Waiting for shard registrations: src={n_src}/{total_chips}, dst={n_dst}/{total_chips}",
      )
    if n_src >= total_chips and n_dst >= total_chips:
      src_units = sorted(
          [
              raiden_controller.RaidenId(
                  m.unit.job_name, m.unit.job_replica_id, m.unit.data_name
              )
              for m in src_m
          ],
          key=lambda u: int(u.job_replica_id),
      )
      dst_units = sorted(
          [
              raiden_controller.RaidenId(
                  m.unit.job_name, m.unit.job_replica_id, m.unit.data_name
              )
              for m in dst_m
          ],
          key=lambda u: int(u.job_replica_id),
      )
      break
    time.sleep(1.0)

  multihost_utils.sync_global_devices("all_units_registered")
  log_stage("PY STAGE 7/8", "All 32 source + 32 dest shards registered! Computing reference shards...")

  expected_vars = [
      jax.jit(
          functools.partial(expected_tensor_fn, shape, i, dtype),
          out_shardings=dst_sharding,
      )()
      for i in range(args.num_vars)
  ]
  jax.block_until_ready(expected_vars)

  total_bytes = sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in dst_vars)
  total_gb = total_bytes / 1e9
  net_times, h2d_times = [], []

  for it in range(args.num_iters + 1):
    sync_uuid = 1000 + it
    tag = "WARMUP" if it == 0 else f"ITER {it}/{args.num_iters}"
    multihost_utils.sync_global_devices(f"iter_{it}_start")
    if proc == 0:
      log_stage("PY STAGE 7/8", f"[{tag}] Starting cross-nodepool H2H transfer (uuid={sync_uuid})...")

    t_net = time.perf_counter()
    if proc == 0:
      ctrl_client.coordinate_transfer(
          src_units=src_units,
          dst_units=dst_units,
          req_id=f"weight_sync_{it}",
          use_block_chunks=True,
          is_sender=True,
          uuid=sync_uuid,
      )
    ws.wait_for_transfer_completion(sync_uuid)
    multihost_utils.sync_global_devices(f"iter_{it}_net_done")
    net_s = time.perf_counter() - t_net

    t_h2d = time.perf_counter()
    ws.h2d()
    jax.block_until_ready(dst_vars)
    multihost_utils.sync_global_devices(f"iter_{it}_h2d_done")
    h2d_s = time.perf_counter() - t_h2d

    if proc == 0:
      log_stage(
          "PY STAGE 7/8",
          f"[{tag}] Done: Net={net_s*1e3:.2f} ms ({total_gb/net_s:.2f} GB/s), "
          f"H2D={h2d_s*1e3:.2f} ms ({total_gb/h2d_s:.2f} GB/s)",
      )

    if it == 0:
      for idx, (got, exp) in enumerate(zip(dst_vars, expected_vars)):
        for s_got, s_exp in zip(got.addressable_shards, exp.addressable_shards):
          if not bool(jnp.allclose(s_got.data, s_exp.data, atol=1e-2, rtol=1e-2)):
            raise AssertionError(f"[DEST rank={proc}] Mismatch on layer_{idx}!")
      multihost_utils.sync_global_devices("warmup_verified")
      if proc == 0:
        log_stage("PY STAGE 7/8", "[WARMUP] 2D reshard numerical check PASSED across all 32 chips!")
    else:
      net_times.append(net_s)
      h2d_times.append(h2d_s)

  if proc == 0:
    avg_net = float(np.mean(net_times))
    avg_h2d = float(np.mean(h2d_times))
    print("=" * 72, flush=True)
    print(f"CROSS-NODEPOOL JAX WEIGHT SYNC SUMMARY ({args.src_mesh} -> {args.dst_mesh})", flush=True)
    print("=" * 72, flush=True)
    print(f"Payload Size      : {total_bytes / (1024**2):.2f} MiB ({args.num_vars} tensors of {shape})", flush=True)
    print(f"Streams / Worker  : {args.parallelism}", flush=True)
    print(f"Avg Network (H2H) : {avg_net*1e3:.2f} ms | {total_gb/avg_net:.2f} GB/s ({total_gb*8/avg_net:.2f} Gbps)", flush=True)
    print(f"Avg Device (H2D)  : {avg_h2d*1e3:.2f} ms | {total_gb/avg_h2d:.2f} GB/s", flush=True)
    print("=" * 72, flush=True)
    ctrl_client.shutdown()

  multihost_utils.sync_global_devices("dest_all_transfers_done")
  ws.shutdown()
  del ws
  gc.collect()
  log_stage("PY STAGE 8/8", "Destination exited cleanly.")


def main() -> None:
  global _ROLE_TAG, _RANK_TAG
  p = argparse.ArgumentParser()
  p.add_argument("--role", choices=["source", "destination"], required=True)
  p.add_argument("--controller_address", required=True)
  p.add_argument("--jax_coordinator", required=True)
  p.add_argument("--num_processes", type=int, default=8)
  p.add_argument("--process_id", type=int, required=True)
  p.add_argument("--src_mesh", default="4,8")
  p.add_argument("--dst_mesh", default="8,4")
  p.add_argument("--num_vars", type=int, default=16)
  p.add_argument("--var_shape", default="8192,8192")
  p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
  p.add_argument("--parallelism", type=int, default=8)
  p.add_argument("--group_size", type=int, default=4)
  p.add_argument("--num_iters", type=int, default=5)
  args = p.parse_args()

  _ROLE_TAG = args.role.upper()
  _RANK_TAG = str(args.process_id)

  if args.role == "source":
    run_source(args)
  else:
    run_destination(args)


if __name__ == "__main__":
  main()