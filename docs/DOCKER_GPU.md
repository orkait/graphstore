---
name: Docker context gotcha
description: Docker Desktop hijacks docker CLI on this host - must switch to default context for GPU/nvidia access
type: project
originSessionId: 3de25efd-041d-48dd-8f15-cc2029386fdb
---
Host has Docker Desktop (QEMU VM at /home/kai/.docker/desktop/) AND native dockerd. `docker` CLI defaults to `desktop-linux` context which routes to the VM - no GPU passthrough.

**Fix before any GPU docker work:**
```
docker context use default
```

Verify with: `docker info | grep -iE "runtime|cdi"` - should show nvidia in Runtimes + nvidia.com/gpu CDI devices.

Switch back: `docker context use desktop-linux`

**Why:** two hours wasted debugging "CDI unresolvable" errors that were actually just wrong daemon socket. Native dockerd has nvidia runtime, CDI spec, and GPU access. Docker Desktop VM has none of these.
