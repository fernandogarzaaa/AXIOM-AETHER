//! hardware.rs — auto-detect host capabilities and recommend safe runtime config.
//!
//! Goal: an open-source user clones Axiom and it "just works" on their hardware
//! without manual tuning — and, critically, **without crashing their machine**.
//!
//! The recurring failure this module defends against is VRAM contention: the
//! inference proxy and a training job both reaching for the same GPU and tripping
//! `CUDA_ERROR_OUT_OF_MEMORY` on a small card (observed on a 6 GB RTX 2060). The
//! co-tenancy guard below makes the proxy yield the GPU whenever a compute tenant
//! (e.g. a training run) is already resident, so the two never fight.
//!
//! Detection is best-effort and never panics: a missing `nvidia-smi`, an unusual
//! platform, or a parse failure degrades to a safe CPU recommendation rather than
//! an error. The decision logic (`recommend`) is a pure function of a
//! [`HardwareProfile`], so it is fully unit-testable with synthetic inputs on a
//! machine with no GPU (important for CI).

use std::fmt;

/// Minimum free VRAM (MB) the proxy needs before it will choose CUDA. Covers the
/// d256/4L production model plus activation headroom. Below this, CPU is safer.
const MIN_PROXY_VRAM_MB: u64 = 1500;

/// Minimum free VRAM (MB) before training will choose CUDA. Below this the
/// autograd graph for even a small model risks OOM, so fall back to CPU.
const MIN_TRAIN_VRAM_MB: u64 = 2000;

/// Fraction of *free* VRAM training is allowed to budget against (30% headroom).
/// Matches the long-standing probe in `train_semantic`.
const VRAM_BUDGET_FRACTION: f64 = 0.70;

/// Cores reserved for the OS / display compositor so a CPU-bound training job
/// cannot starve interrupt handling (the root cause of display tearing under
/// full-core load). Training threads = cores - this, floored at 1.
const OS_RESERVED_CORES: usize = 2;

/// A snapshot of the host's compute resources at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareProfile {
    pub has_cuda: bool,
    pub gpu_name: Option<String>,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    /// True when another CUDA compute application is already resident on the GPU
    /// (e.g. a training process). The proxy yields the GPU when this is set.
    pub gpu_busy: bool,
    pub cpu_cores: usize,
    pub ram_total_mb: u64,
    pub ram_free_mb: u64,
}

/// Which compute device a role should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceChoice {
    Cpu,
    Cuda,
}

impl DeviceChoice {
    /// The string the rest of the engine understands (`device_from_str`).
    pub fn as_device_str(self) -> &'static str {
        match self {
            DeviceChoice::Cpu => "cpu",
            DeviceChoice::Cuda => "cuda",
        }
    }
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_device_str())
    }
}

/// Safe runtime configuration derived from a [`HardwareProfile`].
#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    /// Device the inference proxy should bind to.
    pub proxy_device: DeviceChoice,
    /// Device a training job should use.
    pub training_device: DeviceChoice,
    /// VRAM budget (MB) training should size its model against (0 on CPU).
    pub vram_budget_mb: u64,
    /// Worker threads training should use without starving the OS.
    pub training_threads: usize,
    /// Human-readable explanation of the proxy-device decision.
    pub reason: String,
}

/// Pure decision logic: given a hardware snapshot, recommend safe config.
///
/// Encodes three hard-won rules:
/// 1. **Co-tenancy guard** — if the GPU already hosts a compute tenant, the proxy
///    uses CPU so it cannot OOM a running training job (or be OOM'd by it).
/// 2. **Free-VRAM floor** — the proxy only takes CUDA with real headroom.
/// 3. **OS core reservation** — training leaves cores for display/interrupts.
pub fn recommend(p: &HardwareProfile) -> Recommendation {
    let training_threads = p.cpu_cores.saturating_sub(OS_RESERVED_CORES).max(1);

    if !p.has_cuda {
        return Recommendation {
            proxy_device: DeviceChoice::Cpu,
            training_device: DeviceChoice::Cpu,
            vram_budget_mb: 0,
            training_threads,
            reason: "no CUDA device detected — CPU for all roles".to_string(),
        };
    }

    // Training device: CUDA when there is enough free VRAM, else CPU.
    let training_device = if p.vram_free_mb >= MIN_TRAIN_VRAM_MB {
        DeviceChoice::Cuda
    } else {
        DeviceChoice::Cpu
    };
    let vram_budget_mb = match training_device {
        DeviceChoice::Cuda => (p.vram_free_mb as f64 * VRAM_BUDGET_FRACTION) as u64,
        DeviceChoice::Cpu => 0,
    };

    // Proxy device: the co-tenancy guard is the crash fix.
    let (proxy_device, reason) = if p.gpu_busy {
        (
            DeviceChoice::Cpu,
            format!(
                "GPU already has a compute tenant (e.g. training) — proxy yields to CPU \
                 to avoid VRAM contention; {} MB free",
                p.vram_free_mb
            ),
        )
    } else if p.vram_free_mb < MIN_PROXY_VRAM_MB {
        (
            DeviceChoice::Cpu,
            format!(
                "only {} MB free VRAM (< {} MB floor) — proxy uses CPU",
                p.vram_free_mb, MIN_PROXY_VRAM_MB
            ),
        )
    } else {
        (
            DeviceChoice::Cuda,
            format!(
                "{} idle with {} MB free VRAM — proxy uses CUDA",
                p.gpu_name.clone().unwrap_or_else(|| "GPU".to_string()),
                p.vram_free_mb
            ),
        )
    };

    Recommendation {
        proxy_device,
        training_device,
        vram_budget_mb,
        training_threads,
        reason,
    }
}

/// Detect the live hardware profile. Best-effort and non-panicking.
pub fn detect() -> HardwareProfile {
    let cpu_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let (ram_total_mb, ram_free_mb) = detect_ram_mb();
    let gpu = detect_gpu();

    match gpu {
        Some(g) => HardwareProfile {
            has_cuda: true,
            gpu_name: Some(g.name),
            vram_total_mb: g.total_mb,
            vram_free_mb: g.free_mb,
            gpu_busy: g.busy,
            cpu_cores,
            ram_total_mb,
            ram_free_mb,
        },
        None => HardwareProfile {
            has_cuda: false,
            gpu_name: None,
            vram_total_mb: 0,
            vram_free_mb: 0,
            gpu_busy: false,
            cpu_cores,
            ram_total_mb,
            ram_free_mb,
        },
    }
}

struct GpuInfo {
    name: String,
    total_mb: u64,
    free_mb: u64,
    busy: bool,
}

/// Query the GPU via `nvidia-smi`. Returns None if the tool is absent or the
/// output cannot be parsed (→ treated as "no CUDA", a safe CPU fall-through).
fn detect_gpu() -> Option<GpuInfo> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?;
    let mut parts = line.split(',').map(|f| f.trim());
    let name = parts.next()?.to_string();
    let total_mb: u64 = parts.next()?.parse().ok()?;
    let free_mb: u64 = parts.next()?.parse().ok()?;
    let busy = gpu_has_compute_apps();
    Some(GpuInfo { name, total_mb, free_mb, busy })
}

/// True if any process is currently resident as a CUDA compute app on the GPU.
fn gpu_has_compute_apps() -> bool {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).lines().any(|l| !l.trim().is_empty())
        }
        _ => false,
    }
}

/// Best-effort total/free RAM in MB. Cross-platform; returns (0, 0) when unknown.
/// RAM is advisory only — the recommendation never blocks on it.
fn detect_ram_mb() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(txt) = std::fs::read_to_string("/proc/meminfo") {
            let kb = |key: &str| -> u64 {
                txt.lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0)
            };
            let total = kb("MemTotal:") / 1024;
            // MemAvailable is the realistic "free" figure on modern kernels.
            let avail = kb("MemAvailable:") / 1024;
            return (total, avail);
        }
    }
    #[cfg(target_os = "windows")]
    {
        // wmic is deprecated but present on virtually all Windows hosts; parse
        // FreePhysicalMemory (KB) and TotalVisibleMemorySize (KB) from OS info.
        if let Ok(o) = std::process::Command::new("wmic")
            .args(["OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/value"])
            .output()
        {
            let s = String::from_utf8_lossy(&o.stdout);
            let kv = |key: &str| -> u64 {
                s.lines()
                    .find(|l| l.trim_start().starts_with(key))
                    .and_then(|l| l.split('=').nth(1))
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .unwrap_or(0)
            };
            let total = kv("TotalVisibleMemorySize") / 1024;
            let free = kv("FreePhysicalMemory") / 1024;
            return (total, free);
        }
    }
    (0, 0)
}

/// Render a human-readable hardware + recommendation report (for `--mode doctor`).
pub fn report(p: &HardwareProfile, r: &Recommendation) -> String {
    let mut s = String::new();
    s.push_str("=== Axiom Hardware Doctor ===\n");
    s.push_str(&format!("CPU cores            : {}\n", p.cpu_cores));
    if p.ram_total_mb > 0 {
        s.push_str(&format!(
            "System RAM           : {} MB free / {} MB total\n",
            p.ram_free_mb, p.ram_total_mb
        ));
    } else {
        s.push_str("System RAM           : (unknown)\n");
    }
    if p.has_cuda {
        s.push_str(&format!(
            "GPU                  : {} ({} MB free / {} MB total){}\n",
            p.gpu_name.clone().unwrap_or_else(|| "CUDA device".to_string()),
            p.vram_free_mb,
            p.vram_total_mb,
            if p.gpu_busy { ", BUSY (compute tenant resident)" } else { "" }
        ));
    } else {
        s.push_str("GPU                  : none detected (CPU-only host)\n");
    }
    s.push_str("--- Recommendation ---\n");
    s.push_str(&format!("Proxy device         : {}\n", r.proxy_device));
    s.push_str(&format!("Training device      : {}\n", r.training_device));
    if r.vram_budget_mb > 0 {
        s.push_str(&format!("Training VRAM budget : {} MB (30% headroom)\n", r.vram_budget_mb));
    }
    s.push_str(&format!("Training threads     : {}\n", r.training_threads));
    s.push_str(&format!("Why                  : {}\n", r.reason));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> HardwareProfile {
        HardwareProfile {
            has_cuda: true,
            gpu_name: Some("Test GPU".into()),
            vram_total_mb: 6144,
            vram_free_mb: 5955,
            gpu_busy: false,
            cpu_cores: 8,
            ram_total_mb: 20000,
            ram_free_mb: 12000,
        }
    }

    #[test]
    fn no_cuda_is_all_cpu() {
        let mut p = base();
        p.has_cuda = false;
        let r = recommend(&p);
        assert_eq!(r.proxy_device, DeviceChoice::Cpu);
        assert_eq!(r.training_device, DeviceChoice::Cpu);
        assert_eq!(r.vram_budget_mb, 0);
    }

    #[test]
    fn idle_gpu_with_headroom_uses_cuda_for_proxy() {
        let r = recommend(&base());
        assert_eq!(r.proxy_device, DeviceChoice::Cuda);
        assert_eq!(r.training_device, DeviceChoice::Cuda);
    }

    #[test]
    fn busy_gpu_yields_proxy_to_cpu() {
        // The co-tenancy guard: training resident → proxy must NOT take the GPU.
        let mut p = base();
        p.gpu_busy = true;
        let r = recommend(&p);
        assert_eq!(r.proxy_device, DeviceChoice::Cpu, "proxy must yield to training");
        // Training still targets CUDA (it is the resident tenant).
        assert_eq!(r.training_device, DeviceChoice::Cuda);
    }

    #[test]
    fn low_free_vram_keeps_proxy_on_cpu() {
        let mut p = base();
        p.vram_free_mb = 800; // below MIN_PROXY_VRAM_MB
        let r = recommend(&p);
        assert_eq!(r.proxy_device, DeviceChoice::Cpu);
    }

    #[test]
    fn very_low_vram_forces_training_to_cpu() {
        let mut p = base();
        p.vram_free_mb = 1000; // below MIN_TRAIN_VRAM_MB
        let r = recommend(&p);
        assert_eq!(r.training_device, DeviceChoice::Cpu);
        assert_eq!(r.vram_budget_mb, 0);
    }

    #[test]
    fn vram_budget_applies_30pct_headroom() {
        let p = base(); // 5955 free
        let r = recommend(&p);
        let expected = (5955f64 * 0.70) as u64;
        assert_eq!(r.vram_budget_mb, expected);
    }

    #[test]
    fn training_threads_reserve_os_cores() {
        let p = base(); // 8 cores
        let r = recommend(&p);
        assert_eq!(r.training_threads, 6); // 8 - 2 reserved
    }

    #[test]
    fn training_threads_floor_at_one() {
        let mut p = base();
        p.cpu_cores = 1;
        let r = recommend(&p);
        assert_eq!(r.training_threads, 1);
    }

    #[test]
    fn report_is_nonempty_and_mentions_devices() {
        let p = base();
        let r = recommend(&p);
        let txt = report(&p, &r);
        assert!(txt.contains("Proxy device"));
        assert!(txt.contains("Training device"));
    }
}
