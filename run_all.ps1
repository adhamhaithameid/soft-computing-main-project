param(
  [ValidateSet("cpu", "gpu")]
  [string]$Mode = "cpu",
  [ValidateSet("auto", "linux", "windows", "mac")]
  [string]$PlatformProfile = "auto",
  [switch]$Fresh,
  [switch]$AllowPartial,
  [switch]$StrictDevice,
  [int]$CheckpointEvery = 120,
  [int]$CheckpointPercent = 5,
  [int]$Jobs = 1,
  [int]$SelectionJobs = 1,
  [int]$MaxRows = -1,
  [switch]$ShowConvergenceWarnings
)

$argsList = @(
  "run_all.py",
  "--mode", $Mode,
  "--platform-profile", $PlatformProfile,
  "--non-interactive",
  "--checkpoint-every", "$CheckpointEvery",
  "--checkpoint-percent", "$CheckpointPercent",
  "--jobs", "$Jobs",
  "--selection-jobs", "$SelectionJobs"
)

if ($Fresh) { $argsList += "--fresh" }
if ($AllowPartial) { $argsList += "--allow-partial" }
if ($StrictDevice) { $argsList += "--strict-device" }
if ($MaxRows -gt 0) { $argsList += @("--max-rows", "$MaxRows") }
if ($ShowConvergenceWarnings) { $argsList += "--show-convergence-warnings" }

python @argsList
