param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [switch]$Headless,
    [switch]$BuildSceneOnly,
    [switch]$RebuildScene,
    [switch]$SmokeTest,
    [switch]$GripperSweep,
    [switch]$Experiment3
)

$ErrorActionPreference = 'Stop'

$isaacRoot = $env:ISAAC_SIM_ROOT
if (-not $isaacRoot) {
    $isaacRoot = 'E:\AIRobotic\isaac-sim'
}

$pythonBat = Join-Path $isaacRoot 'python.bat'
$entrypoint = Join-Path $ProjectRoot 'simulation\isaac\start_simulation.py'
$scenePath = Join-Path $ProjectRoot 'simulation\scenes\mecharm_pick_place.usd'
$bridgeLib = Join-Path $isaacRoot 'exts\isaacsim.ros2.bridge\humble\lib'
$rosLogDir = Join-Path $ProjectRoot 'log\ros2'

if (-not (Test-Path -LiteralPath $pythonBat -PathType Leaf)) {
    throw "Isaac Sim Python launcher not found: $pythonBat"
}
if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
    throw "Experiment entry point not found: $entrypoint"
}

$env:ISAAC_SIM_ROOT = $isaacRoot
$env:ROS_DISTRO = 'humble'
$env:MECHARM_TRANSPORT = 'tcp'
$env:MECHARM_TCP_BIND = '0.0.0.0'
$env:MECHARM_TCP_PORT = '8765'
$env:MECHARM_SELF_COLLISION = 'selective'
Remove-Item Env:ROS_DISCOVERY_SERVER -ErrorAction SilentlyContinue
Remove-Item Env:ROS_SUPER_CLIENT -ErrorAction SilentlyContinue
Remove-Item Env:FASTRTPS_DEFAULT_PROFILES_FILE -ErrorAction SilentlyContinue
Remove-Item Env:CYCLONEDDS_URI -ErrorAction SilentlyContinue
$env:PATH = "$bridgeLib;$env:PATH"
$env:ROS_LOG_DIR = $rosLogDir
New-Item -ItemType Directory -Path $rosLogDir -Force | Out-Null

$arguments = @($entrypoint, '--project-root', $ProjectRoot)
if ($Headless) {
    $arguments += '--headless'
}
if ($BuildSceneOnly) {
    $arguments += '--build-scene-only'
}
if ($RebuildScene) {
    $arguments += '--rebuild-scene'
}
if ($SmokeTest) {
    $arguments += '--smoke-test'
}
if ($GripperSweep) {
    $arguments += '--gripper-sweep'
}
if ($Experiment3) {
    $arguments += '--experiment3'
}

Write-Host 'Launching Isaac Sim experiment entry point...'
Write-Host "  Project root: $ProjectRoot"
Write-Host "  Scene path:   $scenePath"
Write-Host "  Rebuild:      $($RebuildScene.IsPresent)"
Write-Host "  Build only:   $($BuildSceneOnly.IsPresent)"
Write-Host "  Arguments:    $($arguments -join ' ')"

$sceneWriteBefore = if (Test-Path -LiteralPath $scenePath -PathType Leaf) {
    (Get-Item -LiteralPath $scenePath).LastWriteTimeUtc
} else {
    [DateTime]::MinValue
}

& $pythonBat @arguments
$isaacExitCode = $LASTEXITCODE
if ($isaacExitCode -ne 0) {
    throw "Isaac Sim experiment exited with code $isaacExitCode"
}
if ($RebuildScene) {
    if (-not (Test-Path -LiteralPath $scenePath -PathType Leaf)) {
        throw "Isaac reported success but the scene does not exist: $scenePath"
    }
    $sceneItem = Get-Item -LiteralPath $scenePath
    if ($sceneItem.LastWriteTimeUtc -le $sceneWriteBefore) {
        throw "Isaac reported success but the USD timestamp did not change: $scenePath"
    }
    Write-Host 'USD rebuild verified:'
    $sceneItem | Select-Object FullName, LastWriteTime, Length | Format-List
}
