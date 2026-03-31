param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$ElfPath,
    [string]$CubeProgrammerCli = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe",
    [string]$ExternalLoaderPath,
    [ValidateRange(0, 15)]
    [int]$AccessPort = 1,
    [ValidateSet("UR", "HOTPLUG", "NORMAL", "POWERDOWN", "HWRSTPULSE")]
    [string]$ConnectionMode = "NORMAL",
    [ValidateSet("SWrst", "HWrst", "Crst")]
    [string]$ResetMode = "SWrst",
    [switch]$NoVerify
)

# Default to the same ELF that CubeIDE produces for the Appli target.
if ([string]::IsNullOrWhiteSpace($ElfPath)) {
    $ElfPath = Join-Path $RepoRoot "firmware\stm32\n657\Appli\Debug\n657_Appli.elf"
}

# Use the external loader bundled with STM32CubeProgrammer unless the caller overrides it.
if ([string]::IsNullOrWhiteSpace($ExternalLoaderPath)) {
    $cubeProgrammerDir = Split-Path -Parent $CubeProgrammerCli
    $loaderCandidates = @(
        Join-Path $cubeProgrammerDir "ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr"
        "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\api\lib\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr"
    )

    $ExternalLoaderPath = $loaderCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not (Test-Path $CubeProgrammerCli)) {
    throw "STM32CubeProgrammer CLI not found at: $CubeProgrammerCli"
}

if (-not (Test-Path $ElfPath)) {
    throw "Build output not found at: $ElfPath"
}

if ([string]::IsNullOrWhiteSpace($ExternalLoaderPath) -or -not (Test-Path $ExternalLoaderPath)) {
    throw "External loader not found. Pass -ExternalLoaderPath or install STM32CubeProgrammer."
}

# Match the CubeIDE launch config: SWD and access port 1.
$args = @(
    "-c", "port=SWD", "ap=$AccessPort", "mode=$ConnectionMode", "reset=$ResetMode"
    "-el", $ExternalLoaderPath
    "-d", $ElfPath
)

if (-not $NoVerify) {
    $args += "-v"
}

$args += "-rst"

Write-Host "Flashing $ElfPath"
Write-Host "Using loader $ExternalLoaderPath"
Write-Host "Running $CubeProgrammerCli $($args -join ' ')"

& $CubeProgrammerCli @args
exit $LASTEXITCODE
