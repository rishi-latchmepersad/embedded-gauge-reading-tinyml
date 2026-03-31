param(
    [string]$ElfPath = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "firmware\stm32\n657\FSBL\Debug\n657_FSBL.elf"),
    [string]$CubeProgrammerCli = "C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe",
    [string]$GoAddress = "0x34180400"
)

# The FSBL image is linked into RAM, so we download it and immediately jump to
# its reset handler address.
if (-not (Test-Path $CubeProgrammerCli)) {
    throw "STM32CubeProgrammer CLI not found at: $CubeProgrammerCli"
}

if (-not (Test-Path $ElfPath)) {
    throw "FSBL build output not found at: $ElfPath"
}

$args = @(
    "-c", "port=SWD", "mode=NORMAL"
    "-d", $ElfPath
    "-g", $GoAddress
)

Write-Host "Launching FSBL blink demo from $ElfPath"
Write-Host "Running $CubeProgrammerCli $($args -join ' ')"

& $CubeProgrammerCli @args
exit $LASTEXITCODE
