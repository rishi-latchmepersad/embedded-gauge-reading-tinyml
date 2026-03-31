param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$packRoot = 'C:\Users\rishi_latchmepersad\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.0'
$packPython = Join-Path $packRoot 'Utilities\windows\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'package_scalar_model_for_n6.py'

& $packPython $scriptPath @Args
