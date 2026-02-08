$CLI="C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"
$LOADER="C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr"

& $CLI -c port=SWD mode=HOTPLUG -el $LOADER -hardRst -w "D:\Projects\embedded-gauge-reading-tinyml\firmware\stm32\N657X0_bringup\Appli\Debug\Project-trusted.bin" 0x70100000
& $CLI -c port=SWD mode=HOTPLUG -el $LOADER -hardRst -w "C:\Users\rishi_latchmepersad\STM32Cube\Repository\STM32Cube_FW_N6_V1.3.0\Projects\NUCLEO-N657X0-Q\Templates\Template_FSBL_LRUN\STM32CubeIDE\Boot\Debug\FSBL-trusted.bin"   0x70000000