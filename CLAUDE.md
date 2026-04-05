# embedded-gauge-reading-tinyml — Project Notes for Claude

## Flashing firmware to the STM32N657 (boot from flash)

### Prerequisites
- Board in NUCLEO dev/programming mode (BOOT1 = position 2-3) for initial flashing
- ST-Link connected via USB
- STM32CubeProgrammer v2.21+ installed at `C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\`

### Tools
- Signing: `STM32_SigningTool_CLI.exe` (in CubeProgrammer `bin/`)
- Flashing: `STM32_Programmer_CLI.exe` (in CubeProgrammer `bin/`)
- External loader: `MX25UM51245G_STM32N6570-NUCLEO.stldr` (in CubeProgrammer `bin/ExternalLoader/`)
- FSBL source: `firmware/stm32/n657/FSBL/` (STM32CubeIDE project, builds to `FSBL/Debug/n657_FSBL.bin`)

### Boot mode jumper positions (NUCLEO-N657X0-Q)
- **Development / programming mode**: BOOT1 = 2-3 (BOOT0 doesn't matter)
- **Boot from external flash**: BOOT0 = 1-2, BOOT1 = 1-2

### Steps

1. **Build FSBL** in STM32CubeIDE → produces `firmware/stm32/n657/FSBL/Debug/n657_FSBL.bin`

2. **Build App** in STM32CubeIDE → produces `firmware/stm32/n657/Appli/Debug/n657_Appli.bin`

3. **Run `flash_boot.bat`** from `firmware/stm32/n657/` with board in development mode.
   - `FLASH_APP=1` is the default — flashes both FSBL and app
   - Set `FLASH_MODEL=1` to also flash the neural network blob (default is 0)
   - The script signs and flashes everything automatically

4. **Set flash-boot mode** (BOOT0 = 1-2, BOOT1 = 1-2) and **full power-cycle** the board
   - Must be a full power-off, not just a reset, for the BOOT pins to be re-read

### Signing details
- FSBL is signed with: `-nk -of 0x80000000 -t fsbl -hv 2.3 -align`
- App is signed with: `-nk -of 0x80000000 -t ssbl -hv 2.3 -align`
- **`-align` is required for CubeProgrammer v2.21+** — without it the ROM rejects the header silently and the board does nothing on flash boot
- `-nk` = no signing key (no secure boot enforcement)
- `-of 0x80000000` is the correct origin address for this boot flow (not the flash address)

### Memory layout (xSPI2 Flash)
- `0x70000000` — Signed FSBL image (STM2 header + binary)
- `0x70100000` — Signed application image
- `0x70100400` — App linker origin (runtime entry point, 0x400 offset past the signed header)

### App memory layout (LRUN — load-and-run)
- App links to run from AXISRAM1: ROM origin `0x34000400`, RAM origin `0x34080000`
- Stack top: `0x34200000` (end of 1536KB RAM region)
- FSBL copies 512KB from `0x70100400` → `0x34000400`, then jumps to Reset_Handler

### FSBL xSPI2 init sequence (MX25UM51245G OctoSPI)
The FSBL re-initialises XSPI2 from scratch (SystemInit resets it). Sequence:
1. `HAL_XSPI_Init` in 1-line SPI mode at 50 MHz (IC3 from PLL1 div/24)
2. `HAL_XSPIM_Config` — port 2, NCS1
3. `WriteEnable` (SPI cmd 0x06)
4. `WriteCfg2Register` addr=0 val=0x01 (SOPI bit) to enter OctoSPI STR mode
5. Set READ_CFG: cmd 0xEC13 (16-bit), 8 lines, 20 dummy cycles (chip default after reset)
6. Set WRITE_CFG: cmd 0x12ED (dummy — read-only for boot)
7. `HAL_XSPI_MemoryMapped` → flash visible at `0x70000000`

### FSBL jump sequence
Before jumping to the app the FSBL:
- Disables SysTick (`SysTick->CTRL = 0`) — critical: FSBL SysTick fires into uninitialized app RAM otherwise
- Sets `VTOR = 0x34000400`
- Sets `MSP = app_sp` and clears `MSPLIM = 0`
- Restores PRIMASK to pre-disable state (so app gets interrupts)

### OTP fuses
- **VDDIO3_HSLV fuse is NOT required** on the NUCLEO-N657X0-Q at 3.3V I/O
- Do not burn OTP fuses unless you have a specific hardware reason (they are permanent)

### Neural network model
- Model blob: `st_ai_output/atonbuf.xSPI2.raw`
- Flashed at `0x70200000` (after the app region) — adjust `FLASH_MODEL` address if needed
- WSL must be restarted before running any Python/ML scripts

### UART debug output
- LPUART1 at 115200 8N1, TX=PE5, RX=PE6
- Both FSBL and App output `printf` via `__io_putchar` → `HAL_UART_Transmit(&hlpuart1, ...)`
- Connect USB-UART adapter: PE5 → adapter RX, GND → GND

### Confirmed working boot chain
ROM → FSBL (xSPI2 OctoSPI init + LRUN copy) → Full App (ThreadX + camera thread)
- FSBL: `FSBL_LED_ONLY_SMOKE_TEST 0`
- App: `APP_LED_ONLY_SMOKE_TEST 0` (full app with ThreadX, SystemIsolation, camera)
- Expected final UART output: `[AR] Calling App_ThreadX_Start().`
