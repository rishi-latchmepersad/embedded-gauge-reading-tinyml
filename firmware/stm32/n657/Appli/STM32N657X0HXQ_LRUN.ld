/*
******************************************************************************
**
** @file        : STM32N6XX_LRUN.ld
**
** @author      : GPM Application Team
**
** @brief       : Linker script for STM32N6XX Device from STM32N6 series
**                      2048 KBytes RAM
**
**                Set heap size, stack size and stack location according
**                to application requirements.
**
**                Set memory bank area and size if external memory is used
**
**  Target      : STMicroelectronics STM32
**
**  Distribution: The file is distributed as is, without any warranty
**                of any kind.
**
******************************************************************************
** @attention
**
** Copyright (c) 2025 STMicroelectronics.
** All rights reserved.
**
** This software is licensed under terms that can be found in the LICENSE file
** in the root directory of this software component.
** If no LICENSE file comes with this software, it is provided AS-IS.
**
******************************************************************************
*/

/* Entry Point */
ENTRY(Reset_Handler)

/* Highest address of the user mode stack */
_estack = ORIGIN(RAM) + LENGTH(RAM); /* end of "RAM" Ram type memory */
_sstack = _estack - _Min_Stack_Size;

_Min_Heap_Size = 0x2000; /* required amount of heap (8KB — newlib printf needs malloc) */
_Min_Stack_Size = 0x8000; /* required amount of stack (32KB -- main() init path needs >16KB for HAL_RCC_OscConfig + HAL_RCC_ClockConfig) */

/* Memories definition */
MEMORY
{
  ROM    (rx)     : ORIGIN = 0x34000400,   LENGTH = 600K
  /* Shrink RAM to 443K (AXISRAM1 only) so BSS cannot grow into AXISRAM2
     where the OBB model input buffer lives at 0x34110000.
     ROM grew to 600K to accommodate DS-CNN v4 heatmap CD .rodata.  */
  RAM    (xrw)    : ORIGIN = 0x34099400,   LENGTH = 423K
  /* Scalar model blob at 0x70200000 (matches FLASH_SCALAR in flash_boot.bat). */
  EXTRAM (rw)           : ORIGIN = 0x70200000,   LENGTH = 3200K
  /* Rectifier model blob at 0x70600000 (matches FLASH_RECTIFIER in flash_boot.bat).
   * The rectifier pool symbol must live here so the NPU reads weights from the
   * correct flash region. */
  EXTRAM_RECTIFIER (rw) : ORIGIN = 0x70600000,   LENGTH = 256K
  EXTRAM_OBB (rw)       : ORIGIN = 0x71400000,   LENGTH = 4096K
  /* HyperRAM scratch for QARepVGG-Pro (1 MB activation overflow).
   * The N657 Nucleo has no physical HyperRAM; this section exists
   * for linker resolution only.  NPU access to 0x90000000 will
   * bus-fault at runtime. */
  XSPI1_RAM (rw)         : ORIGIN = 0x90000000,   LENGTH = 32M
  /* Source-crop-box model blob at 0x70B00000 (matches FLASH_SOURCE_CROP_BOX in flash_boot.bat). */
  EXTRAM_SOURCE_CROP_BOX (rw) : ORIGIN = 0x70B00000,   LENGTH = 512K
  /* Non-secure alias A/B test — DCMIPP with MSEC=1 writes to 0x24xxxxxx.
   * Placed at top of AXISRAM1 non-secure alias, away from code/data. */
  RAM_NC (xrw)    : ORIGIN = 0x24160000,   LENGTH = 512K
  /* AXISRAM6 — polar_luma (100K, CD stage only) is overlaid on the bottom
   * of the OBB activation region (0x34350000).  Both use the same memory but
   * at different times: OBB runs first (writes scratch here), then the CD
   * stage overrwrites it with fresh polar_luma before the polar vote.
   * polar_luma was moved here from .tip_focus_activations because the 4K tail
   * overflowed past TIP_FOCUS_RAM (0x342E0000) into AXISRAM5 where the CD
   * NPU activation metadata lives, corrupting ATON state across runs. */
  NPU_SRAM6_POLAR (xrw) : ORIGIN = 0x34350000, LENGTH = 100K
  NPU_SRAM6 (xrw)       : ORIGIN = 0x3439C000, LENGTH = 144K
   /* Tip-focus geometry model: weights ~2 MiB, placed in unused xSPI2 flash
    * at 0x70400000 (matches model's c_info.json offset_start).
    * Flash_boot.bat programs tip_focus_weights.raw here. */
   TIP_FOCUS_WEIGHTS (rw) : ORIGIN = 0x70400000,   LENGTH = 4096K
  /* Tip-focus geometry model: activations ~632 KiB scratch buffer placed in
   * unused AXISRAM3+4 (contiguous, 0x34200000, 896K).  AXISRAM3/4 are not
   * used by any NPU model or application code. */
  TIP_FOCUS_RAM (xrw) : ORIGIN = 0x34200000,   LENGTH = 896K
}

PHDRS
{
  text PT_LOAD FLAGS(5);
  data PT_LOAD FLAGS(6);
}

/* Sections */
SECTIONS
{
  /* The startup code into "Rom" Rom type memory */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Startup code */
    . = ALIGN(4);
  } >ROM :text

  /* The program code and other data into "ROM" Rom type memory */
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)
    *(.RamFunc)        /* .RamFunc sections */
    *(.RamFunc*)       /* .RamFunc* sections */

    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >ROM :text

   /* Tip-focus weights: placed BEFORE .rodata so the linker routes
    * .rodata.g_network_weights_array here instead of into ROM.
    * NOLOAD — the data is NOT included in the .elf binary.  The raw
    * weight blob is extracted from the .o file in a post-compile step
    * and must be flashed to 0x70400000 separately. */
  .tip_focus_weights (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.rodata.g_network_weights_array))
    . = ALIGN(32);
  } >TIP_FOCUS_WEIGHTS

  /* Constant data into "ROM" Rom type memory */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)         /* .rodata sections (constants, strings, etc.) */
    *(.rodata*)        /* .rodata* sections (constants, strings, etc.) */
    . = ALIGN(4);
  } >ROM :text

  /* Scalar model pool: placed at the scalar flash region (0x70200000). */
  .xspi2_pool (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.xspi2_pool))
    . = ALIGN(32);
  } >EXTRAM

  /* Rectifier model pool: placed at the rectifier flash region (0x70600000).
   * Separating this from .xspi2_pool ensures _mem_pool_xSPI2_mobilenetv2_rectifier_*
   * resolves to the correct flash address so the NPU reads weights from there. */
  .xspi2_rectifier_pool (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.xspi2_rectifier_pool))
    . = ALIGN(32);
  } >EXTRAM_RECTIFIER

  .xspi2_obb_pool (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.xspi2_obb_pool))
    . = ALIGN(32);
  } >EXTRAM_OBB

  /* QARepVGG-Pro activation overflow pool (HyperRAM, 0x90000000).
   * The N657 Nucleo has no physical HyperRAM; this section exists
   * for linker symbol resolution only.  NPU access will bus-fault. */
  .xspi1_obb_pool (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.xspi1_obb_pool))
    . = ALIGN(32);
  } >XSPI1_RAM

  /* Source-crop-box model pool: placed at the source-crop-box flash region (0x70B00000).
   * Separating this ensures _mem_pool_xSPI2_mobilenetv2_source_crop_box_*
   * resolves to the correct flash address so the NPU reads weights from there. */
  .xspi2_source_crop_box_pool (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.xspi2_source_crop_box_pool))
    . = ALIGN(32);
  } >EXTRAM_SOURCE_CROP_BOX

  /* Tip-focus geometry model pool: placed at the tip-focus weights flash
    * region (0x70400000).  Matches TIP_FOCUS_WEIGHTS in the MEMORY block and
    * network_atonbuf.xSPI2.raw in flash_boot.bat. */
  .xspi2_tip_focus_pool (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.xspi2_tip_focus_pool))
    . = ALIGN(32);
  } >TIP_FOCUS_WEIGHTS

  .ARM.extab   : {
    . = ALIGN(4);
    *(.ARM.extab* .gnu.linkonce.armextab.*)
    . = ALIGN(4);
  } >ROM :text

  .ARM : {
    . = ALIGN(4);
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
    . = ALIGN(4);
  } >ROM :text

  .preinit_array     :
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
    . = ALIGN(4);
  } >ROM :text

  .init_array :
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
    . = ALIGN(4);
  } >ROM :text

  .fini_array :
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
    . = ALIGN(4);
  } >ROM :text

  /* Used by the startup to initialize data */
  _sidata = LOADADDR(.data);

  /* Initialized data sections into "RAM" Ram type memory */
  .data :
  {
    . = ALIGN(4);
    _sdata = .;        /* create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data*)          /* .data* sections */

    . = ALIGN(4);
    _edata = .;        /* define a global symbol at data end */

  } >RAM AT> ROM :data

  .noncacheable (NOLOAD) :
  {
    . = ALIGN(32);
    __snoncacheable = .;/* create symbol for start of section */
    KEEP(*(.noncacheable))
    KEEP(*(.noncacheable*))
    . = ALIGN(32);
    __enoncacheable = .;  /* create symbol for end of section */
  } > RAM_NC

  .gnu.sgstubs :
  {
    _sNSCVeneer = .;
    . = ALIGN(4);
    *(.gnu.sgstubs*)   /* Secure Gateway stubs */
    . = ALIGN(4);
  } >ROM :text
  _eNSCVeneer = .;

  /* Uninitialized data section into "RAM" Ram type memory */
  . = ALIGN(4);
  .bss :
  {
    /* This is used by the startup in order to initialize the .bss section */
    _sbss = .;         /* define a global symbol at bss start */
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;         /* define a global symbol at bss end */
    __bss_end__ = _ebss;
  } >RAM :data

  /* Tip-focus / camera scratch stays out of the OBB reloc runtime window.
   * The OBB vpool starts at 0x34100000, so this section must not live in the
   * old 0x34200000 AXISRAM3+4 placement. Keeping the shared camera snapshot and
   * center-detector scratch in RAM_NC avoids colliding with the reloc image. */
  .tip_focus_activations (NOLOAD) :
  {
    . = ALIGN(32);
    __stip_focus_activations = .;
    KEEP(*(.tip_focus_activations))
    KEEP(*(.tip_focus_activations*))
    . = ALIGN(32);
  } >RAM_NC

  /* AXISRAM6 section — AI thread stack lives here, after polar_luma (below)
   * and the OBB activation region (which both share the same 304K below). */
  .npusram6 (NOLOAD) :
  {
    . = ALIGN(32);
    __snpusram6 = .;
    KEEP(*(.npusram6))
    KEEP(*(.npusram6*))
    . = ALIGN(32);
    __enpusram6 = .;
  } >NPU_SRAM6

  /* polar_luma — temporally overlaid on OBB activation region at 0x34350000.
   * OBB writes scratch here during its stage, then the CD stage overwrites
   * with fresh polar_luma before the polar vote.  Never in use at the same
   * time as the OBB NPU inference. */
  .npusram6_polar (NOLOAD) :
  {
    . = ALIGN(32);
    KEEP(*(.npusram6_polar))
    KEEP(*(.npusram6_polar*))
    . = ALIGN(32);
  } >NPU_SRAM6_POLAR

  /* User_heap_stack section, used to check that there is enough "RAM" Ram  type memory left */
  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >RAM :data

  /* Remove information from the compiler libraries */
  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
