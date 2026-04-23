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
_Min_Stack_Size = 0x4000; /* required amount of stack (16KB — printf needs headroom) */

/* Memories definition */
MEMORY
{
  ROM    (rx)     : ORIGIN = 0x34000400,   LENGTH = 511K
  RAM    (xrw)    : ORIGIN = 0x34080000,   LENGTH = 1536K
  /* Scalar model blob at 0x70200000 (matches FLASH_SCALAR in flash_boot.bat). */
  EXTRAM (rw)           : ORIGIN = 0x70200000,   LENGTH = 3200K
  /* Rectifier model blob at 0x70600000 (matches FLASH_RECTIFIER in flash_boot.bat).
   * The rectifier pool symbol must live here so the NPU reads weights from the
   * correct flash region. */
  EXTRAM_RECTIFIER (rw) : ORIGIN = 0x70600000,   LENGTH = 256K
  EXTRAM_OBB (rw)       : ORIGIN = 0x70700000,   LENGTH = 4096K
  /* Non-secure alias A/B test — DCMIPP with MSEC=1 writes to 0x24xxxxxx.
   * Placed at top of AXISRAM1 non-secure alias, away from code/data. */
  RAM_NC (xrw)    : ORIGIN = 0x24160000,   LENGTH = 128K
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
