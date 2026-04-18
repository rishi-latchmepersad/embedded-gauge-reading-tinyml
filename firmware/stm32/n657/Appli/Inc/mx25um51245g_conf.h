#ifndef MX25UM51245G_CONF_H
#define MX25UM51245G_CONF_H

/*
 * Project-local override for the MX25UM51245G BSP component.
 *
 * The wrapper source includes this file before the vendor driver so the build
 * uses the same dummy-cycle constants everywhere without touching pack files.
 */

#define DUMMY_CYCLES_READ            8U
#define DUMMY_CYCLES_READ_OCTAL      6U
#define DUMMY_CYCLES_READ_OCTAL_DTR  6U
#define DUMMY_CYCLES_REG_OCTAL       4U
#define DUMMY_CYCLES_REG_OCTAL_DTR   5U

#endif /* MX25UM51245G_CONF_H */
