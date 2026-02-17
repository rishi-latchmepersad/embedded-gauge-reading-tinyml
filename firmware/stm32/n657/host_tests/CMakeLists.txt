cmake_minimum_required(VERSION 3.20)

project(host_unit_tests C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Adjust this if your Unity location differs
set(UNITY_DIR "../Appli/Middlewares/Third_Party/Unity/src")

add_executable(unit_tests
    "${UNITY_DIR}/unity.c"
    "../Appli/Src/sd_spi_protocol.c"
	"../Appli/Src/sd_debug_log_core.c"
    "test_runner.c"
    "test_sanity.c"
	"test_sd_spi_protocol.c"
	"test_sd_debug_log_core.c"
)


target_include_directories(unit_tests PRIVATE
    "${UNITY_DIR}"
    "../Appli/Inc"
)
