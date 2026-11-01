# The tray app of the Windows package (scripts/windows/package.ps1). Windows and the Universal CRT are all it needs:
# the C++ runtime is linked statically and the UCRT comes from the system, as for q27.dll (engine/CMakeLists.txt).
cmake_minimum_required(VERSION 3.25)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
project(qwen3-27b-tray LANGUAGES CXX RC)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qwen3-27b-tray WIN32 tray.cc tray.rc)
target_compile_options(qwen3-27b-tray PRIVATE /utf-8 /W4 /permissive- /D_CRT_SECURE_NO_WARNINGS)
target_link_libraries(qwen3-27b-tray PRIVATE winhttp shell32 ole32 user32 gdi32 advapi32)
target_link_options(qwen3-27b-tray PRIVATE /MANIFEST:NO /NODEFAULTLIB:libucrt.lib /DEFAULTLIB:ucrt.lib)
