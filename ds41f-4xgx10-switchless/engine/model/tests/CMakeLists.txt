# The model's host-only tests (no CUDA): built with the engine, runnable on a box without a GPU.
# page_tests: the shared KV pool's host side (pages.h/pages.cpp), docs/shared-kv.md.
add_executable(page_tests page_tests.cpp ../pages.cpp)
target_include_directories(page_tests PRIVATE ${CMAKE_SOURCE_DIR})
