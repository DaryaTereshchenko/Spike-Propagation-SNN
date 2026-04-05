add_executable(test_formats test_formats.cpp)
target_link_libraries(test_formats PRIVATE spike_lib Catch2::Catch2WithMain)

add_executable(test_topology test_topology.cpp)
target_link_libraries(test_topology PRIVATE spike_lib Catch2::Catch2WithMain)

add_executable(test_lif test_lif.cpp)
target_link_libraries(test_lif PRIVATE spike_lib Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(test_formats)
catch_discover_tests(test_topology)
catch_discover_tests(test_lif)
