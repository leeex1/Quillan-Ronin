# CMake generated Testfile for 
# Source directory: C:/02_QUILLAN/09 - Projects/projects/jdxx-chains
# Build directory: C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test("chain_smoke" "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/build/Debug/chain_smoke.exe")
  set_tests_properties("chain_smoke" PROPERTIES  _BACKTRACE_TRIPLES "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;103;add_test;C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test("chain_smoke" "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/build/Release/chain_smoke.exe")
  set_tests_properties("chain_smoke" PROPERTIES  _BACKTRACE_TRIPLES "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;103;add_test;C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test("chain_smoke" "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/build/MinSizeRel/chain_smoke.exe")
  set_tests_properties("chain_smoke" PROPERTIES  _BACKTRACE_TRIPLES "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;103;add_test;C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test("chain_smoke" "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/build/RelWithDebInfo/chain_smoke.exe")
  set_tests_properties("chain_smoke" PROPERTIES  _BACKTRACE_TRIPLES "C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;103;add_test;C:/02_QUILLAN/09 - Projects/projects/jdxx-chains/CMakeLists.txt;0;")
else()
  add_test("chain_smoke" NOT_AVAILABLE)
endif()
subdirs("_deps/juce-build")
