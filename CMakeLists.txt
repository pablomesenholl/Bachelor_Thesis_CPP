cmake_minimum_required(VERSION 3.12)
project(bachelor_thesis_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ----------------------------------------------------------------------------
#  ROOT
# ----------------------------------------------------------------------------
find_package(ROOT REQUIRED COMPONENTS RIO Hist Gpad Tree Physics)
include(${ROOT_USE_FILE})

# ----------------------------------------------------------------------------
#  Pythia8 (manual find)
# ----------------------------------------------------------------------------
# Make sure you pass -DPYTHIA8_DIR=/home/pablo/pythia8-install on the CMake command line
find_path( PYTHIA8_INCLUDE_DIR
        NAMES Pythia8/Pythia.h
        HINTS ${PYTHIA8_DIR}/include
)
find_library( PYTHIA8_LIBRARY
        NAMES pythia8
        HINTS ${PYTHIA8_DIR}/lib
)
if (NOT PYTHIA8_INCLUDE_DIR OR NOT PYTHIA8_LIBRARY)
    message(FATAL_ERROR "Could not find Pythia8 under ${PYTHIA8_DIR}")
endif()

# ----------------------------------------------------------------------------
#  EvtGen (config‐file find)
# ----------------------------------------------------------------------------
# Tell CMake to look in EvtGen’s own cmake directory for its config files:
#   /home/pablo/evtgen-install/share/EvtGen/cmake/EvtGenConfig.cmake
list(APPEND CMAKE_PREFIX_PATH "${EVTGEN_DIR}/share/EvtGen")
find_package(EvtGen REQUIRED)

# ----------------------------------------------------------------------------
#  Your executables
# ----------------------------------------------------------------------------
add_executable(RootInCLion main.cpp
)

add_executable(test_evtgen test_evtgen.cpp)

add_executable(spec_bckg_simulation specific_background_simulation.cpp)

add_executable(comb_bckg_simulation combinatorial_background.cpp)

#find external decay models
find_library(TAUOLA_FORTRAN TauolaFortran HINTS /home/pablo/tauola-install/lib)
find_library(TAUOLA_CXX TauolaCxxInterface HINTS /home/pablo/tauola-install/lib)
find_library(PHOTOS_PP Photospp HINTS /home/pablo/photos-install/lib)
find_library(PHOTOS_PP_HEPMC3 PhotosppHepMC3Event HINTS /home/pablo/photos-install/lib)

target_include_directories(test_evtgen PRIVATE
        ${PYTHIA8_INCLUDE_DIR}
        /home/pablo/evtgen-new-install/include
)

target_link_directories(test_evtgen PRIVATE
        /home/pablo/evtgen-new-install/lib
)

target_link_libraries(test_evtgen
        ${PYTHIA8_LIBRARY}
        EvtGen
        EvtGenExternal
        ${TAUOLA_FORTRAN}
        ${TAUOLA_CXX}
        ${PHOTOS_PP}
        ${PHOTOS_PP_HEPMC3}
)


# include dirs (ROOT targets bring THEIR include dirs automatically)
target_include_directories(RootInCLion PRIVATE
        ${PYTHIA8_INCLUDE_DIR}
)

# link everything
target_link_libraries(RootInCLion PRIVATE
        ROOT::RIO
        ROOT::Hist
        ROOT::Gpad
        ROOT::Tree
        ROOT::Physics
        ${PYTHIA8_LIBRARY}

        EvtGen::EvtGen
        EvtGen::EvtGenExternal
)

target_include_directories(spec_bckg_simulation PRIVATE
        ${PYTHIA8_INCLUDE_DIR}
)
target_link_libraries(spec_bckg_simulation PRIVATE
        ROOT::RIO
        ROOT::Hist
        ROOT::Gpad
        ROOT::Tree
        ROOT::Physics
        ${PYTHIA8_LIBRARY}

        EvtGen::EvtGen
        EvtGen::EvtGenExternal
)

target_include_directories(comb_bckg_simulation PRIVATE
        ${PYTHIA8_INCLUDE_DIR}
)

target_link_libraries(comb_bckg_simulation PRIVATE
        ROOT::RIO
        ROOT::Hist
        ROOT::Gpad
        ROOT::Tree
        ROOT::Physics
        ${PYTHIA8_LIBRARY}

        EvtGen::EvtGen
        EvtGen::EvtGenExternal
)