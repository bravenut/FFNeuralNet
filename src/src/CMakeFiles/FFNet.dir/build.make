# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/max/programming/cpp/ffneuralnet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/max/programming/cpp/ffneuralnet/src

# Include any dependencies generated for this target.
include src/CMakeFiles/FFNet.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/FFNet.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/FFNet.dir/flags.make

src/CMakeFiles/FFNet.dir/main.cpp.o: src/CMakeFiles/FFNet.dir/flags.make
src/CMakeFiles/FFNet.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/max/programming/cpp/ffneuralnet/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/FFNet.dir/main.cpp.o"
	cd /home/max/programming/cpp/ffneuralnet/src/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FFNet.dir/main.cpp.o -c /home/max/programming/cpp/ffneuralnet/src/main.cpp

src/CMakeFiles/FFNet.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FFNet.dir/main.cpp.i"
	cd /home/max/programming/cpp/ffneuralnet/src/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/max/programming/cpp/ffneuralnet/src/main.cpp > CMakeFiles/FFNet.dir/main.cpp.i

src/CMakeFiles/FFNet.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FFNet.dir/main.cpp.s"
	cd /home/max/programming/cpp/ffneuralnet/src/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/max/programming/cpp/ffneuralnet/src/main.cpp -o CMakeFiles/FFNet.dir/main.cpp.s

# Object files for target FFNet
FFNet_OBJECTS = \
"CMakeFiles/FFNet.dir/main.cpp.o"

# External object files for target FFNet
FFNet_EXTERNAL_OBJECTS =

src/FFNet: src/CMakeFiles/FFNet.dir/main.cpp.o
src/FFNet: src/CMakeFiles/FFNet.dir/build.make
src/FFNet: /usr/lib/x86_64-linux-gnu/libblas.so
src/FFNet: /usr/lib/x86_64-linux-gnu/liblapack.so
src/FFNet: /usr/lib/x86_64-linux-gnu/libblas.so
src/FFNet: /usr/lib/x86_64-linux-gnu/liblapack.so
src/FFNet: src/CMakeFiles/FFNet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/max/programming/cpp/ffneuralnet/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FFNet"
	cd /home/max/programming/cpp/ffneuralnet/src/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FFNet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/FFNet.dir/build: src/FFNet

.PHONY : src/CMakeFiles/FFNet.dir/build

src/CMakeFiles/FFNet.dir/clean:
	cd /home/max/programming/cpp/ffneuralnet/src/src && $(CMAKE_COMMAND) -P CMakeFiles/FFNet.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/FFNet.dir/clean

src/CMakeFiles/FFNet.dir/depend:
	cd /home/max/programming/cpp/ffneuralnet/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/max/programming/cpp/ffneuralnet /home/max/programming/cpp/ffneuralnet/src /home/max/programming/cpp/ffneuralnet/src /home/max/programming/cpp/ffneuralnet/src/src /home/max/programming/cpp/ffneuralnet/src/src/CMakeFiles/FFNet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/FFNet.dir/depend

