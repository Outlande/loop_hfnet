cmake_minimum_required(VERSION 2.8.3)
project(fundmental_matrix)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)	

set(HEADERS
		fundmental_matrix.h
		)

set(SOURCES
		fundmental_matrix.cpp
		)

add_library(fund_mat SHARED
    ${SOURCES}
)

target_include_directories(fund_mat PUBLIC
						   ${CMAKE_CURRENT_SOURCE_DIR}
							)