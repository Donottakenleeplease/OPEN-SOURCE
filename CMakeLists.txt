cmake_minimum_required( VERSION 2.8 )
project( http_conn )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# opencv 
find_package( OpenCV REQUIRED )
find_package( Threads )

include_directories( 
    ${OpenCV_INCLUDE_DIRS}
)

include_directories( ${PROJECT_SOURCE_DIR}/include )


add_executable( http_conn src/mongoose.c src/test.cpp )

target_link_libraries( http_conn   
    ${OpenCV_LIBS} 
    ${CMAKE_THREAD_LIBS_INIT}
    )
