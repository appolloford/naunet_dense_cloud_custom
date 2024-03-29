if(MAKE_SHARED)
    set(NAUNETLIB naunet_shared)
elseif(MAKE_STATIC)
    set(NAUNETLIB naunet_static)
endif()

# check enabled languages
get_property(
    languages 
        GLOBAL 
            PROPERTY ENABLED_LANGUAGES
)

set(
    TESTTARGETS
    singlegrid
    renorm
)

# force cmake to have the same macro defined in naunet_macros.h
# use CapWords convention to use in test.py
set(USE_CUDA False)

if("CUDA" IN_LIST languages)
    set(USE_CUDA True)

    list(
        APPEND TESTTARGETS
            multiplegrid
            
    )
endif()

message(STATUS "Test targets               = ${TESTTARGETS}")

foreach(name ${TESTTARGETS})

    set(targetname test_${name})
    add_executable(${targetname} ${name}.cpp)

    set_target_properties(
        ${targetname}
            PROPERTIES
                OUTPUT_NAME ${targetname}.x
    )

    # for odeint test
    if(USE_BOOST)
        target_include_directories(
            ${targetname}
                PRIVATE 
                    ${Boost_INCLUDE_DIRS}
        )
    endif()

    target_link_libraries(
        ${targetname} 
            PRIVATE
                ${NAUNETLIB}
    )

    # test if gpu device is found when CUDA is enabled
    if(${NVIDIA_GPU_COUNT} OR NOT ("CUDA" IN_LIST languages))
        add_test(
            NAME ${targetname}
            COMMAND $<TARGET_FILE:${targetname}>
        )
    endif()

endforeach(name ${TESTTARGETS})

if(MAKE_PYTHON)
    configure_file(singlegrid.py.in singlegrid.py)

    if( ${NVIDIA_GPU_COUNT} OR NOT ("CUDA" IN_LIST languages) )
        add_test(
            NAME
                test_pymodule
            COMMAND
                ${CMAKE_COMMAND} -E env 
                    PYNAUNET_PATH=$<TARGET_FILE_DIR:${PYMODNAME}>
                    NAUNET_CONFIG_PATH=${CMAKE_SOURCE_DIR}
                ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/singlegrid.py
        )
    endif()
endif()

