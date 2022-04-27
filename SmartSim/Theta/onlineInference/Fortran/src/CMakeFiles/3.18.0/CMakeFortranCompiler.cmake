set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/2.6.5/bin/ftn")
set(CMAKE_Fortran_COMPILER_ARG1 "")
set(CMAKE_Fortran_COMPILER_ID "GNU")
set(CMAKE_Fortran_COMPILER_VERSION "9.3.0")
set(CMAKE_Fortran_COMPILER_WRAPPER "CrayPrgEnv")
set(CMAKE_Fortran_PLATFORM_ID "")
set(CMAKE_Fortran_SIMULATE_ID "")
set(CMAKE_Fortran_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_Fortran_COMPILER_AR "/usr/bin/gcc-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_Fortran_COMPILER_RANLIB "/usr/bin/gcc-ranlib")
set(CMAKE_COMPILER_IS_GNUG77 1)
set(CMAKE_Fortran_COMPILER_LOADED 1)
set(CMAKE_Fortran_COMPILER_WORKS TRUE)
set(CMAKE_Fortran_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_Fortran_COMPILER_ENV_VAR "FC")

set(CMAKE_Fortran_COMPILER_SUPPORTS_F90 1)

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_Fortran_COMPILER_ID_RUN 1)
set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS f;F;fpp;FPP;f77;F77;f90;F90;for;For;FOR;f95;F95)
set(CMAKE_Fortran_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_Fortran_LINKER_PREFERENCE 20)
if(UNIX)
  set(CMAKE_Fortran_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_Fortran_OUTPUT_EXTENSION .obj)
endif()

# Save compiler ABI information.
set(CMAKE_Fortran_SIZEOF_DATA_PTR "8")
set(CMAKE_Fortran_COMPILER_ABI "")
set(CMAKE_Fortran_LIBRARY_ARCHITECTURE "")

if(CMAKE_Fortran_SIZEOF_DATA_PTR AND NOT CMAKE_SIZEOF_VOID_P)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_Fortran_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_Fortran_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_Fortran_COMPILER_ABI}")
endif()

if(CMAKE_Fortran_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()





set(CMAKE_Fortran_IMPLICIT_INCLUDE_DIRECTORIES "/opt/cray/pe/libsci/20.06.1/GNU/8.1/x86_64/include;/opt/cray/pe/mpt/7.7.14/gni/mpich-gnu/8.2/include;/opt/cray/rca/2.2.20-7.0.2.1_2.98__g8e3fb5b.ari/include;/opt/cray/alps/6.6.59-7.0.2.1_3.90__g872a8d62.ari/include;/opt/cray/xpmem/2.2.20-7.0.2.1_2.75__g87eb960.ari/include;/opt/cray/gni-headers/5.0.12.0-7.0.2.1_2.38__g3b1768f.ari/include;/opt/cray/pe/pmi/5.0.16/include;/opt/cray/ugni/6.0.14.0-7.0.2.1_3.83__ge78e5b0.ari/include;/opt/cray/udreg/2.3.2-7.0.2.1_2.59__g8175d3d.ari/include;/opt/cray/pe/atp/3.6.4/include;/opt/cray/wlm_detect/1.3.3-7.0.2.1_2.26__g7109084.ari/include;/opt/cray/krca/2.2.7-7.0.2.1_2.86__ge897ee1.ari/include;/opt/cray-hss-devel/9.0.0/include;/opt/gcc/9.3.0/snos/lib/gcc/x86_64-suse-linux/9.3.0/finclude;/opt/gcc/9.3.0/snos/lib/gcc/x86_64-suse-linux/9.3.0/include;/usr/local/include;/opt/gcc/9.3.0/snos/include;/opt/gcc/9.3.0/snos/lib/gcc/x86_64-suse-linux/9.3.0/include-fixed;/usr/include")
set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "AtpSigHandler;rca;darshan;lustreapi;z;sci_gnu_82_mpi;sci_gnu_82;mpich_gnu_82;mpichf90_gnu_82;gfortran;quadmath;pthread;gfortran;m;gcc_s;gcc;quadmath;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES "/opt/cray/pe/libsci/20.06.1/GNU/8.1/x86_64/lib;/opt/cray/dmapp/default/lib64;/opt/cray/pe/mpt/7.7.14/gni/mpich-gnu/8.2/lib;/soft/perftools/darshan/darshan-3.3.0/lib;/opt/cray/rca/2.2.20-7.0.2.1_2.98__g8e3fb5b.ari/lib64;/opt/cray/pe/atp/3.6.4/lib;/opt/gcc/9.3.0/snos/lib/gcc/x86_64-suse-linux/9.3.0;/opt/gcc/9.3.0/snos/lib64;/lib64;/usr/lib64;/opt/gcc/9.3.0/snos/lib")
set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
