#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#   include <OpenCL/cl_ext.h>
#   include <OpenCL/cl_gl.h>
#else
#   include <CL/cl.h>
#   include <CL/cl_ext.h>
#   include <CL/cl_gl.h>
#endif

#ifndef CL_UNORM_INT24
#define CL_UNORM_INT24 0x10DF
#endif

#ifndef CL_DEPTH_STENCIL
#define CL_DEPTH_STENCIL 0x10BE
#endif
