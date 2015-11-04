#ifndef _OPENCL_HELPERS_H

#define _OPENCL_HELPERS_H

#ifdef __APPLE__
#	include <OpenCL/cl.h>
#else
#	include <CL/cl.h>
#endif

const char* getErrorName(cl_int errcode);
void checkForError(cl_int status, const char *filename, int line);
#define CHECK_FOR_ERROR(errcode)	checkForError(errcode, __FILE__, __LINE__)
char* readFileIntoString(const char* filename);
void checkBuildProgram(cl_int status, cl_program program, cl_device_id device);

#endif
