#include "opencl_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

/**************************************************************************************************
** OpenCL helper routines
**************************************************************************************************/

#define GEN_CASE(errcod)	case errcod: return #errcod;

const char* getErrorName(cl_int errcode)
{
	switch (errcode) {
		GEN_CASE(CL_SUCCESS)
			GEN_CASE(CL_DEVICE_NOT_FOUND)
			GEN_CASE(CL_DEVICE_NOT_AVAILABLE)
			GEN_CASE(CL_COMPILER_NOT_AVAILABLE)
			GEN_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
			GEN_CASE(CL_OUT_OF_RESOURCES)
			GEN_CASE(CL_OUT_OF_HOST_MEMORY)
			GEN_CASE(CL_PROFILING_INFO_NOT_AVAILABLE)
			GEN_CASE(CL_MEM_COPY_OVERLAP)
			GEN_CASE(CL_IMAGE_FORMAT_MISMATCH)
			GEN_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED)
			GEN_CASE(CL_BUILD_PROGRAM_FAILURE)
			GEN_CASE(CL_MAP_FAILURE)
			GEN_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET)
			GEN_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
			GEN_CASE(CL_INVALID_VALUE)
			GEN_CASE(CL_INVALID_DEVICE_TYPE)
			GEN_CASE(CL_INVALID_PLATFORM)
			GEN_CASE(CL_INVALID_DEVICE)
			GEN_CASE(CL_INVALID_CONTEXT)
			GEN_CASE(CL_INVALID_QUEUE_PROPERTIES)
			GEN_CASE(CL_INVALID_COMMAND_QUEUE)
			GEN_CASE(CL_INVALID_HOST_PTR)
			GEN_CASE(CL_INVALID_MEM_OBJECT)
			GEN_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
			GEN_CASE(CL_INVALID_IMAGE_SIZE)
			GEN_CASE(CL_INVALID_SAMPLER)
			GEN_CASE(CL_INVALID_BINARY)
			GEN_CASE(CL_INVALID_BUILD_OPTIONS)
			GEN_CASE(CL_INVALID_PROGRAM)
			GEN_CASE(CL_INVALID_PROGRAM_EXECUTABLE)
			GEN_CASE(CL_INVALID_KERNEL_NAME)
			GEN_CASE(CL_INVALID_KERNEL_DEFINITION)
			GEN_CASE(CL_INVALID_KERNEL)
			GEN_CASE(CL_INVALID_ARG_INDEX)
			GEN_CASE(CL_INVALID_ARG_VALUE)
			GEN_CASE(CL_INVALID_ARG_SIZE)
			GEN_CASE(CL_INVALID_KERNEL_ARGS)
			GEN_CASE(CL_INVALID_WORK_DIMENSION)
			GEN_CASE(CL_INVALID_WORK_GROUP_SIZE)
			GEN_CASE(CL_INVALID_WORK_ITEM_SIZE)
			GEN_CASE(CL_INVALID_GLOBAL_OFFSET)
			GEN_CASE(CL_INVALID_EVENT_WAIT_LIST)
			GEN_CASE(CL_INVALID_EVENT)
			GEN_CASE(CL_INVALID_OPERATION)
			GEN_CASE(CL_INVALID_GL_OBJECT)
			GEN_CASE(CL_INVALID_BUFFER_SIZE)
			GEN_CASE(CL_INVALID_MIP_LEVEL)
			GEN_CASE(CL_INVALID_GLOBAL_WORK_SIZE)
			GEN_CASE(CL_INVALID_PROPERTY)
	default: return "<Unknown>";
	}
}

void checkForError(cl_int status, const char *filename, int line)
{
	if (status != CL_SUCCESS) {
		fprintf(stderr, "Error %s (%d) at %s:%d\n", getErrorName(status), status, filename, line);
		exit(status);
	}
}

char* readFileIntoString(const char* filename)
{
	char *buffer = NULL;
	long len;
	FILE *f = fopen(filename, "rb");
	assert(f != NULL);
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	buffer = new char[len + 1];
	size_t res = fread(buffer, len, 1, f);
	assert(res == 1);
	buffer[len] = '\0';
	fclose(f);

	return buffer;
}

void checkBuildProgram(cl_int status, cl_program program, cl_device_id device)
{
	if (status != CL_SUCCESS) {
		size_t len;
		CHECK_FOR_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len));
		char *buf = new char[len];
		CHECK_FOR_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buf, NULL));
		fprintf(stderr, "Error while building:\n%s\n", buf);
		fprintf(stderr, "Quitting with OpenCL error %s (%d)\n", getErrorName(status), status);
		delete[] buf;
		exit(status);
	}
}