//go:build cgo && opencl

package deepcfr

/*
#cgo pkg-config: OpenCL
#cgo !windows LDFLAGS: -ldl
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <CL/opencl.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef int CLBlastLayout;
typedef int CLBlastTranspose;
typedef int CLBlastStatusCode;

typedef CLBlastStatusCode (*clblast_sgemm_fn)(
	CLBlastLayout layout,
	CLBlastTranspose a_transpose,
	CLBlastTranspose b_transpose,
	size_t m, size_t n, size_t k,
	float alpha,
	const cl_mem a_buffer, size_t a_offset, size_t a_ld,
	const cl_mem b_buffer, size_t b_offset, size_t b_ld,
	float beta,
	cl_mem c_buffer, size_t c_offset, size_t c_ld,
	cl_command_queue* queue,
	cl_event* event
);

#define CLBLAST_LAYOUT_ROW_MAJOR 101
#define CLBLAST_TRANSPOSE_NO 111
#define CLBLAST_TRANSPOSE_YES 112
#define CLBLAST_SUCCESS 0

typedef struct {
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	cl_program program;
	cl_kernel add_bias_relu_kernel;
	cl_kernel add_bias_kernel;
	cl_kernel relu_backward_kernel;
	void *clblast_lib;
	clblast_sgemm_fn clblast_sgemm;
	char clblast_lib_name[64];
	char device_name[256];
} opencl_trainer;

static void set_error(char *buffer, size_t buffer_len, const char *message) {
	if (buffer == NULL || buffer_len == 0) {
		return;
	}
	if (message == NULL) {
		buffer[0] = '\0';
		return;
	}
	snprintf(buffer, buffer_len, "%s", message);
}

static int contains_case_insensitive(const char *haystack, const char *needle) {
	if (needle == NULL || needle[0] == '\0') {
		return 1;
	}
	if (haystack == NULL) {
		return 0;
	}
	size_t needle_len = strlen(needle);
	size_t hay_len = strlen(haystack);
	if (needle_len > hay_len) {
		return 0;
	}
	for (size_t i = 0; i + needle_len <= hay_len; ++i) {
		size_t matched = 0;
		for (; matched < needle_len; ++matched) {
			char a = (char)tolower((unsigned char)haystack[i + matched]);
			char b = (char)tolower((unsigned char)needle[matched]);
			if (a != b) {
				break;
			}
		}
		if (matched == needle_len) {
			return 1;
		}
	}
	return 0;
}

static int pick_device(cl_platform_id *platform_out, cl_device_id *device_out, char *device_name, size_t device_name_len, const char *platform_hint, const char *device_hint) {
	cl_uint num_platforms = 0;
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
		return 0;
	}
	cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	if (platforms == NULL) {
		return 0;
	}
	if (clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS) {
		free(platforms);
		return 0;
	}

	const cl_device_type device_types[2] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
	for (size_t pass = 0; pass < 2; ++pass) {
		for (cl_uint pi = 0; pi < num_platforms; ++pi) {
			char platform_name[256] = {0};
			clGetPlatformInfo(platforms[pi], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
			if (!contains_case_insensitive(platform_name, platform_hint)) {
				continue;
			}
			cl_uint num_devices = 0;
			cl_int status = clGetDeviceIDs(platforms[pi], device_types[pass], 0, NULL, &num_devices);
			if (status != CL_SUCCESS || num_devices == 0) {
				continue;
			}
			cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
			if (devices == NULL) {
				continue;
			}
			if (clGetDeviceIDs(platforms[pi], device_types[pass], num_devices, devices, NULL) != CL_SUCCESS) {
				free(devices);
				continue;
			}
			for (cl_uint di = 0; di < num_devices; ++di) {
				char current_name[256] = {0};
				clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(current_name), current_name, NULL);
				if (!contains_case_insensitive(current_name, device_hint)) {
					continue;
				}
				*platform_out = platforms[pi];
				*device_out = devices[di];
				snprintf(device_name, device_name_len, "%s", current_name);
				free(devices);
				free(platforms);
				return 1;
			}
			free(devices);
		}
	}
	free(platforms);
	return 0;
}

static void opencl_unload_clblast(opencl_trainer *trainer) {
	if (trainer == NULL || trainer->clblast_lib == NULL) {
		return;
	}
#ifdef _WIN32
	FreeLibrary((HMODULE)trainer->clblast_lib);
#else
	dlclose(trainer->clblast_lib);
#endif
	trainer->clblast_lib = NULL;
	trainer->clblast_sgemm = NULL;
	trainer->clblast_lib_name[0] = '\0';
}

static int opencl_load_clblast(opencl_trainer *trainer, char *error_buffer, size_t error_buffer_len) {
	if (trainer == NULL) {
		set_error(error_buffer, error_buffer_len, "invalid OpenCL trainer");
		return 0;
	}
#ifdef _WIN32
	const char *candidates[] = {"clblast.dll", "libclblast.dll"};
	for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
		HMODULE handle = LoadLibraryA(candidates[i]);
		if (handle != NULL) {
			trainer->clblast_lib = (void*)handle;
			snprintf(trainer->clblast_lib_name, sizeof(trainer->clblast_lib_name), "%s", candidates[i]);
			break;
		}
	}
	if (trainer->clblast_lib == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to load CLBlast DLL (tried clblast.dll, libclblast.dll)");
		return 0;
	}
	trainer->clblast_sgemm = (clblast_sgemm_fn)GetProcAddress((HMODULE)trainer->clblast_lib, "CLBlastSgemm");
#else
	const char *candidates[] = {"libclblast.so.1", "libclblast.so", "libclblast.dylib"};
	for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
		void *handle = dlopen(candidates[i], RTLD_LOCAL | RTLD_LAZY);
		if (handle != NULL) {
			trainer->clblast_lib = handle;
			snprintf(trainer->clblast_lib_name, sizeof(trainer->clblast_lib_name), "%s", candidates[i]);
			break;
		}
	}
	if (trainer->clblast_lib == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to load CLBlast shared library");
		return 0;
	}
	trainer->clblast_sgemm = (clblast_sgemm_fn)dlsym(trainer->clblast_lib, "CLBlastSgemm");
#endif
	if (trainer->clblast_sgemm == NULL) {
		set_error(error_buffer, error_buffer_len, "CLBlastSgemm symbol not found in CLBlast library");
		opencl_unload_clblast(trainer);
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static opencl_trainer* opencl_trainer_create(const char *platform_hint, const char *device_hint, const char *program_source, char *error_buffer, size_t error_buffer_len) {
	cl_platform_id platform = NULL;
	cl_device_id device = NULL;
	char device_name[256] = {0};
	if (!pick_device(&platform, &device, device_name, sizeof(device_name), platform_hint, device_hint)) {
		set_error(error_buffer, error_buffer_len, "no matching OpenCL device found");
		return NULL;
	}

	cl_int status = CL_SUCCESS;
	opencl_trainer *trainer = (opencl_trainer*)calloc(1, sizeof(opencl_trainer));
	if (trainer == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to allocate OpenCL trainer");
		return NULL;
	}

	trainer->context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	if (status != CL_SUCCESS || trainer->context == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to create OpenCL context");
		free(trainer);
		return NULL;
	}

	trainer->queue = clCreateCommandQueue(trainer->context, device, 0, &status);
	if (status != CL_SUCCESS || trainer->queue == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to create OpenCL command queue");
		clReleaseContext(trainer->context);
		free(trainer);
		return NULL;
	}

	const char *sources[1] = {program_source};
	size_t lengths[1] = {strlen(program_source)};
	trainer->program = clCreateProgramWithSource(trainer->context, 1, sources, lengths, &status);
	if (status != CL_SUCCESS || trainer->program == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to create OpenCL program");
		clReleaseCommandQueue(trainer->queue);
		clReleaseContext(trainer->context);
		free(trainer);
		return NULL;
	}

	status = clBuildProgram(trainer->program, 1, &device, "", NULL, NULL);
	if (status != CL_SUCCESS) {
		size_t log_size = 0;
		clGetProgramBuildInfo(trainer->program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char*)malloc(log_size + 1);
		if (log != NULL) {
			clGetProgramBuildInfo(trainer->program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			log[log_size] = '\0';
			set_error(error_buffer, error_buffer_len, log);
			free(log);
		} else {
			set_error(error_buffer, error_buffer_len, "failed to build OpenCL program");
		}
		clReleaseProgram(trainer->program);
		clReleaseCommandQueue(trainer->queue);
		clReleaseContext(trainer->context);
		free(trainer);
		return NULL;
	}

	trainer->add_bias_relu_kernel = clCreateKernel(trainer->program, "add_bias_relu", &status);
	if (status != CL_SUCCESS || trainer->add_bias_relu_kernel == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to create add_bias_relu kernel");
		clReleaseProgram(trainer->program);
		clReleaseCommandQueue(trainer->queue);
		clReleaseContext(trainer->context);
		free(trainer);
		return NULL;
	}
	trainer->add_bias_kernel = clCreateKernel(trainer->program, "add_bias", &status);
	if (status != CL_SUCCESS || trainer->add_bias_kernel == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to create add_bias kernel");
		clReleaseKernel(trainer->add_bias_relu_kernel);
		clReleaseProgram(trainer->program);
		clReleaseCommandQueue(trainer->queue);
		clReleaseContext(trainer->context);
		free(trainer);
		return NULL;
	}
	trainer->relu_backward_kernel = clCreateKernel(trainer->program, "relu_backward", &status);
	if (status != CL_SUCCESS || trainer->relu_backward_kernel == NULL) {
		set_error(error_buffer, error_buffer_len, "failed to create relu_backward kernel");
		clReleaseKernel(trainer->add_bias_kernel);
		clReleaseKernel(trainer->add_bias_relu_kernel);
		clReleaseProgram(trainer->program);
		clReleaseCommandQueue(trainer->queue);
		clReleaseContext(trainer->context);
		free(trainer);
		return NULL;
	}
	if (!opencl_load_clblast(trainer, error_buffer, error_buffer_len)) {
		clReleaseKernel(trainer->relu_backward_kernel);
		clReleaseKernel(trainer->add_bias_kernel);
		clReleaseKernel(trainer->add_bias_relu_kernel);
		clReleaseProgram(trainer->program);
		clReleaseCommandQueue(trainer->queue);
		clReleaseContext(trainer->context);
		free(trainer);
		return NULL;
	}

	trainer->device = device;
	snprintf(trainer->device_name, sizeof(trainer->device_name), "%s", device_name);
	set_error(error_buffer, error_buffer_len, "");
	return trainer;
}

static void opencl_trainer_destroy(opencl_trainer *trainer) {
	if (trainer == NULL) {
		return;
	}
	if (trainer->relu_backward_kernel != NULL) {
		clReleaseKernel(trainer->relu_backward_kernel);
	}
	if (trainer->add_bias_kernel != NULL) {
		clReleaseKernel(trainer->add_bias_kernel);
	}
	if (trainer->add_bias_relu_kernel != NULL) {
		clReleaseKernel(trainer->add_bias_relu_kernel);
	}
	if (trainer->program != NULL) {
		clReleaseProgram(trainer->program);
	}
	opencl_unload_clblast(trainer);
	if (trainer->queue != NULL) {
		clReleaseCommandQueue(trainer->queue);
	}
	if (trainer->context != NULL) {
		clReleaseContext(trainer->context);
	}
	free(trainer);
}

static cl_mem opencl_alloc(opencl_trainer *trainer, size_t bytes, char *error_buffer, size_t error_buffer_len) {
	cl_int status = CL_SUCCESS;
	cl_mem buffer = clCreateBuffer(trainer->context, CL_MEM_READ_WRITE, bytes, NULL, &status);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to allocate OpenCL buffer");
		return NULL;
	}
	set_error(error_buffer, error_buffer_len, "");
	return buffer;
}

static int opencl_free(cl_mem buffer, char *error_buffer, size_t error_buffer_len) {
	if (buffer == NULL) {
		return 1;
	}
	if (clReleaseMemObject(buffer) != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to release OpenCL buffer");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static int opencl_write(opencl_trainer *trainer, cl_mem buffer, const void *data, size_t bytes, char *error_buffer, size_t error_buffer_len) {
	if (bytes == 0) {
		return 1;
	}
	cl_int status = clEnqueueWriteBuffer(trainer->queue, buffer, CL_TRUE, 0, bytes, data, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to write OpenCL buffer");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static int opencl_read(opencl_trainer *trainer, cl_mem buffer, void *data, size_t bytes, char *error_buffer, size_t error_buffer_len) {
	if (bytes == 0) {
		return 1;
	}
	cl_int status = clEnqueueReadBuffer(trainer->queue, buffer, CL_TRUE, 0, bytes, data, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to read OpenCL buffer");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static int opencl_zero(opencl_trainer *trainer, cl_mem buffer, size_t bytes, char *error_buffer, size_t error_buffer_len) {
	if (bytes == 0) {
		return 1;
	}
	float pattern = 0.0f;
	cl_int status = clEnqueueFillBuffer(trainer->queue, buffer, &pattern, sizeof(pattern), 0, bytes, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to zero OpenCL buffer");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static int opencl_sgemm(opencl_trainer *trainer, int transpose_a, int transpose_b,
                        size_t m, size_t n, size_t k, float alpha,
                        cl_mem a_buffer, size_t a_ld,
                        cl_mem b_buffer, size_t b_ld,
                        float beta,
                        cl_mem c_buffer, size_t c_ld,
                        char *error_buffer, size_t error_buffer_len) {
	if (trainer == NULL || trainer->clblast_sgemm == NULL) {
		set_error(error_buffer, error_buffer_len, "CLBlast is not initialized");
		return 0;
	}
	CLBlastTranspose ta = transpose_a ? CLBLAST_TRANSPOSE_YES : CLBLAST_TRANSPOSE_NO;
	CLBlastTranspose tb = transpose_b ? CLBLAST_TRANSPOSE_YES : CLBLAST_TRANSPOSE_NO;
	CLBlastStatusCode status = trainer->clblast_sgemm(CLBLAST_LAYOUT_ROW_MAJOR, ta, tb,
			m, n, k, alpha,
			a_buffer, 0, a_ld,
			b_buffer, 0, b_ld,
			beta,
			c_buffer, 0, c_ld,
			&trainer->queue, NULL);
	if (status != CLBLAST_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "CLBlast SGEMM failed");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static int opencl_add_bias_relu(opencl_trainer *trainer, cl_mem matrix, cl_mem bias, int rows, int cols, char *error_buffer, size_t error_buffer_len) {
	size_t total = (size_t)rows * (size_t)cols;
	if (total == 0) {
		return 1;
	}
	cl_int status = CL_SUCCESS;
	status |= clSetKernelArg(trainer->add_bias_relu_kernel, 0, sizeof(cl_mem), &matrix);
	status |= clSetKernelArg(trainer->add_bias_relu_kernel, 1, sizeof(cl_mem), &bias);
	status |= clSetKernelArg(trainer->add_bias_relu_kernel, 2, sizeof(int), &cols);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to set add_bias_relu kernel args");
		return 0;
	}
	size_t global = total;
	status = clEnqueueNDRangeKernel(trainer->queue, trainer->add_bias_relu_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to launch add_bias_relu kernel");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static int opencl_add_bias(opencl_trainer *trainer, cl_mem matrix, cl_mem bias, int rows, int cols, char *error_buffer, size_t error_buffer_len) {
	size_t total = (size_t)rows * (size_t)cols;
	if (total == 0) {
		return 1;
	}
	cl_int status = CL_SUCCESS;
	status |= clSetKernelArg(trainer->add_bias_kernel, 0, sizeof(cl_mem), &matrix);
	status |= clSetKernelArg(trainer->add_bias_kernel, 1, sizeof(cl_mem), &bias);
	status |= clSetKernelArg(trainer->add_bias_kernel, 2, sizeof(int), &cols);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to set add_bias kernel args");
		return 0;
	}
	size_t global = total;
	status = clEnqueueNDRangeKernel(trainer->queue, trainer->add_bias_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to launch add_bias kernel");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}

static int opencl_relu_backward(opencl_trainer *trainer, cl_mem gradient, cl_mem activations, int total, char *error_buffer, size_t error_buffer_len) {
	if (total == 0) {
		return 1;
	}
	cl_int status = CL_SUCCESS;
	status |= clSetKernelArg(trainer->relu_backward_kernel, 0, sizeof(cl_mem), &gradient);
	status |= clSetKernelArg(trainer->relu_backward_kernel, 1, sizeof(cl_mem), &activations);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to set relu_backward kernel args");
		return 0;
	}
	size_t global = (size_t)total;
	status = clEnqueueNDRangeKernel(trainer->queue, trainer->relu_backward_kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		set_error(error_buffer, error_buffer_len, "failed to launch relu_backward kernel");
		return 0;
	}
	set_error(error_buffer, error_buffer_len, "");
	return 1;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"strings"
	"unsafe"

	"github.com/pokemon-engine/simulator"
)

const openCLTrainerKernels = `
__kernel void add_bias_relu(__global float* matrix, __global const float* bias, const int cols) {
	const int idx = get_global_id(0);
	const int col = idx % cols;
	float v = matrix[idx] + bias[col];
	matrix[idx] = v > 0.0f ? v : 0.0f;
}

__kernel void add_bias(__global float* matrix, __global const float* bias, const int cols) {
	const int idx = get_global_id(0);
	const int col = idx % cols;
	matrix[idx] += bias[col];
}

__kernel void relu_backward(__global float* gradient, __global const float* activations) {
	const int idx = get_global_id(0);
	if (activations[idx] <= 0.0f) {
		gradient[idx] = 0.0f;
	}
}
`

type openCLBuffers struct {
	x              C.cl_mem
	w1             C.cl_mem
	b1             C.cl_mem
	h1             C.cl_mem
	w2             C.cl_mem
	b2             C.cl_mem
	h2             C.cl_mem
	wRegret        C.cl_mem
	bRegret        C.cl_mem
	regret         C.cl_mem
	wStrategy      C.cl_mem
	bStrategy      C.cl_mem
	strategyLogits C.cl_mem
	wValue         C.cl_mem
	valueLogits    C.cl_mem
	dRegret        C.cl_mem
	dStrategy      C.cl_mem
	dValue         C.cl_mem
	dH2            C.cl_mem
	dH1            C.cl_mem
	gradWRegret    C.cl_mem
	gradWStrategy  C.cl_mem
	gradWValue     C.cl_mem
	gradW2         C.cl_mem
	gradW1         C.cl_mem
}

type openCLHostWorkspace struct {
	x              []float32
	legalMask      []float32
	regretTargets  []float32
	strategyTarget []float32
	valueTargets   []float32
	h1             []float32
	h2             []float32
	regret         []float32
	strategyLogits []float32
	valueLogits    []float32
	dRegret        []float32
	dStrategy      []float32
	dValue         []float32
	dH2            []float32
	dH1            []float32
	gradWRegret    []float32
	gradWStrategy  []float32
	gradWValue     []float32
	gradW2         []float32
	gradW1         []float32
	w1             []float32
	b1             []float32
	w2             []float32
	b2             []float32
	wRegret        []float32
	bRegret        []float32
	wStrategy      []float32
	bStrategy      []float32
	wValue         []float32
}

type openCLTrainer struct {
	model      *Model
	hp         TrainingHyperParams
	batchSize  int
	deviceName string
	ctx        *C.opencl_trainer
	buffers    openCLBuffers
	host       openCLHostWorkspace
	examples   []TrainingExample
}

func newOpenCLTrainer(model *Model, hp TrainingHyperParams, batchSize int, platformHint string, deviceHint string) (exampleTrainer, error) {
	if batchSize <= 1 {
		batchSize = 64
	}

	cPlatform := C.CString(platformHint)
	cDevice := C.CString(deviceHint)
	cSource := C.CString(openCLTrainerKernels)
	defer C.free(unsafe.Pointer(cPlatform))
	defer C.free(unsafe.Pointer(cDevice))
	defer C.free(unsafe.Pointer(cSource))

	errBuf := make([]byte, 4096)
	ctx := C.opencl_trainer_create(cPlatform, cDevice, cSource, (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf)))
	if ctx == nil {
		return nil, fmt.Errorf("opencl init failed: %s", cStringBytes(errBuf))
	}

	trainer := &openCLTrainer{
		model:      model,
		hp:         hp,
		batchSize:  batchSize,
		deviceName: C.GoString(&ctx.device_name[0]),
		ctx:        ctx,
		examples:   make([]TrainingExample, 0, batchSize),
	}
	if err := trainer.initBuffers(); err != nil {
		trainer.Close()
		return nil, err
	}
	return trainer, nil
}

func (t *openCLTrainer) initBuffers() error {
	t.host = openCLHostWorkspace{
		x:              make([]float32, t.batchSize*FeatureSize),
		legalMask:      make([]float32, t.batchSize*simulator.MaxActions),
		regretTargets:  make([]float32, t.batchSize*simulator.MaxActions),
		strategyTarget: make([]float32, t.batchSize*simulator.MaxActions),
		valueTargets:   make([]float32, t.batchSize),
		h1:             make([]float32, t.batchSize*t.model.Hidden1),
		h2:             make([]float32, t.batchSize*t.model.Hidden2),
		regret:         make([]float32, t.batchSize*simulator.MaxActions),
		strategyLogits: make([]float32, t.batchSize*simulator.MaxActions),
		valueLogits:    make([]float32, t.batchSize),
		dRegret:        make([]float32, t.batchSize*simulator.MaxActions),
		dStrategy:      make([]float32, t.batchSize*simulator.MaxActions),
		dValue:         make([]float32, t.batchSize),
		dH2:            make([]float32, t.batchSize*t.model.Hidden2),
		dH1:            make([]float32, t.batchSize*t.model.Hidden1),
		gradWRegret:    make([]float32, len(t.model.WRegret)),
		gradWStrategy:  make([]float32, len(t.model.WStrategy)),
		gradWValue:     make([]float32, len(t.model.WValue)),
		gradW2:         make([]float32, len(t.model.W2)),
		gradW1:         make([]float32, len(t.model.W1)),
		w1:             make([]float32, len(t.model.W1)),
		b1:             make([]float32, len(t.model.B1)),
		w2:             make([]float32, len(t.model.W2)),
		b2:             make([]float32, len(t.model.B2)),
		wRegret:        make([]float32, len(t.model.WRegret)),
		bRegret:        make([]float32, len(t.model.BRegret)),
		wStrategy:      make([]float32, len(t.model.WStrategy)),
		bStrategy:      make([]float32, len(t.model.BStrategy)),
		wValue:         make([]float32, len(t.model.WValue)),
	}

	var err error
	if t.buffers.x, err = t.allocFloat32Buffer(len(t.host.x)); err != nil {
		return err
	}
	if t.buffers.w1, err = t.allocFloat32Buffer(len(t.host.w1)); err != nil {
		return err
	}
	if t.buffers.b1, err = t.allocFloat32Buffer(len(t.host.b1)); err != nil {
		return err
	}
	if t.buffers.h1, err = t.allocFloat32Buffer(len(t.host.h1)); err != nil {
		return err
	}
	if t.buffers.w2, err = t.allocFloat32Buffer(len(t.host.w2)); err != nil {
		return err
	}
	if t.buffers.b2, err = t.allocFloat32Buffer(len(t.host.b2)); err != nil {
		return err
	}
	if t.buffers.h2, err = t.allocFloat32Buffer(len(t.host.h2)); err != nil {
		return err
	}
	if t.buffers.wRegret, err = t.allocFloat32Buffer(len(t.host.wRegret)); err != nil {
		return err
	}
	if t.buffers.bRegret, err = t.allocFloat32Buffer(len(t.host.bRegret)); err != nil {
		return err
	}
	if t.buffers.regret, err = t.allocFloat32Buffer(len(t.host.regret)); err != nil {
		return err
	}
	if t.buffers.wStrategy, err = t.allocFloat32Buffer(len(t.host.wStrategy)); err != nil {
		return err
	}
	if t.buffers.bStrategy, err = t.allocFloat32Buffer(len(t.host.bStrategy)); err != nil {
		return err
	}
	if t.buffers.strategyLogits, err = t.allocFloat32Buffer(len(t.host.strategyLogits)); err != nil {
		return err
	}
	if t.buffers.wValue, err = t.allocFloat32Buffer(len(t.host.wValue)); err != nil {
		return err
	}
	if t.buffers.valueLogits, err = t.allocFloat32Buffer(len(t.host.valueLogits)); err != nil {
		return err
	}
	if t.buffers.dRegret, err = t.allocFloat32Buffer(len(t.host.dRegret)); err != nil {
		return err
	}
	if t.buffers.dStrategy, err = t.allocFloat32Buffer(len(t.host.dStrategy)); err != nil {
		return err
	}
	if t.buffers.dValue, err = t.allocFloat32Buffer(len(t.host.dValue)); err != nil {
		return err
	}
	if t.buffers.dH2, err = t.allocFloat32Buffer(len(t.host.dH2)); err != nil {
		return err
	}
	if t.buffers.dH1, err = t.allocFloat32Buffer(len(t.host.dH1)); err != nil {
		return err
	}
	if t.buffers.gradWRegret, err = t.allocFloat32Buffer(len(t.host.gradWRegret)); err != nil {
		return err
	}
	if t.buffers.gradWStrategy, err = t.allocFloat32Buffer(len(t.host.gradWStrategy)); err != nil {
		return err
	}
	if t.buffers.gradWValue, err = t.allocFloat32Buffer(len(t.host.gradWValue)); err != nil {
		return err
	}
	if t.buffers.gradW2, err = t.allocFloat32Buffer(len(t.host.gradW2)); err != nil {
		return err
	}
	if t.buffers.gradW1, err = t.allocFloat32Buffer(len(t.host.gradW1)); err != nil {
		return err
	}
	return nil
}

func (t *openCLTrainer) Train(example TrainingExample) ([]TrainingMetrics, error) {
	t.examples = append(t.examples, example)
	if len(t.examples) < t.batchSize {
		return nil, nil
	}
	return t.flushBatch()
}

func (t *openCLTrainer) Flush() ([]TrainingMetrics, error) {
	return t.flushBatch()
}

func (t *openCLTrainer) flushBatch() ([]TrainingMetrics, error) {
	if len(t.examples) == 0 {
		return nil, nil
	}
	n := len(t.examples)
	if err := t.packBatch(n); err != nil {
		return nil, err
	}
	if err := t.runForward(n); err != nil {
		return nil, err
	}
	metrics := t.computeOutputGradients(n)
	if err := t.runBackward(n); err != nil {
		return nil, err
	}
	t.applyGradients(n)
	t.examples = t.examples[:0]
	return metrics, nil
}

func (t *openCLTrainer) packBatch(n int) error {
	clear(t.host.x)
	clear(t.host.legalMask)
	clear(t.host.regretTargets)
	clear(t.host.strategyTarget)
	clear(t.host.valueTargets)

	for row, example := range t.examples[:n] {
		baseFeature := row * FeatureSize
		for i := 0; i < FeatureSize && i < len(example.Features); i++ {
			t.host.x[baseFeature+i] = float32(example.Features[i])
		}
		baseAction := row * simulator.MaxActions
		for i := 0; i < simulator.MaxActions && i < len(example.LegalMask); i++ {
			t.host.legalMask[baseAction+i] = float32(example.LegalMask[i])
		}
		for i := 0; i < simulator.MaxActions; i++ {
			t.host.regretTargets[baseAction+i] = float32(example.RegretTargets[i])
			t.host.strategyTarget[baseAction+i] = float32(example.StrategyTargets[i])
		}
		t.host.valueTargets[row] = float32(example.ValueTarget)
	}

	copyFloat64SliceTo32(t.host.w1, t.model.W1)
	copyFloat64SliceTo32(t.host.b1, t.model.B1)
	copyFloat64SliceTo32(t.host.w2, t.model.W2)
	copyFloat64SliceTo32(t.host.b2, t.model.B2)
	copyFloat64SliceTo32(t.host.wRegret, t.model.WRegret)
	copyFloat64SliceTo32(t.host.bRegret, t.model.BRegret)
	copyFloat64SliceTo32(t.host.wStrategy, t.model.WStrategy)
	copyFloat64SliceTo32(t.host.bStrategy, t.model.BStrategy)
	copyFloat64SliceTo32(t.host.wValue, t.model.WValue)

	if err := t.writeFloat32(t.buffers.x, t.host.x[:n*FeatureSize]); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.w1, t.host.w1); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.b1, t.host.b1); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.w2, t.host.w2); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.b2, t.host.b2); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.wRegret, t.host.wRegret); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.bRegret, t.host.bRegret); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.wStrategy, t.host.wStrategy); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.bStrategy, t.host.bStrategy); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.wValue, t.host.wValue); err != nil {
		return err
	}
	return nil
}

func (t *openCLTrainer) runForward(n int) error {
	if err := t.sgemm(false, true, n, t.model.Hidden1, FeatureSize, 1, t.buffers.x, FeatureSize, t.buffers.w1, FeatureSize, 0, t.buffers.h1, t.model.Hidden1); err != nil {
		return err
	}
	if err := t.addBiasRelu(t.buffers.h1, t.buffers.b1, n, t.model.Hidden1); err != nil {
		return err
	}
	if err := t.sgemm(false, true, n, t.model.Hidden2, t.model.Hidden1, 1, t.buffers.h1, t.model.Hidden1, t.buffers.w2, t.model.Hidden1, 0, t.buffers.h2, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.addBiasRelu(t.buffers.h2, t.buffers.b2, n, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.sgemm(false, true, n, simulator.MaxActions, t.model.Hidden2, 1, t.buffers.h2, t.model.Hidden2, t.buffers.wRegret, t.model.Hidden2, 0, t.buffers.regret, simulator.MaxActions); err != nil {
		return err
	}
	if err := t.addBias(t.buffers.regret, t.buffers.bRegret, n, simulator.MaxActions); err != nil {
		return err
	}
	if err := t.sgemm(false, true, n, simulator.MaxActions, t.model.Hidden2, 1, t.buffers.h2, t.model.Hidden2, t.buffers.wStrategy, t.model.Hidden2, 0, t.buffers.strategyLogits, simulator.MaxActions); err != nil {
		return err
	}
	if err := t.addBias(t.buffers.strategyLogits, t.buffers.bStrategy, n, simulator.MaxActions); err != nil {
		return err
	}
	if err := t.sgemm(false, false, n, 1, t.model.Hidden2, 1, t.buffers.h2, t.model.Hidden2, t.buffers.wValue, 1, 0, t.buffers.valueLogits, 1); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.h1, t.host.h1[:n*t.model.Hidden1]); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.h2, t.host.h2[:n*t.model.Hidden2]); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.regret, t.host.regret[:n*simulator.MaxActions]); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.strategyLogits, t.host.strategyLogits[:n*simulator.MaxActions]); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.valueLogits, t.host.valueLogits[:n]); err != nil {
		return err
	}
	return nil
}

func (t *openCLTrainer) computeOutputGradients(n int) []TrainingMetrics {
	clear(t.host.dRegret)
	clear(t.host.dStrategy)
	clear(t.host.dValue)
	metrics := make([]TrainingMetrics, n)
	strategy := make([]float32, simulator.MaxActions)

	for row := 0; row < n; row++ {
		actionOffset := row * simulator.MaxActions
		logits := t.host.strategyLogits[actionOffset : actionOffset+simulator.MaxActions]
		mask := t.host.legalMask[actionOffset : actionOffset+simulator.MaxActions]
		maskedSoftmax32(logits, mask, strategy)

		value := sigmoid(float64(t.host.valueLogits[row] + float32(t.model.BValue)))
		valueError := math.Abs(value - float64(t.host.valueTargets[row]))
		valueGrad := float32((value - float64(t.host.valueTargets[row])) * t.hp.ValueWeight)
		t.host.dValue[row] = valueGrad

		loss := valueError * valueError * t.hp.ValueWeight
		policyCross := 0.0
		for out := 0; out < simulator.MaxActions; out++ {
			if mask[out] == 0 {
				continue
			}
			regretPred := t.host.regret[actionOffset+out]
			regretTarget := t.host.regretTargets[actionOffset+out]
			dRegret := float32(2.0) * (regretPred - regretTarget) * float32(t.hp.RegretWeight)
			t.host.dRegret[actionOffset+out] = dRegret
			diff := float64(regretPred - regretTarget)
			loss += diff * diff * t.hp.RegretWeight

			strategyTarget := t.host.strategyTarget[actionOffset+out]
			strategyPred := strategy[out]
			dStrategy := (strategyPred - strategyTarget) * float32(t.hp.StrategyWeight)
			t.host.dStrategy[actionOffset+out] = dStrategy
			if strategyTarget > 0 && strategyPred > 1e-9 {
				policyCross += -float64(strategyTarget) * math.Log(float64(strategyPred))
			}
			loss += float64(dStrategy * dStrategy)
		}
		metrics[row] = TrainingMetrics{
			Loss:        loss,
			ValueError:  valueError,
			PolicyCross: policyCross,
		}
	}
	return metrics
}

func (t *openCLTrainer) runBackward(n int) error {
	if err := t.writeFloat32(t.buffers.dRegret, t.host.dRegret[:n*simulator.MaxActions]); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.dStrategy, t.host.dStrategy[:n*simulator.MaxActions]); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.dValue, t.host.dValue[:n]); err != nil {
		return err
	}

	if err := t.sgemm(false, false, n, t.model.Hidden2, simulator.MaxActions, 1, t.buffers.dRegret, simulator.MaxActions, t.buffers.wRegret, t.model.Hidden2, 0, t.buffers.dH2, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.sgemm(false, false, n, t.model.Hidden2, simulator.MaxActions, 1, t.buffers.dStrategy, simulator.MaxActions, t.buffers.wStrategy, t.model.Hidden2, 1, t.buffers.dH2, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.sgemm(false, false, n, t.model.Hidden2, 1, 1, t.buffers.dValue, 1, t.buffers.wValue, t.model.Hidden2, 1, t.buffers.dH2, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.reluBackward(t.buffers.dH2, t.buffers.h2, n*t.model.Hidden2); err != nil {
		return err
	}
	if err := t.sgemm(true, false, 1, t.model.Hidden2, n, 1, t.buffers.dValue, 1, t.buffers.h2, t.model.Hidden2, 0, t.buffers.gradWValue, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.sgemm(true, false, simulator.MaxActions, t.model.Hidden2, n, 1, t.buffers.dRegret, simulator.MaxActions, t.buffers.h2, t.model.Hidden2, 0, t.buffers.gradWRegret, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.sgemm(true, false, simulator.MaxActions, t.model.Hidden2, n, 1, t.buffers.dStrategy, simulator.MaxActions, t.buffers.h2, t.model.Hidden2, 0, t.buffers.gradWStrategy, t.model.Hidden2); err != nil {
		return err
	}
	if err := t.sgemm(false, false, n, t.model.Hidden1, t.model.Hidden2, 1, t.buffers.dH2, t.model.Hidden2, t.buffers.w2, t.model.Hidden1, 0, t.buffers.dH1, t.model.Hidden1); err != nil {
		return err
	}
	if err := t.reluBackward(t.buffers.dH1, t.buffers.h1, n*t.model.Hidden1); err != nil {
		return err
	}
	if err := t.sgemm(true, false, t.model.Hidden2, t.model.Hidden1, n, 1, t.buffers.dH2, t.model.Hidden2, t.buffers.h1, t.model.Hidden1, 0, t.buffers.gradW2, t.model.Hidden1); err != nil {
		return err
	}
	if err := t.sgemm(true, false, t.model.Hidden1, FeatureSize, n, 1, t.buffers.dH1, t.model.Hidden1, t.buffers.x, FeatureSize, 0, t.buffers.gradW1, FeatureSize); err != nil {
		return err
	}

	if err := t.readFloat32(t.buffers.dH2, t.host.dH2[:n*t.model.Hidden2]); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.dH1, t.host.dH1[:n*t.model.Hidden1]); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.gradWValue, t.host.gradWValue); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.gradWRegret, t.host.gradWRegret); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.gradWStrategy, t.host.gradWStrategy); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.gradW2, t.host.gradW2); err != nil {
		return err
	}
	if err := t.readFloat32(t.buffers.gradW1, t.host.gradW1); err != nil {
		return err
	}
	return nil
}

func (t *openCLTrainer) applyGradients(n int) {
	scale := 1.0 / float64(n)
	for i := range t.model.WValue {
		grad := float64(t.host.gradWValue[i]) * scale
		t.model.WValue[i] -= t.hp.LearningRate * clipGrad(grad)
	}
	t.model.BValue -= t.hp.LearningRate * clipGrad(sumFloat32(t.host.dValue[:n])*scale)

	for out := 0; out < simulator.MaxActions; out++ {
		row := out * t.model.Hidden2
		for i := 0; i < t.model.Hidden2; i++ {
			grad := float64(t.host.gradWRegret[row+i]) * scale
			t.model.WRegret[row+i] -= t.hp.LearningRate * clipGrad(grad)
		}
		t.model.BRegret[out] -= t.hp.LearningRate * clipGrad(sumColumn(t.host.dRegret[:n*simulator.MaxActions], simulator.MaxActions, out)*scale)
	}

	for out := 0; out < simulator.MaxActions; out++ {
		row := out * t.model.Hidden2
		for i := 0; i < t.model.Hidden2; i++ {
			grad := float64(t.host.gradWStrategy[row+i]) * scale
			t.model.WStrategy[row+i] -= t.hp.LearningRate * clipGrad(grad)
		}
		t.model.BStrategy[out] -= t.hp.LearningRate * clipGrad(sumColumn(t.host.dStrategy[:n*simulator.MaxActions], simulator.MaxActions, out)*scale)
	}

	for i := 0; i < t.model.Hidden2; i++ {
		t.model.B2[i] -= t.hp.LearningRate * clipGrad(sumColumn(t.host.dH2[:n*t.model.Hidden2], t.model.Hidden2, i)*scale)
	}
	for i := range t.model.W2 {
		grad := float64(t.host.gradW2[i]) * scale
		t.model.W2[i] -= t.hp.LearningRate * clipGrad(grad)
	}

	for i := 0; i < t.model.Hidden1; i++ {
		t.model.B1[i] -= t.hp.LearningRate * clipGrad(sumColumn(t.host.dH1[:n*t.model.Hidden1], t.model.Hidden1, i)*scale)
	}
	for i := range t.model.W1 {
		grad := float64(t.host.gradW1[i]) * scale
		t.model.W1[i] -= t.hp.LearningRate * clipGrad(grad)
	}
}

func (t *openCLTrainer) Close() error {
	for _, buffer := range []C.cl_mem{
		t.buffers.x, t.buffers.w1, t.buffers.b1, t.buffers.h1, t.buffers.w2, t.buffers.b2, t.buffers.h2,
		t.buffers.wRegret, t.buffers.bRegret, t.buffers.regret, t.buffers.wStrategy, t.buffers.bStrategy,
		t.buffers.strategyLogits, t.buffers.wValue, t.buffers.valueLogits, t.buffers.dRegret, t.buffers.dStrategy,
		t.buffers.dValue, t.buffers.dH2, t.buffers.dH1, t.buffers.gradWRegret, t.buffers.gradWStrategy,
		t.buffers.gradWValue, t.buffers.gradW2, t.buffers.gradW1,
	} {
		if buffer != nil {
			_ = t.freeBuffer(buffer)
		}
	}
	if t.ctx != nil {
		C.opencl_trainer_destroy(t.ctx)
		t.ctx = nil
	}
	return nil
}

func (t *openCLTrainer) Name() string {
	name := strings.TrimSpace(t.deviceName)
	if name == "" {
		return "opencl"
	}
	return "opencl:" + name
}

func (t *openCLTrainer) allocFloat32Buffer(length int) (C.cl_mem, error) {
	if length <= 0 {
		length = 1
	}
	var errBuf [256]byte
	buffer := C.opencl_alloc(t.ctx, C.size_t(length*4), (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf)))
	if buffer == nil {
		return nil, errors.New(cStringBytes(errBuf[:]))
	}
	return buffer, nil
}

func (t *openCLTrainer) freeBuffer(buffer C.cl_mem) error {
	var errBuf [256]byte
	if C.opencl_free(buffer, (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) == 0 {
		return errors.New(cStringBytes(errBuf[:]))
	}
	return nil
}

func (t *openCLTrainer) writeFloat32(buffer C.cl_mem, data []float32) error {
	if len(data) == 0 {
		return nil
	}
	var errBuf [256]byte
	if C.opencl_write(t.ctx, buffer, unsafe.Pointer(&data[0]), C.size_t(len(data)*4), (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) == 0 {
		return errors.New(cStringBytes(errBuf[:]))
	}
	runtime.KeepAlive(data)
	return nil
}

func (t *openCLTrainer) readFloat32(buffer C.cl_mem, data []float32) error {
	if len(data) == 0 {
		return nil
	}
	var errBuf [256]byte
	if C.opencl_read(t.ctx, buffer, unsafe.Pointer(&data[0]), C.size_t(len(data)*4), (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) == 0 {
		return errors.New(cStringBytes(errBuf[:]))
	}
	runtime.KeepAlive(data)
	return nil
}

func (t *openCLTrainer) sgemm(transposeA bool, transposeB bool, m int, n int, k int, alpha float32, a C.cl_mem, aLD int, b C.cl_mem, bLD int, beta float32, c C.cl_mem, cLD int) error {
	var errBuf [256]byte
	var ta, tb C.int
	if transposeA {
		ta = 1
	}
	if transposeB {
		tb = 1
	}
	if C.opencl_sgemm(t.ctx, ta, tb, C.size_t(m), C.size_t(n), C.size_t(k), C.float(alpha), a, C.size_t(aLD), b, C.size_t(bLD), C.float(beta), c, C.size_t(cLD), (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) == 0 {
		return errors.New(cStringBytes(errBuf[:]))
	}
	return nil
}

func (t *openCLTrainer) addBiasRelu(matrix C.cl_mem, bias C.cl_mem, rows int, cols int) error {
	var errBuf [256]byte
	if C.opencl_add_bias_relu(t.ctx, matrix, bias, C.int(rows), C.int(cols), (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) == 0 {
		return errors.New(cStringBytes(errBuf[:]))
	}
	return nil
}

func (t *openCLTrainer) addBias(matrix C.cl_mem, bias C.cl_mem, rows int, cols int) error {
	var errBuf [256]byte
	if C.opencl_add_bias(t.ctx, matrix, bias, C.int(rows), C.int(cols), (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) == 0 {
		return errors.New(cStringBytes(errBuf[:]))
	}
	return nil
}

func (t *openCLTrainer) reluBackward(gradient C.cl_mem, activations C.cl_mem, total int) error {
	var errBuf [256]byte
	if C.opencl_relu_backward(t.ctx, gradient, activations, C.int(total), (*C.char)(unsafe.Pointer(&errBuf[0])), C.size_t(len(errBuf))) == 0 {
		return errors.New(cStringBytes(errBuf[:]))
	}
	return nil
}

type openCLBatchPredictBackend struct {
	trainer       *openCLTrainer
	weightsLoaded bool
}

func newOpenCLBatchPredictBackend(model *Model, platformHint string, deviceHint string, batchSize int) (batchPredictBackend, error) {
	baseTrainer, err := newOpenCLTrainer(model, TrainingHyperParams{LearningRate: 0.0005, RegretWeight: 1, StrategyWeight: 1, ValueWeight: 0.5}, batchSize, platformHint, deviceHint)
	if err != nil {
		return nil, err
	}
	trainer, ok := baseTrainer.(*openCLTrainer)
	if !ok {
		_ = baseTrainer.Close()
		return nil, fmt.Errorf("unexpected opencl trainer type %T", baseTrainer)
	}
	return &openCLBatchPredictBackend{trainer: trainer}, nil
}

func (b *openCLBatchPredictBackend) Name() string {
	if b == nil || b.trainer == nil {
		return "opencl-predictor"
	}
	return "opencl-predictor:" + b.trainer.Name()
}

func (b *openCLBatchPredictBackend) Close() error {
	if b == nil || b.trainer == nil {
		return nil
	}
	return b.trainer.Close()
}

func (b *openCLBatchPredictBackend) PredictBatch(features []float32, legalMasks []float32, batch int, outRegret []float32, outPolicy []float32, outValue []float32) error {
	if b == nil || b.trainer == nil {
		return errors.New("opencl predictor is not initialized")
	}
	t := b.trainer
	if batch < 0 || batch > t.batchSize {
		return fmt.Errorf("predict batch size %d exceeds backend capacity %d", batch, t.batchSize)
	}
	if batch == 0 {
		return nil
	}

	clear(t.host.x)
	clear(t.host.legalMask)
	copy(t.host.x[:batch*FeatureSize], features[:batch*FeatureSize])
	copy(t.host.legalMask[:batch*simulator.MaxActions], legalMasks[:batch*simulator.MaxActions])

	if err := t.writeFloat32(t.buffers.x, t.host.x[:batch*FeatureSize]); err != nil {
		return err
	}
	if err := b.ensureWeightsLoaded(); err != nil {
		return err
	}

	if err := t.runForward(batch); err != nil {
		return err
	}

	strategy := make([]float32, simulator.MaxActions)
	for row := 0; row < batch; row++ {
		actionOffset := row * simulator.MaxActions
		logits := t.host.strategyLogits[actionOffset : actionOffset+simulator.MaxActions]
		mask := t.host.legalMask[actionOffset : actionOffset+simulator.MaxActions]
		maskedSoftmax32(logits, mask, strategy)
		copy(outRegret[actionOffset:actionOffset+simulator.MaxActions], t.host.regret[actionOffset:actionOffset+simulator.MaxActions])
		copy(outPolicy[actionOffset:actionOffset+simulator.MaxActions], strategy)
		outValue[row] = float32(sigmoid(float64(t.host.valueLogits[row] + float32(t.model.BValue))))
	}
	return nil
}

func (b *openCLBatchPredictBackend) ensureWeightsLoaded() error {
	if b.weightsLoaded {
		return nil
	}
	t := b.trainer
	copyFloat64SliceTo32(t.host.w1, t.model.W1)
	copyFloat64SliceTo32(t.host.b1, t.model.B1)
	copyFloat64SliceTo32(t.host.w2, t.model.W2)
	copyFloat64SliceTo32(t.host.b2, t.model.B2)
	copyFloat64SliceTo32(t.host.wRegret, t.model.WRegret)
	copyFloat64SliceTo32(t.host.bRegret, t.model.BRegret)
	copyFloat64SliceTo32(t.host.wStrategy, t.model.WStrategy)
	copyFloat64SliceTo32(t.host.bStrategy, t.model.BStrategy)
	copyFloat64SliceTo32(t.host.wValue, t.model.WValue)

	if err := t.writeFloat32(t.buffers.w1, t.host.w1); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.b1, t.host.b1); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.w2, t.host.w2); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.b2, t.host.b2); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.wRegret, t.host.wRegret); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.bRegret, t.host.bRegret); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.wStrategy, t.host.wStrategy); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.bStrategy, t.host.bStrategy); err != nil {
		return err
	}
	if err := t.writeFloat32(t.buffers.wValue, t.host.wValue); err != nil {
		return err
	}
	b.weightsLoaded = true
	return nil
}

func copyFloat64SliceTo32(dst []float32, src []float64) {
	for i := range src {
		dst[i] = float32(src[i])
	}
}

func maskedSoftmax32(logits []float32, legalMask []float32, out []float32) {
	clear(out)
	maxLogit := float32(math.Inf(-1))
	valid := 0
	for i := range logits {
		if i >= len(legalMask) || legalMask[i] == 0 {
			continue
		}
		valid++
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}
	if valid == 0 {
		return
	}
	sum := 0.0
	for i := range logits {
		if i >= len(legalMask) || legalMask[i] == 0 {
			continue
		}
		out[i] = float32(math.Exp(float64(logits[i] - maxLogit)))
		sum += float64(out[i])
	}
	if sum == 0 {
		uniform := float32(1.0 / float64(valid))
		for i := range logits {
			if i < len(legalMask) && legalMask[i] > 0 {
				out[i] = uniform
			}
		}
		return
	}
	for i := range out {
		out[i] = out[i] / float32(sum)
	}
}

func sumFloat32(values []float32) float64 {
	total := 0.0
	for _, value := range values {
		total += float64(value)
	}
	return total
}

func sumColumn(values []float32, width int, column int) float64 {
	total := 0.0
	for idx := column; idx < len(values); idx += width {
		total += float64(values[idx])
	}
	return total
}

func cStringBytes(buf []byte) string {
	if idx := strings.IndexByte(string(buf), 0); idx >= 0 {
		return strings.TrimSpace(string(buf[:idx]))
	}
	return strings.TrimSpace(string(buf))
}
