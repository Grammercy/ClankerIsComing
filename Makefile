SHELL := /bin/bash

APP ?= pokemon-engine.exe
PKG ?= .
CROSS_CC ?= x86_64-w64-mingw32-gcc
CROSS_CXX ?= x86_64-w64-mingw32-g++
CROSS_TRIPLE ?= x86_64-w64-mingw32

# Optional Windows OpenCL SDK paths (leave empty to use system defaults).
OPENCL_INCLUDE ?=
OPENCL_LIB ?=

OPENCL_CFLAGS :=
ifeq ($(strip $(OPENCL_INCLUDE)),)
ifneq ($(wildcard /usr/$(CROSS_TRIPLE)/include/CL/cl.h),)
OPENCL_INCLUDE := /usr/$(CROSS_TRIPLE)/include
endif
endif
ifneq ($(strip $(OPENCL_INCLUDE)),)
OPENCL_CFLAGS += -I$(OPENCL_INCLUDE)
endif

OPENCL_LDFLAGS :=
ifneq ($(strip $(OPENCL_LIB)),)
OPENCL_LDFLAGS += -L$(OPENCL_LIB)
else ifneq ($(wildcard /usr/$(CROSS_TRIPLE)/lib/libOpenCL.dll.a),)
OPENCL_LDFLAGS += -L/usr/$(CROSS_TRIPLE)/lib
endif
OPENCL_LDFLAGS += -lOpenCL

.PHONY: help check-cross-toolchain windows-opencl

help:
	@echo "Targets:"
	@echo "  make windows-opencl   Build Windows binary with OpenCL support (GOOS=windows, -tags opencl)"
	@echo ""
	@echo "Config vars:"
	@echo "  CROSS_CC, CROSS_CXX   Cross compilers (default: x86_64-w64-mingw32-gcc/g++)"
	@echo "  OPENCL_INCLUDE        Optional Windows OpenCL include dir"
	@echo "  OPENCL_LIB            Optional Windows OpenCL lib dir containing OpenCL import library"
	@echo "  APP                   Output binary name (default: pokemon-engine.exe)"
	@echo "  PKG                   Package path to build (default: .)"

check-cross-toolchain:
	@command -v $(CROSS_CC) >/dev/null || { echo "Missing cross C compiler: $(CROSS_CC)"; exit 1; }
	@command -v $(CROSS_CXX) >/dev/null || { echo "Missing cross C++ compiler: $(CROSS_CXX)"; exit 1; }

windows-opencl: check-cross-toolchain
	@echo "Building $(APP) for Windows with OpenCL..."
	CGO_ENABLED=1 \
	GOOS=windows \
	GOARCH=amd64 \
	CC=$(CROSS_CC) \
	CXX=$(CROSS_CXX) \
	CGO_CFLAGS="$(OPENCL_CFLAGS)" \
	CGO_LDFLAGS="$(OPENCL_LDFLAGS)" \
	go build -tags opencl -o $(APP) $(PKG)
