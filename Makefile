SHELL := /bin/bash

APP ?= pokemon-engine.exe
PKG ?= .
CROSS_CC ?= x86_64-w64-mingw32-gcc
CROSS_CXX ?= x86_64-w64-mingw32-g++
CROSS_PKG_CONFIG ?= x86_64-w64-mingw32-pkg-config

.PHONY: help build build-local test check-cross-toolchain check-cross-opencl windows windows-opencl

help:
	@echo "Targets:"
	@echo "  make build       Cross-compile Windows binary"
	@echo "  make build-local Build local binary"
	@echo "  make test         Run all tests"
	@echo "  make windows      Build Windows binary (CPU backend, static/pure-Go)"
	@echo "  make windows-opencl Build Windows binary with OpenCL backend"
	@echo ""
	@echo "Config vars:"
	@echo "  APP            Output binary name (default: pokemon-engine.exe)"
	@echo "  PKG            Package path to build (default: .)"
	@echo "  CROSS_CC       Windows cross C compiler (OpenCL build only)"
	@echo "  CROSS_CXX      Windows cross C++ compiler (OpenCL build only)"
	@echo "  CROSS_PKG_CONFIG Windows cross pkg-config binary (OpenCL build only)"

build:
	@$(MAKE) windows

build-local:
	go build -o $(APP) $(PKG)

test:
	go test ./...

check-cross-toolchain:
	@command -v $(CROSS_CC) >/dev/null || { echo "Missing cross C compiler: $(CROSS_CC)"; exit 1; }
	@command -v $(CROSS_CXX) >/dev/null || { echo "Missing cross C++ compiler: $(CROSS_CXX)"; exit 1; }

windows:
	CGO_ENABLED=0 \
	GOOS=windows \
	GOARCH=amd64 \
	go build -o $(APP) $(PKG)

check-cross-opencl:
	@command -v $(CROSS_PKG_CONFIG) >/dev/null || { echo "Missing cross pkg-config: $(CROSS_PKG_CONFIG)"; exit 1; }
	@PKG_CONFIG=$(CROSS_PKG_CONFIG) $(CROSS_PKG_CONFIG) --exists OpenCL || { echo "Missing OpenCL pkg-config entry for cross toolchain"; exit 1; }

windows-opencl: check-cross-toolchain check-cross-opencl
	CGO_ENABLED=1 \
	GOOS=windows \
	GOARCH=amd64 \
	CC=$(CROSS_CC) \
	CXX=$(CROSS_CXX) \
	PKG_CONFIG=$(CROSS_PKG_CONFIG) \
	go build -tags opencl -o $(APP) $(PKG)
