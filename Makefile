SHELL := /bin/bash

APP ?= pokemon-engine.exe
PKG ?= .
CROSS_CC ?= x86_64-w64-mingw32-gcc
CROSS_CXX ?= x86_64-w64-mingw32-g++
CROSS_PKG_CONFIG ?= x86_64-w64-mingw32-pkg-config
GO_BUILD_FLAGS ?= -trimpath -buildvcs=false
GO_LDFLAGS ?= -s -w
WINDOWS_LDFLAGS ?= -s -w -linkmode external -extldflags '-static-libgcc -static-libstdc++'

.PHONY: help build build-local build-local-opencl test check-cross-toolchain check-cross-opencl windows

help:
	@echo "Targets:"
	@echo "  make build       Alias for make windows"
	@echo "  make build-local Build local binary"
	@echo "  make build-local-opencl Build local binary with OpenCL tags"
	@echo "  make test         Run all tests"
	@echo "  make windows      Build Windows binary (single target: self-contained + OpenCL/ROCm-compatible)"
	@echo ""
	@echo "Config vars:"
	@echo "  APP            Output binary name (default: pokemon-engine.exe)"
	@echo "  PKG            Package path to build (default: .)"
	@echo "  CROSS_CC       Windows cross C compiler"
	@echo "  CROSS_CXX      Windows cross C++ compiler"
	@echo "  CROSS_PKG_CONFIG Windows cross pkg-config binary"

build:
	@$(MAKE) windows

build-local:
	go build -o $(APP) $(PKG)

build-local-opencl:
	CGO_ENABLED=1 go build -tags "opencl" -o $(APP) $(PKG)

test:
	go test ./...

check-cross-toolchain:
	@command -v $(CROSS_CC) >/dev/null || { echo "Missing cross C compiler: $(CROSS_CC)"; exit 1; }
	@command -v $(CROSS_CXX) >/dev/null || { echo "Missing cross C++ compiler: $(CROSS_CXX)"; exit 1; }

windows: check-cross-toolchain check-cross-opencl
	CGO_ENABLED=1 \
	GOOS=windows \
	GOARCH=amd64 \
	CC=$(CROSS_CC) \
	CXX=$(CROSS_CXX) \
	PKG_CONFIG=$(CROSS_PKG_CONFIG) \
	go build $(GO_BUILD_FLAGS) -tags "opencl netgo osusergo" -ldflags "$(WINDOWS_LDFLAGS)" -o $(APP) $(PKG)

check-cross-opencl:
	@command -v $(CROSS_PKG_CONFIG) >/dev/null || { echo "Missing cross pkg-config: $(CROSS_PKG_CONFIG)"; exit 1; }
	@PKG_CONFIG=$(CROSS_PKG_CONFIG) $(CROSS_PKG_CONFIG) --exists OpenCL || { echo "Missing OpenCL pkg-config entry for cross toolchain"; exit 1; }
