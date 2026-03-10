SHELL := /bin/bash

APP ?= pokemon-engine.exe
PKG ?= .
CROSS_CC ?= x86_64-w64-mingw32-gcc
CROSS_CXX ?= x86_64-w64-mingw32-g++

.PHONY: help build test check-cross-toolchain windows

help:
	@echo "Targets:"
	@echo "  make build     Build local binary"
	@echo "  make test      Run all tests"
	@echo "  make windows   Build Windows binary"
	@echo ""
	@echo "Config vars:"
	@echo "  APP            Output binary name (default: pokemon-engine.exe)"
	@echo "  PKG            Package path to build (default: .)"
	@echo "  CROSS_CC       Windows cross C compiler"
	@echo "  CROSS_CXX      Windows cross C++ compiler"

build:
	go build -o $(APP) $(PKG)

test:
	go test ./...

check-cross-toolchain:
	@command -v $(CROSS_CC) >/dev/null || { echo "Missing cross C compiler: $(CROSS_CC)"; exit 1; }
	@command -v $(CROSS_CXX) >/dev/null || { echo "Missing cross C++ compiler: $(CROSS_CXX)"; exit 1; }

windows: check-cross-toolchain
	CGO_ENABLED=1 \
	GOOS=windows \
	GOARCH=amd64 \
	CC=$(CROSS_CC) \
	CXX=$(CROSS_CXX) \
	go build -o $(APP) $(PKG)
