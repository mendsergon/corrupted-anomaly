# anomaly — Makefile
#
# Targets:
#   make          release build
#   make debug    -O0 -g with sanitizers
#   make run      build and run
#   make deps     install dev packages (Arch, Debian/Ubuntu, Fedora)
#   make clean    remove build artefacts

CXX       ?= g++
CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS   := -lOpenCL -lGL -lglfw -lm

DEBUG_CXXFLAGS := -O0 -g3 -std=c++17 -Wall -Wextra -pedantic -fsanitize=address,undefined
DEBUG_LDFLAGS  := $(LDFLAGS) -fsanitize=address,undefined

SRC_DIR := src
OBJ_DIR := build
BIN     := anomaly

INCLUDES := -I$(SRC_DIR)

SRCS := $(SRC_DIR)/main.cpp \
        $(SRC_DIR)/anomaly.cpp

OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

KERNELS := $(SRC_DIR)/anomaly.cl

.PHONY: all debug run clean deps check-assets

all: $(BIN)

$(BIN): $(OBJS) | check-assets
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)
	@echo "Built $(BIN). Run from project root so src/anomaly.cl resolves."

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

-include $(DEPS)

check-assets:
	@for k in $(KERNELS); do \
	  if [ ! -f "$$k" ]; then echo "ERROR: kernel $$k missing"; exit 1; fi; \
	done

debug: CXXFLAGS := $(DEBUG_CXXFLAGS)
debug: LDFLAGS  := $(DEBUG_LDFLAGS)
debug: clean all

run: $(BIN)
	@./$(BIN)

clean:
	rm -rf $(OBJ_DIR) $(BIN)

deps:
	@if [ -f /etc/arch-release ]; then \
	  echo "[deps] Arch Linux"; \
	  sudo pacman -S --needed --noconfirm opencl-headers opencl-icd-loader glfw mesa; \
	elif [ -f /etc/debian_version ]; then \
	  echo "[deps] Debian / Ubuntu"; \
	  sudo apt-get update && sudo apt-get install -y \
	    opencl-headers ocl-icd-opencl-dev libglfw3-dev libgl1-mesa-dev; \
	elif [ -f /etc/fedora-release ]; then \
	  echo "[deps] Fedora"; \
	  sudo dnf install -y opencl-headers ocl-icd-devel glfw-devel mesa-libGL-devel; \
	else \
	  echo "Unknown distro. Install OpenCL headers + ICD, GLFW, and libGL manually."; \
	  exit 1; \
	fi
