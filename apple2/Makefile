# Generic CC65 TARGETS makefile++
#
# Set the TARGETS and PROGRAM values as required.
# See makefiles/build.mk for details on directory structure for src files and how to add custom extensions to the build.

TARGETS = apple2enh
PROGRAM := yail

SUB_TASKS := clean disk test release
.PHONY: all help $(SUB_TASKS)

all:
	@for target in $(TARGETS); do \
		echo "-------------------------------------"; \
		echo "Building $$target"; \
		echo "-------------------------------------"; \
		$(MAKE) --no-print-directory -f ../../fujinet-build-tools/makefiles/build.mk CURRENT_TARGET=$$target PROGRAM=$(PROGRAM) $(MAKECMDGOALS); \
	done

# if disk images were built show them
	@if [ -d ./dist ]; then \
		echo "Contents of dist:"; \
		ls -1 ./dist; \
	fi

	@echo ""; 
	@echo "Build Complete"; 
	@echo ""; 


$(SUB_TASKS): _do_all
$(SUB_TASKS):
	@:

_do_all: all

help:
	@echo "Makefile for $(PROGRAM)"
	@echo ""
	@echo "Available tasks:"
	@echo "all       - do all compilation tasks, create app in build directory"
	@echo "clean     - remove all build artifacts"
	@echo "release   - create a release of the executable in the build/ dir"
	@echo "disk      - generate platform specific disk images in dist/ dir"
	@echo "deploy    - deploy dist/ images to a local TNFS or other server platform"
	@echo "test      - run application in emulator for given platform."
	@echo "            specific platforms may expose additional variables to run with"
	@echo "            different emulators, see makefiles/custom-<platform>.mk"
	