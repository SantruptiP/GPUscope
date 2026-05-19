# gpuscope build.
#
# Targets:
#   make            -- build gpuscope binary and tests
#   make test       -- run all tests
#   make tsan       -- run ring buffer test under ThreadSanitizer (proves
#                      the lock-free contract holds; expect zero races)
#   make clean
#
# We're not using CMake on purpose: it's overkill for ~6 files and a Makefile
# makes the build steps legible. Move to CMake once we add CUPTI integration.

CXX      ?= g++
CXXFLAGS  = -std=c++20 -Wall -Wextra -Wpedantic -Wshadow -O2 -pthread -Iinclude
LDFLAGS   = -pthread

BIN_DIR    = build
BIN        = $(BIN_DIR)/gpuscope
TEST_RING  = $(BIN_DIR)/test_ring_buffer
TEST_DET   = $(BIN_DIR)/test_detector

.PHONY: all test tsan clean

all: $(BIN) $(TEST_RING) $(TEST_DET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN): src/main.cc $(BIN_DIR) include/gpuscope/*.h include/gpuscope/detectors/*.h
	$(CXX) $(CXXFLAGS) src/main.cc -o $@ $(LDFLAGS)

$(TEST_RING): tests/test_ring_buffer.cc $(BIN_DIR) include/gpuscope/ring_buffer.h
	$(CXX) $(CXXFLAGS) tests/test_ring_buffer.cc -o $@ $(LDFLAGS)

$(TEST_DET): tests/test_detector.cc $(BIN_DIR) include/gpuscope/*.h include/gpuscope/detectors/*.h
	$(CXX) $(CXXFLAGS) tests/test_detector.cc -o $@ $(LDFLAGS)

test: $(TEST_RING) $(TEST_DET)
	@./$(TEST_RING)
	@./$(TEST_DET)

# ThreadSanitizer build of the ring buffer test. The test pushes 200k items
# across two threads; if our memory orderings are wrong, TSan will fire.
# Note: TSan adds substantial overhead (~5x), so this is for verification,
# not perf measurement.
tsan: $(BIN_DIR)
	$(CXX) -std=c++20 -O1 -g -fsanitize=thread -pthread -Iinclude \
	  tests/test_ring_buffer.cc -o $(BIN_DIR)/test_ring_buffer_tsan
	./$(BIN_DIR)/test_ring_buffer_tsan

clean:
	rm -rf $(BIN_DIR)
