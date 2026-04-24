CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
TARGET   = sis_simulation

$(TARGET): sis_simulation.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) sis_simulation.cpp

clean:
	rm -f $(TARGET)

.PHONY: clean
