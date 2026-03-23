TARGET = gpu-specs-check
SRC = gpu-specs-check.cu

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
