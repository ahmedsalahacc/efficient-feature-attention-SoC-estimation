import sys

if len(sys.argv) != 3:
    print(
        "Usage: python convert_tflite_to_c_array.py <input_model.tflite> <output_header.h>"
    )
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:
    f.write(f"unsigned char {input_file.split('.')[0]}[] = {{\n")
    for i, byte in enumerate(data):
        f.write(f" 0x{byte:02x},")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"unsigned int {input_file.split('.')[0]}_len = {len(data)};\n")
