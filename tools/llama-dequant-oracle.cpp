// Research-only Class-Q helper. This is not linked into Kestrel-Q production.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"

static int hex_value(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

static bool parse_hex(const char *text, std::vector<std::uint8_t> &bytes) {
    const std::size_t length = std::strlen(text);
    if (length == 0 || (length % 2) != 0) {
        return false;
    }
    bytes.resize(length / 2);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        const int high = hex_value(text[index * 2]);
        const int low = hex_value(text[index * 2 + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        bytes[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return true;
}

static bool parse_type(const std::string &name, ggml_type &type) {
    if (name == "F32") type = GGML_TYPE_F32;
    else if (name == "BF16") type = GGML_TYPE_BF16;
    else if (name == "Q5_1") type = GGML_TYPE_Q5_1;
    else if (name == "Q8_0") type = GGML_TYPE_Q8_0;
    else if (name == "Q4_K") type = GGML_TYPE_Q4_K;
    else if (name == "Q5_K") type = GGML_TYPE_Q5_K;
    else if (name == "IQ4_NL") type = GGML_TYPE_IQ4_NL;
    else return false;
    return true;
}

int main(int argc, char **argv) {
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<std::uint8_t> packed;
    const ggml_type_traits *traits;
    std::vector<float> output;
    std::int64_t elements;

    if (argc != 3 || !parse_type(argv[1], type) ||
        !parse_hex(argv[2], packed)) {
        std::fprintf(stderr, "usage: llama-dequant-oracle TYPE PACKED_HEX\n");
        return 2;
    }
    traits = ggml_get_type_traits(type);
    if (traits == nullptr || traits->type_size == 0 || traits->blck_size <= 0 ||
        packed.size() % traits->type_size != 0) {
        std::fprintf(stderr, "unsupported or malformed block span\n");
        return 3;
    }
    elements = static_cast<std::int64_t>(
        (packed.size() / traits->type_size) *
        static_cast<std::size_t>(traits->blck_size));
    output.resize(static_cast<std::size_t>(elements));
    if (type == GGML_TYPE_F32) {
        std::memcpy(output.data(), packed.data(), packed.size());
    } else if (traits->to_float != nullptr) {
        traits->to_float(packed.data(), output.data(), elements);
    } else {
        std::fprintf(stderr, "type has no F32 decode trait\n");
        return 3;
    }

    std::printf("bits=");
    for (std::size_t index = 0; index < output.size(); ++index) {
        std::uint32_t bits;
        std::memcpy(&bits, &output[index], sizeof(bits));
        if (index != 0) std::putchar(',');
        std::printf("%08x", static_cast<unsigned int>(bits));
    }
    std::putchar('\n');
    return 0;
}
