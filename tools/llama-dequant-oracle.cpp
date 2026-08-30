// Research-only Class-Q helper. This is not linked into Kestrel-Q production.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

static int dequant_file(int argc, char **argv) {
    ggml_type type = GGML_TYPE_COUNT;
    const ggml_type_traits *traits;
    char *end = nullptr;
    unsigned long long offset;
    unsigned long long elements;
    std::size_t packed_bytes;
    std::vector<std::uint8_t> packed;
    std::vector<float> output;
    std::FILE *input = nullptr;
    std::FILE *result = nullptr;
    if (argc != 7 || std::strcmp(argv[1], "dequant-file") != 0 ||
        !parse_type(argv[2], type))
        return 2;
    offset = std::strtoull(argv[4], &end, 10);
    if (end == argv[4] || *end != '\0') return 2;
    end = nullptr;
    elements = std::strtoull(argv[5], &end, 10);
    if (end == argv[5] || *end != '\0' || elements == 0)
        return 2;
    traits = ggml_get_type_traits(type);
    if (traits == nullptr || traits->type_size == 0 ||
        traits->blck_size <= 0 ||
        elements % static_cast<unsigned long long>(traits->blck_size) != 0)
        return 3;
    if (elements / static_cast<unsigned long long>(traits->blck_size) >
        static_cast<unsigned long long>(SIZE_MAX / traits->type_size))
        return 3;
    packed_bytes = static_cast<std::size_t>(
        elements / static_cast<unsigned long long>(traits->blck_size)) *
        traits->type_size;
    if (packed_bytes > 256U * 1024U * 1024U ||
        elements > static_cast<unsigned long long>(SIZE_MAX / sizeof(float)))
        return 3;
    if (fopen_s(&input, argv[3], "rb") != 0 || input == nullptr)
        return 4;
    if (_fseeki64(input, static_cast<__int64>(offset), SEEK_SET) != 0) {
        std::fclose(input);
        return 4;
    }
    packed.resize(packed_bytes);
    if (std::fread(packed.data(), 1, packed_bytes, input) != packed_bytes) {
        std::fclose(input);
        return 4;
    }
    std::fclose(input);
    output.resize(static_cast<std::size_t>(elements));
    if (type == GGML_TYPE_F32)
        std::memcpy(output.data(), packed.data(), packed.size());
    else if (traits->to_float != nullptr)
        traits->to_float(packed.data(), output.data(),
                         static_cast<std::int64_t>(elements));
    else
        return 3;
    if (fopen_s(&result, argv[6], "wb") != 0 || result == nullptr)
        return 4;
    if (std::fwrite(output.data(), sizeof(float), output.size(), result) !=
        output.size()) {
        std::fclose(result);
        return 4;
    }
    std::fclose(result);
    std::printf("elements=%llu packed_bytes=%zu\n", elements, packed_bytes);
    return 0;
}

int main(int argc, char **argv) {
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<std::uint8_t> packed;
    const ggml_type_traits *traits;
    std::vector<float> output;
    std::int64_t elements;

    if (argc >= 2 && std::strcmp(argv[1], "dequant-file") == 0)
        return dequant_file(argc, argv);
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
