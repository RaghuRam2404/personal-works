import time
import torch

#representation_format:
# (-1)^(s) * (1.fraction) * 2^(exponent-bias)

#storing no of bits
fp_32 = {
    "s": 1,
    "exponent": 8,
    "fraction": 23,
    "bias": 127
}

fp_16 = {
    "s": 1,
    "exponent": 5,
    "fraction": 10,
    "bias": 15
}

bf_16 = {
    "s": 1,
    "exponent": 8,
    "fraction": 7,
    "bias": 127
}


def float_limits(fmt):
    e, f, bias = fmt["exponent"], fmt["fraction"], fmt["bias"]
    # all-ones exponent is reserved for inf/nan, so the largest usable exponent is 2^e - 2
    max_exp = (2 ** e - 2) - bias
    max_val = (2 - 2 ** -f) * 2 ** max_exp      # 1.111...1 * 2^max_exp
    min_normal = 2.0 ** (1 - bias)              # 1.0 * 2^(1-bias), exponent field = 1
    min_subnormal = 2.0 ** (1 - bias - f)       # 0.000...1 * 2^(1-bias), exponent field = 0
    return max_val, -max_val, min_normal, min_subnormal


formats = {
    "fp32": (fp_32, torch.float32),
    "fp16": (fp_16, torch.float16),
    "bf16": (bf_16, torch.bfloat16),
}

for name, (fmt, dtype) in formats.items():
    max_val, min_val, min_normal, min_subnormal = float_limits(fmt)
    info = torch.finfo(dtype)
    print(f"{name}:")
    print(f"  max value           : {max_val:.6e}  (torch: {info.max:.6e})")
    print(f"  min value (lowest)  : {min_val:.6e}  (torch: {info.min:.6e})")
    print(f"  smallest +ve normal : {min_normal:.6e}  (torch: {info.tiny:.6e})")
    print(f"  smallest +ve subnorm: {min_subnormal:.6e}")
    assert max_val == info.max and min_val == info.min and min_normal == info.tiny


# Let's test the script again with the correct parsing (big endian or little endian issue?)
# Ah, struct.unpack('<f', struct.pack('>I', 0x3f800000)) ? No, frombuffer needs the right shape or array type
import numpy as np
import torch

def bin_to_fp32_correct(b_str):
    val = int(b_str, 2)
    # create a numpy int32 array, view it as float32
    return torch.from_numpy(np.array([val], dtype=np.uint32).view(np.float32))

def bin_to_fp16_correct(b_str):
    val = int(b_str, 2)
    return torch.from_numpy(np.array([val], dtype=np.uint16).view(np.float16))

def bin_to_bf16_correct(b_str):
    # BF16 can be viewed as the top 16 bits of a float32, so we pad it with 16 zero bits to get a float32 representation
    val = int(b_str + "0"*16, 2)
    fp32_tensor = torch.from_numpy(np.array([val], dtype=np.uint32).view(np.float32))
    return fp32_tensor.to(torch.bfloat16)

print("FP32 1.0:", bin_to_fp32_correct("00111111100000000000000000000000").item())
print("FP32 1.0:", bin_to_fp32_correct("10000000011111111111111111111111").item())

print("FP16 1.0:", bin_to_fp16_correct("0011110000000000").item())
print("BF16 1.0:", bin_to_bf16_correct("0011111110000000").item())


# Ranges of each format, decoded from their bit patterns
def range_bits(fmt):
    e, f = fmt["exponent"], fmt["fraction"]
    max_exp = "1" * (e - 1) + "0"      # all ones is reserved for inf/nan
    min_exp = "0" * (e - 1) + "1"      # exponent field = 1 (smallest normal)
    return {
        "max":           "0" + max_exp + "1" * f,
        "min (lowest)":  "1" + max_exp + "1" * f,
        "min normal":    "0" + min_exp + "0" * f,
        "min subnormal": "0" + "0" * e + "0" * (f - 1) + "1",
    }

decoders = {
    "FP32": (fp_32, bin_to_fp32_correct),
    "FP16": (fp_16, bin_to_fp16_correct),
    "BF16": (bf_16, bin_to_bf16_correct),
}

for name, (fmt, decode) in decoders.items():
    print(f"\n{name} range:")
    for label, bits in range_bits(fmt).items():
        print(f"  {label:<14}: {bits}  ->  {decode(bits).item():.6e}")


# How many values lie in [0, 1)?
# Positive floats sort like their bit patterns as integers, so the count is
# the integer value of the bit pattern of 1.0 = (exponent field = bias, fraction = 0)
print("\nValues in [0, 1):")
for name, (fmt, decode) in decoders.items():
    e, f, bias = fmt["exponent"], fmt["fraction"], fmt["bias"]
    one_bits = format(bias, f"0{e}b") + "0" * f
    assert decode("0" + one_bits).item() == 1.0
    count = int("0" + one_bits, 2)
    total = 2 ** (fmt["s"] + e + f)
    print(f"  {name}: {count:>13,}  ({count / total * 100:.1f}% of all {total:,} bit patterns)")



# How many values lie in [0, 0.125)? (0, all subnormals, and every normal binade below 2^-3)
# Bit pattern of 0.125 = 2^-3 -> exponent field = bias - 3; everything below it counts
print("\nValues in [0, 0.125):")
for name, (fmt, decode) in decoders.items():
    e, f, bias = fmt["exponent"], fmt["fraction"], fmt["bias"]
    eighth_bits = "0" + format(bias - 3, f"0{e}b") + "0" * f
    assert decode(eighth_bits).item() == 0.125
    count = int(eighth_bits, 2)
    print(f"  {name}: {count:>13,}  ({count / int('0' + format(bias, f'0{e}b') + '0' * f, 2) * 100:.1f}% of the values in [0, 1))")

# How many values lie in [0.5, 1)? (one binade: same exponent, only the fraction changes)
# Every binade [2^k, 2^(k+1)) holds exactly 2^fraction values
print("\nValues in [0.5, 1):")
for name, (fmt, decode) in decoders.items():
    e, f, bias = fmt["exponent"], fmt["fraction"], fmt["bias"]
    half_bits = "0" + format(bias - 1, f"0{e}b") + "0" * f   # 0.5 -> exponent field = bias - 1
    one_bits = "0" + format(bias, f"0{e}b") + "0" * f        # 1.0 -> exponent field = bias
    assert decode(half_bits).item() == 0.5
    count = int(one_bits, 2) - int(half_bits, 2)
    assert count == 2 ** f
    print(f"  {name}: {count:>9,}  (= 2^{f})")


# How many values lie in [0.25, 1)? (two binades: [0.25, 0.5) and [0.5, 1))
print("\nValues in [0.25, 1):")
for name, (fmt, decode) in decoders.items():
    e, f, bias = fmt["exponent"], fmt["fraction"], fmt["bias"]
    quarter_bits = "0" + format(bias - 2, f"0{e}b") + "0" * f   # 0.25 -> exponent field = bias - 2
    one_bits = "0" + format(bias, f"0{e}b") + "0" * f
    assert decode(quarter_bits).item() == 0.25
    count = int(one_bits, 2) - int(quarter_bits, 2)
    assert count == 2 * 2 ** f
    print(f"  {name}: {count:>10,}  (= 2 * 2^{f})")
