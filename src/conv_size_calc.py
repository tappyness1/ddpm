
def get_output_size(input_size, kernel_size, stride = 1, num_times=1):
    def output_size(input_size):
        return ((input_size - kernel_size)/stride) + 1
    
    for _ in range(num_times):
        input_size = output_size(input_size)

    return int(input_size)

def get_trans_conv_out_size(input_size, kernel_size, stride = 1):
    return (input_size - 1) * stride - 2 + kernel_size + 1

if __name__ == "__main__":
    print (get_output_size(224, 3, 6))