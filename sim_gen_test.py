bypass_first_fwd = False
microbatch_num_per_batch = 4



def get_model_idx_for_pp(seq_id, pass_agg=True):
    '''
    pass_agg 标志了如何排序。
    如果为真，则microbatch之间是类似于“fwd1-fwd2-fwd3-fwd4 - bkwd1-bkwd2-bkwd3-bkwd4”的形式排列
    否则，microbatch之间是类似于“fwd1-bkwd1 - fwd2-bkwd2 - fwd3-bkwd3 - fwd4-bkwd4”的形式排列
    '''
    is_first_pass = False
    is_last_pass = False
    if bypass_first_fwd == True:
        assert pass_agg == True, f"bypass_first_fwd cannot obtained without pass_agg"
        seq_id += microbatch_num_per_batch
    b_idx = seq_id // (2 * microbatch_num_per_batch)
    if pass_agg:
        mb_idx = seq_id % microbatch_num_per_batch
        if mb_idx == 0:
            is_first_pass = True
        if mb_idx == microbatch_num_per_batch - 1:
            is_last_pass = True
        pass_type = 'FWD' if seq_id - b_idx * (2 * microbatch_num_per_batch) < microbatch_num_per_batch else 'BKWD'
    else:
        mb_idx = (seq_id - b_idx * (2 * microbatch_num_per_batch)) // 2
        pass_type = 'FWD' if (seq_id - b_idx * (2 * microbatch_num_per_batch)) % 2 == 0 else 'BKWD'
        if mb_idx == 0:
            is_first_pass = True
        if mb_idx == microbatch_num_per_batch - 1:
            is_last_pass = True
        
    return b_idx, mb_idx, pass_type, is_first_pass, is_last_pass


if __name__ == '__main__':
    passes = 12
    for i in range (passes):
        print(get_model_idx_for_pp(i))
    print("")
    for i in range (passes):
        print(get_model_idx_for_pp(i, False))
    print("")

    bypass_first_fwd = True
    
    for i in range (passes):
        print(get_model_idx_for_pp(i))
    print("")
    for i in range (passes):
        print(get_model_idx_for_pp(i, False))
    print("")