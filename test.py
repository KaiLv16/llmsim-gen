import pandas as pd
import numpy as np
import itertools

def find_feasible_partition(df, n, M):
    # Step 1: Create a list of tuples (output_weight, index)
    weight_idx = list(df[['output_weight', 'index']].itertuples(index=False, name=None))
    weight_idx = [t for t in weight_idx if t[1] != 0]
    # Step 2: Sort the list by output_weight
    weight_idx.sort()
    # print(weight_idx)
    print("weight_idx.len =", len(weight_idx) + 1)

    k = n
    
    while True:
        print("searching in", k+1, "intervals.")
        # Step 3: Take the first k tuples where index is not 0
        k_tuples = weight_idx[:k]
        
        # Step 4: Calculate the number of elements to remove
        remove_count = k - n
        
        if remove_count <= 0:
            # If k <= n, we don't need to remove any elements
            combinations = [k_tuples]
        else:
            # Otherwise, generate all combinations of removing remove_count elements
            combinations = list(itertools.combinations(k_tuples, n))
        
        for comb in combinations:
            indices = [t[1] for t in comb]
            
            # Create partitions
            partitions = sorted([-1] + indices + [len(df)-1])
            # if k <= n + 1:
            #     print(partitions)

            feasible = True
            
            # Check sums of partitions
            for i in range(len(partitions) - 1):
                segment_sum = df.loc[partitions[i] + 1 : partitions[i+1], 'arr'].sum()
                if segment_sum > M:
                    feasible = False
                    break
            
            if feasible:
                # Update the original DataFrame with group information
                df['grp_id'] = -1
                df['grp_sum'] = 0
                df['grp_last_weight'] = 0
                
                for i in range(len(partitions) - 1):
                    print("[", partitions[i]+1, ", ", partitions[i+1], "]")
                    df.loc[partitions[i]+1:partitions[i+1], 'grp_id'] = i
                    grp_sum = df.loc[partitions[i]+1:partitions[i+1], 'arr'].sum()
                    df.loc[partitions[i]+1:partitions[i+1], 'grp_sum'] = grp_sum
                    grp_last_weight = df.loc[partitions[i+1], 'output_weight']
                    df.loc[partitions[i]+1:partitions[i+1], 'grp_last_weight'] = grp_last_weight
                
                print(f"Feasible partition found with k={k}: {partitions}")
                print(df)
                return df
        
        # Increment k and try again
        k += 1

# Example usage
# pd.set_option('display.max_rows', None)

data = {
    'index': list(range(100)),
    'arr': [np.random.randint(1, 101) for _ in range(100)],
    'output_weight': [np.random.randint(1, 101) for _ in range(100)],
    'grp_sum': 0
}
df = pd.DataFrame(data)

n = 15 - 1
M = 1000

result_df = find_feasible_partition(df, n, M)
