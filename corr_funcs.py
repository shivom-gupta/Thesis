import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


def corr_1_slow(configurations, r):
    c = 0
    R = configurations.shape[0]
    L = configurations.shape[1]
    for k in range(R):
        for i in range(L):
            c += np.sum(configurations[k, i] * configurations[k, int((i + r) % L)])
    return c / (R * L)

def corr_1(configurations, r):
    R, L = configurations.shape

    shifted_indices = ((np.arange(L) + r) % L).astype(int)

    product_sum = np.sum(configurations * configurations[:, shifted_indices], axis=1)
    return np.sum(product_sum) / (R * L)

def corr_2(configurations, r):
    R, L = configurations.shape

    shifted_indices = ((np.arange(L) + r) % L).astype(int)

    product_sum = (np.sum(configurations, axis=0)/R) * (np.sum(configurations[:, shifted_indices], axis=0)/R)
    
    second_term = np.sum(product_sum) / L
    cr_1 = corr_1(configurations, r)
    
    return cr_1 - second_term

def corr_3(configurations, r):
    return corr_2(configurations, r)/corr_2(configurations, 0)

def corr_4(configurations, r):
    R, L = configurations.shape

    shifted_indices = ((np.arange(L) + r) % L).astype(int)

    sum_prod = np.sum(configurations * configurations[:, shifted_indices], axis=1)/L
    sum_sq = (np.sum(configurations, axis=1)/L)**2
    
    return np.sum((sum_prod - sum_sq)/(1-sum_sq)) / R

def corr_5(configurations, r):
    R, L = configurations.shape
    cr_1 = corr_1(configurations, r)
    sum_sq = (np.sum(np.sum(configurations, axis=1)/L)/R)**2
    
    return cr_1 - sum_sq

def corr_6(configurations, r):
    return corr_5(configurations, r)/corr_5(configurations, 0)

def corr_function_parallel(configurations, rs, corr_function):
    results = np.zeros(len(rs))
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(corr_function, configurations, r): r for r in rs}
        
        for future in as_completed(futures):
            r = futures[future]
            results[np.where(rs == r)[0][0]] = future.result()

    return results

def worker(size, beta, corr_function, rs, preloaded_data):
    configurations, params = preloaded_data[(size, beta)]
    result = corr_function_parallel(configurations, rs, corr_function)
    return size, beta, corr_function.__name__, result

def parallelize_computations(sizes, betas, corr_functions, preloaded_data):
    computed_data = {size: {} for size in sizes}
    
    with ProcessPoolExecutor() as executor:
        tasks = []
        for size in sizes:
            rs = np.concatenate((np.arange(0, 10, 1), np.arange(10, size - 10, 10), np.arange(size - 10, size + 1, 1)))
            for beta in betas:
                for corr_function in corr_functions:
                    tasks.append(executor.submit(worker, size, beta, corr_function, rs, preloaded_data))

        for future in tasks:
            size, beta, corr_name, result = future.result()
            if beta not in computed_data[size]:
                computed_data[size][beta] = {}
            computed_data[size][beta][corr_name] = result

    return computed_data
