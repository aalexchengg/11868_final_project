from datasets import load_dataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from statistics import mean, median, stdev, mode
from scipy.stats import zscore
import numpy as np

def import_data():
    print("importing data...")
    java_ds = load_dataset("google/code_x_glue_cc_code_completion_token", "java")
    python_ds = load_dataset("google/code_x_glue_cc_code_completion_token", "python")

    java_train = java_ds['train']
    python_train = python_ds['train']

    java_val = java_ds['validation']
    python_test = python_ds['test']
    print("done")
    return java_train, python_train, java_val, python_test


def grab_lengths(data, name):
    code_examples = data['code']
    res = []
    for code in tqdm(code_examples, desc=f'reading {name}'):
        res.append(len(code))
    return res

def histogram(lens, name):
    # plotting without outliers because the outliers are very large
    z_scores = zscore(lens)
    threshold = 1 # Define threshold for outliers
    filtered_lens = np.array(lens)[np.abs(z_scores) < threshold]
    print(f"excluding {len(lens) - len(filtered_lens)} lengths from {name} as they are outliers, which is {(len(lens) - len(filtered_lens)) * 100.0 / (len(lens))}%")

    plt.hist(filtered_lens)
    # plt.hist(lens)
    plt.title(f"Histogram of Lengths of Code Examples for {name}")
    # plt.title(f"Histogram of Lengths of Code Examples for {name}, UNFILTERED")
    plt.xlabel("# of Tokens")
    plt.savefig(f"{name}_histogram.png")
    plt.close()

def summary(lens):
    return mean(lens), stdev(lens), median(lens), mode(lens)

if __name__ == '__main__':
    java_train, python_train, java_val, python_test = import_data()

    jt_lens = grab_lengths(java_train, 'java_train')
    jv_lens = grab_lengths(java_val, 'java_val')
    pt_lens = grab_lengths(python_train, 'python_train')
    pv_lens = grab_lengths(python_test, 'python_val')

    histogram(jt_lens, 'java_train')
    histogram(jv_lens, 'java_val')
    histogram(pt_lens, 'python_train')
    histogram(pv_lens, 'python_val')

    with open("number_report.txt", "w") as f:

        jt_mean, jt_std, jt_median, jt_mode = summary(jt_lens)
        f.write(f"[JAVA TRAINING SET]\nmean: {jt_mean}, standard deviation: {jt_std}, median: {jt_median}, mode: {jt_mode}\n")
        jv_mean, jv_std, jv_median, jv_mode = summary(jv_lens)
        f.write(f"[JAVA VALIDATION SET]\nmean: {jv_mean}, standard deviation {jv_std}, median: {jv_median}, mode: {jv_mode}\n")

        pt_mean, pt_std, pt_median, pt_mode = summary(pt_lens)
        f.write(f"[PYTHON TRAINING SET]\nmean: {pt_mean}, standard deviation: {pt_std}, median: {pt_median}, mode: {pt_mode}\n")
        pv_mean, pv_std, pv_median, pv_mode = summary(pv_lens)
        f.write(f"[PYTHON VALIDATION SET]\nmean: {pv_mean}, standard deviation {pv_std}, median: {pv_median}, mode: {pv_mode}\n")

    f.close()
