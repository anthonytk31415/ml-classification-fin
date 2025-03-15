import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt



# Calculate the value counts and percentages

def plot(df, var, colname): 

    y_counts = df[var].value_counts().reset_index()
    y_counts.columns = [var, colname]
    y_counts['percentage'] = (y_counts[colname] / y_counts[colname].sum() * 100).round(2)

    # Create a nicer plot with percentages
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=y_counts, x=var, y=colname, palette='viridis')
    plt.title("Distribution of Variable: {}".format(var), fontsize=16, fontweight='bold')
    plt.xlabel('Target Value', fontsize=14)
    plt.ylabel(colname, fontsize=14)

    # Add count and percentage labels on top of each bar
    for i, row in enumerate(y_counts.itertuples()):
        ax.text(i, row.count + (y_counts[colname].max() * 0.02),  # Slight offset above bar
                f'Count: {row.count}', 
                ha='center', fontsize=12, fontweight='bold')
        
        # Add percentage inside the bar
        ax.text(i, row.count/2,  # Middle of the bar
                f'{row.percentage:.2f}%', 
                ha='center', fontsize=14, fontweight='bold', color='white')

    # Add a horizontal line representing the total count
    total_count = y_counts[colname].sum()
    plt.axhline(y=total_count, color='red', linestyle='--', alpha=0.5)
    plt.text(plt.xlim()[1] * 0.95, total_count * 1.05, 
            f'Total: {total_count}', 
            ha='right', color='red', fontsize=12)

    # Enhance the plot
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    # Print class distribution table
    print("\nClass Distribution:")
    distribution_table = pd.DataFrame({
        'Class': y_counts[var],
        colname: y_counts[colname],
        'Percentage (%)': y_counts['percentage']
    })
    print(distribution_table)