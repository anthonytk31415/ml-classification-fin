import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt



# Calculate the value counts and percentages

def plot(df, var, colname): 

    y_counts = df[var].value_counts().reset_index()
    y_counts.columns = [var, colname]
    y_counts['percentage'] = (y_counts[colname] / y_counts[colname].sum() * 100).round(0)  # No decimals for percentage

    # Create a nicer plot with percentages
    plt.figure(figsize=(4, 3))
    ax = sns.barplot(data=y_counts, x=var, y=colname, palette='viridis')
    plt.title("Distribution of Variable: {}".format(var), fontsize=12, fontweight='bold')  # Smaller font size
    plt.xlabel('Target Value', fontsize=10)  # Smaller font size
    plt.ylabel('')  # Remove the y-axis label
    ax.set_yticklabels([])  # Remove y-axis ticks

    # Add count labels on top of each bar
    for i, row in enumerate(y_counts.itertuples()):
        ax.text(i, row.count + (y_counts[colname].max() * 0.02),  # Slight offset above bar
                f'{row.count}', 
                ha='center', fontsize=9, fontweight='bold')  # No "count" label, just the number
        
        # Add percentage inside the bar with dark gray color (no decimals)
        ax.text(i, row.count/2,  # Middle of the bar
                f'{int(row.percentage)}%', 
                ha='center', fontsize=8, fontweight='bold', color='darkgray')  # Dark gray color for percentage and smaller font size

    # Enhance the plot
    sns.despine(left=True, bottom=True)
    plt.xticks(fontsize=8)  # Reduce font size of x-axis labels
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