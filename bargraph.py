import matplotlib.pyplot as plt
import os

def plot_bargraph(data, save_dir='plots'):
   
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Function to shorten labels for better display
    def shorten_label(label):
        parts = label.split()
        if "sentiment" in label:
            return "Sent. " + parts[-2]
        elif "paraphrase" in label:
            return "Para. " + parts[-2]
        elif "sts" in label:
            return "STS " + parts[-2]
        else:
            return "Other"

    # Separate data by assessment type and shorten labels
    sentiment_data = [(shorten_label(item[0]), item[1], item[2]) for item in data if "sentiment" in item[0]]
    para_data = [(shorten_label(item[0]), item[1], item[2]) for item in data if "paraphrase" in item[0]]
    sts_data = [(shorten_label(item[0]), item[1], item[2]) for item in data if "sts" in item[0]]
    
    # Define colors for each category
    colors = {'SimCSE': 'skyblue', 'SimCSE + Gaussian': 'lightgreen', 'multitask': 'lightcoral'}  # Added 'new_category' color
    
    # Plotting function with aesthetic improvements, adjusted y-axis, and saving functionality
    def plot_individual_graph(assessment_data, title):
        labels, scores, categories = zip(*assessment_data)
        min_score = min(scores)
        fig, ax = plt.subplots()
        for label, score, category in zip(labels, scores, categories):
            ax.bar(label, score, color=colors.get(category, 'grey'))  # Use 'grey' as default color for unspecified categories
        ax.set_title(f'{title} Scores by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Score')
        ax.set_ylim([min_score * 0.9, max(scores) * 1.1])  # Adjust y-axis to start close to the smallest score
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_scores.png'))
        plt.close()
    
    # Plot each assessment type with shortened labels, adjusted y-axis, and save them
    plot_individual_graph(sentiment_data, 'Sentiment')
    plot_individual_graph(para_data, 'Paraphrase')
    plot_individual_graph(sts_data, 'STS')

data = [
    ("sentiment acc with Unsupervised SimCSE", .460, 'SimCSE'),
    ("paraphrase acc with Unsupervised SimCSE", .780, 'SimCSE'),
    ("sts corr with Unsupervised SimCSE", .337, 'SimCSE'),
    ("sentiment acc with Unsupervised SimCSE + Gaussian dropout", .431, 'SimCSE + Gaussian'),
    ("paraphrase acc with Unsupervised SimCSE + Gaussian dropout", .776, 'SimCSE + Gaussian'),
    ("sts corr with Unsupervised SimCSE + Gaussian dropout", .329, 'SimCSE + Gaussian'),
    ("sentiment acc with multitask :", .496, 'multitask'),
    ("paraphrase acc with miltitask :", .777, 'multitask'),
    ("sts corr with multitask :", .338, 'multitask'),
]

plot_bargraph(data)

